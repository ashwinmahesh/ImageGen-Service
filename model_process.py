"""One reusable model child at a time; the HTTP process never loads GPU weights."""
import os
import logging
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from runtime_protocol import send,receive
from queue_storage import queue_database
from job_logs import record_event


class ModelProcessError(RuntimeError):
    def __init__(self,message,status=500):
        super().__init__(message);self.http_status=status


class ModelProcess:
    def __init__(self,project,*,loras_enabled=True,llm_enabled=True,stage=lambda value:None,command=None):
        self.project=Path(project);self.loras_enabled=loras_enabled;self.llm_enabled=llm_enabled
        self.stage=stage;self.command=command;self.process=None;self.channel=None;self.active_model=None;self.loaded=False;self.closing=False

    def select(self,model):
        if not self.closing and self.process is not None and self.process.poll() is None and self.active_model==model:return
        self.close()
        self.stage('loading_model')
        channel,child=socket.socketpair()
        environment={**os.environ,'IMAGE_MODEL':model,'PYTHONUNBUFFERED':'1'}
        folder=self.project/'state'/'model-process';folder.mkdir(parents=True,exist_ok=True)
        try:
            with (folder/'server.log').open('ab') as output:
                command=self.command or [sys.executable,str(self.project/'model_child.py')]
                self.process=subprocess.Popen([*command,'--socket-fd',str(child.fileno())],cwd=self.project,env=environment,
                    stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,pass_fds=(child.fileno(),),start_new_session=True)
            self.channel=channel;self.active_model=model
        except BaseException:
            channel.close();raise
        finally:child.close()

    def execute(self,job):
        try:
            record_event(self.project,job,'Selecting runtime: '+job['request']['model'])
            self.select(job['request']['model'])
            data={key:value for key,value in job.items() if key!='_inherited_fds'}
            send(self.channel,{'action':'execute','database_path':str(queue_database(self.project)),'job':data,'llm_enabled':self.llm_enabled,'loras_enabled':self.loras_enabled},job.get('_inherited_fds',()))
            while True:
                event,fds=receive(self.channel)
                for descriptor in fds:os.close(descriptor)
                if event['event']=='stage':self.stage(event['stage'])
                elif event['event']=='loaded':self.loaded=True
                elif event['event']=='result':return event['result']
                elif event['event']=='error':raise ModelProcessError(event['message'],event.get('http_status',500))
                else:raise RuntimeError('Unknown model process event')
        except BaseException as error:
            record_event(self.project,job,'Runtime execution failed',error=True)
            if not getattr(error,'fatal',False):
                try:self.close()
                except BaseException:
                    record_event(self.project,job,'Runtime cleanup failed',error=True);raise
            raise

    @staticmethod
    def group_has_live_members(group):
        # A signal permission error is not evidence of exit. Verify process
        # state explicitly; zombie members no longer own memory or descriptors.
        import psutil
        # Do not request status for every process on the machine: macOS may
        # deny unrelated processes. Inspect status only for our owned group.
        for pid in psutil.pids():
            try:
                if os.getpgid(pid)!=group:continue
            except ProcessLookupError:continue
            except PermissionError:continue
            try:
                if psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE:return True
            except psutil.NoSuchProcess:continue
            except psutil.AccessDenied:return True  # still potentially alive
        return False

    def close(self):
        process=self.process
        channel,self.channel=self.channel,None
        if channel:
            try:send(channel,{'action':'stop'})
            except (OSError,RuntimeError):pass
            channel.close()
        if process:
            self.closing=True
            try:
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
                # The known owned group includes native grandchildren after a crash.
                for sig,seconds in [(signal.SIGTERM,3),(signal.SIGKILL,2)]:
                    process.poll()  # reap the leader before checking descendants
                    if not self.group_has_live_members(process.pid):break
                    try:os.killpg(process.pid,sig)
                    except ProcessLookupError:break
                    except PermissionError:
                        if not self.group_has_live_members(process.pid):break
                        raise
                    deadline=time.monotonic()+seconds
                    while time.monotonic()<deadline:
                        process.poll()
                        if not self.group_has_live_members(process.pid):break
                        time.sleep(.05)
                    else:continue
                    break
                if self.group_has_live_members(process.pid):raise RuntimeError('Runtime descendants have not exited')
                process.wait(timeout=2)
            except Exception as error:
                # This failure occurs in the scheduler, outside the child's log.
                # Persist the actual cause as well as the user-facing safeguard.
                logging.getLogger(__name__).exception('Model runtime cleanup failed: model=%s pid=%s',self.active_model,process.pid)
                try:
                    import traceback
                    with (self.project/'state/model-process/server.log').open('a') as output:
                        output.write(f'\nModel runtime cleanup failed: model={self.active_model} pid={process.pid}\n')
                        traceback.print_exc(file=output)
                except OSError:pass
                failure=ModelProcessError('Cannot safely release the previous model runtime; the consumer stopped. Inspect state/model-process/server.log before restarting.',503)
                failure.fatal=True
                raise failure from error
        self.process=None;self.active_model=None;self.loaded=False;self.closing=False
