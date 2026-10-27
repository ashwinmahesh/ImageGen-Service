"""Private model runtime. No HTTP server or consumer; parent owns scheduling."""
import argparse
import os
import socket
import traceback
from runtime_protocol import receive,send


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--socket-fd',type=int,required=True)
    args=parser.parse_args();channel=socket.socket(fileno=args.socket_fd)
    import server
    from job_logs import capture_runtime,event
    from job_queue import JobStore
    from queue_storage import queue_database
    from model_lifecycle import configure
    original_load=server.load_model
    def load():
        previous=server.job_worker.stage
        server.job_worker.stage='loading_model';original_load();send(channel,{'event':'loaded'})
        server.job_worker.stage=previous
    server.load_model=load
    class Stage:
        @property
        def stage(self):return self.value
        @stage.setter
        def stage(self,value):
            self.value=value;event('Stage: '+value);send(channel,{'event':'stage','stage':value})
    server.job_worker=Stage()
    lifetime_lease=None
    ready_sent=False
    while True:
        try:message,fds=receive(channel)
        except EOFError:break
        try:
            if message['action']=='stop':break
            if message['action']!='execute':raise ValueError('Unknown model command')
            # Existing running parents may spawn this updated child after a pull.
            # Emit new private events only when the parent advertises support.
            enabled=message.get('lifecycle_events',False)
            configure((lambda event:send(channel,{'event':'lifecycle','data':event})) if enabled else (lambda event:None))
            if enabled and not ready_sent:
                send(channel,{'event':'ready'});ready_sent=True
            server.app.state.llm_enabled=message['llm_enabled'];server.app.state.loras_enabled=message['loras_enabled']
            job=message['job'];job['_inherited_fds']=tuple(fds)
            # Warm weights must retain the global consumer lease even while idle.
            # Only the GPU descriptor is strictly per-job; received copies close below.
            if lifetime_lease is None and fds:lifetime_lease=os.dup(fds[0])
            if job['request']['model']!=server.API_MODEL:raise ValueError('Model process request mismatch')
            if message['database_path']!=str(queue_database(server.PROJECT_ROOT)):raise ValueError('Model queue path mismatch')
            server.job_store=JobStore(message['database_path'])
            with capture_runtime(server.PROJECT_ROOT,job):
                result=server.execute_job(job)
            send(channel,{'event':'result','result':result})
        except Exception as error:
            traceback.print_exc()
            send(channel,{'event':'error','message':str(error)[:2000] or type(error).__name__,'http_status':getattr(error,'http_status',500)})
        finally:
            for descriptor in fds:os.close(descriptor)
    if lifetime_lease is not None:os.close(lifetime_lease)
    channel.close()


if __name__=='__main__':main()
