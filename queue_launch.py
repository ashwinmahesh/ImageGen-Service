"""Start/reuse the independent localhost queue API without loading image assets."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from queue_api import DEFAULT_QUEUE_PORT

ROOT=Path(__file__).resolve().parent


def state(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:
            value=json.load(response)
        if value.get('service')=='imagegen-queue' and value.get('project_dir')==str(ROOT):
            return value
    except (OSError,ValueError):
        pass
    return None


def port_free(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try: probe.bind(('127.0.0.1',port))
        except OSError: return False
    return True


@contextmanager
def start_lock():
    folder=ROOT/'state'/'queue-api'
    if not folder.resolve().is_relative_to(ROOT):
        raise RuntimeError('Queue state must stay inside the project')
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'start.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        yield folder


def ensure_queue(port=DEFAULT_QUEUE_PORT):
    """Detach only the queue process; Ctrl-C in the generator leaves it available."""
    with start_lock() as folder:
        if state(port):
            return
        if not port_free(port):
            raise SystemExit(f'Queue port {port} belongs to another service. Choose QUEUE_PORT=<unused-port>; leaving that service running.')
        log=folder/'server.log'
        with log.open('ab') as output:
            process=subprocess.Popen([sys.executable,str(ROOT/'queue_api.py'),'--port',str(port)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(100):
            if state(port):
                print(f'Queue API: http://127.0.0.1:{port}/docs (stays running when the generator stops)',flush=True)
                return
            if process.poll() is not None:
                break
            time.sleep(.1)
        # This PID is our own child, never a port-derived guess.
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
        raise SystemExit(f'Queue API failed to start. See {log}')


def stop_queue(port):
    with start_lock() as folder:
        if not state(port):
            if port_free(port):
                print('Queue API is already stopped.'); return
            raise SystemExit('The queue port is in use by an unrecognized service; leaving it running.')
        # Verify process command and checkout as well as the health response before signalling.
        found=subprocess.check_output(['/usr/sbin/lsof','-nP','-t',f'-iTCP:{port}','-sTCP:LISTEN'],text=True).split()
        if len(found)!=1:
            raise SystemExit('Cannot identify exactly one queue process; leaving it running.')
        pid=int(found[0])
        command=subprocess.check_output(['/bin/ps','-p',str(pid),'-o','command='],text=True)
        cwd=subprocess.check_output(['/usr/sbin/lsof','-a','-p',str(pid),'-d','cwd','-Fn'],text=True)
        if str(ROOT/'queue_api.py') not in command or 'python' not in command.lower() or f'n{ROOT}\n' not in cwd:
            raise SystemExit('Queue process ownership could not be verified; leaving it running.')
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if port_free(port):
                print('Queue API stopped. Persisted jobs are retained.'); return
            time.sleep(.1)
        raise SystemExit('Queue API is still shutting down; no forced termination was sent.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=DEFAULT_QUEUE_PORT)
    parser.add_argument('--stop',action='store_true')
    args=parser.parse_args()
    if args.stop:
        stop_queue(args.port)
    else:
        ensure_queue(args.port)
        print(f'Queue API ready at http://127.0.0.1:{args.port}/docs')


if __name__=='__main__':main()
