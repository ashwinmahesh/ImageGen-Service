"""Switch between the managed localhost image services, only while idle."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from config import ROOT, API_MODEL


def port_free(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1',port))
        except OSError:
            return False
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--no-loras',action='store_true',help='Disable LoRAs for this server')
    args=parser.parse_args()
    if not port_free(args.port):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health',timeout=5) as response:
                state=json.load(response)
        except Exception:
            raise SystemExit(f'Port {args.port} is in use by another service. Choose PORT=8001 or stop it first.')
        allowed={'z-image-turbo','chroma1-hd','chroma1-flash'}
        if state.get('model') not in allowed:
            raise SystemExit(f'Port {args.port} belongs to another service; leaving it running.')
        if state.get('model') == API_MODEL and state.get('loras_enabled',True) == (not args.no_loras):
            print(f'{API_MODEL} is already serving at http://127.0.0.1:{args.port}/')
            return
        if state.get('busy'):
            raise SystemExit(f"{state['model']} is generating. Wait for it to finish, then run this command again.")
        found=subprocess.check_output(['/usr/sbin/lsof','-nP','-t',f'-iTCP:{args.port}','-sTCP:LISTEN'],text=True).split()
        if len(found) != 1:
            raise SystemExit('Cannot identify one existing image-server process; leaving it running.')
        pid=int(found[0])
        cwd=subprocess.check_output(['/usr/sbin/lsof','-a','-p',str(pid),'-d','cwd','-Fn'],text=True)
        accepted={str(ROOT.parent/'z-image-turbo'),str(ROOT.parent/'chroma1-hd'),str(ROOT.parent/'chroma1-flash')}
        directories={line[1:] for line in cwd.splitlines() if line.startswith('n')}
        if not directories.intersection(accepted):
            raise SystemExit('The existing service is outside the managed image folders; leaving it running.')
        mode = "base model only" if args.no_loras else "LoRAs enabled"
        print(f"Switching idle service from {state['model']} to {API_MODEL} ({mode})...",flush=True)
        os.kill(pid,signal.SIGTERM)
        for _ in range(150):
            if port_free(args.port):
                break
            time.sleep(0.2)
        else:
            raise SystemExit('The previous server has not stopped yet. Try again shortly.')
    os.chdir(ROOT)
    command=[sys.executable,str(ROOT/'server.py'),'--port',str(args.port)]
    if args.no_loras:
        command.append('--no-loras')
    os.execv(sys.executable,command)


if __name__=='__main__':
    main()
