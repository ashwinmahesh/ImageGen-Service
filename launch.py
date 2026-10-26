"""Start one cross-model generator; choose its default from setup models."""
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
from model_profiles import PROFILES, DEFAULT_MODEL, select_model
from queue_launch import ensure_queue, DEFAULT_QUEUE_PORT
from enhancer_profile import check_assets as check_enhancer_assets
from model_assets import check_model_assets
from model_catalog import default_model
from queue_storage import initialize_shared_queue
ROOT=Path(__file__).resolve().parent
API_MODEL=DEFAULT_MODEL


def port_free(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1',port))
        except OSError:
            return False
    return True


def check_assets(selected):
    check_model_assets(ROOT,selected)


def available_models():
    ready=[]
    for identifier in PROFILES:
        try:
            check_assets(identifier)
        except SystemExit:
            continue
        ready.append(identifier)
    return ready


def select_llm_mode(*,interactive=None,prompt=input):
    if interactive is None:
        interactive=sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get('NONINTERACTIVE')!='1'
    if not interactive:
        return True
    print("A small AI text model on your Mac can add visual detail to your image descriptions before generation; you can turn this off for each image.")
    while True:
        try:
            answer=prompt('Enable local LLM prompt expansion? [Y/n]: ').strip().lower()
        except EOFError:
            raise SystemExit('Startup cancelled. Use NONINTERACTIVE=1 to accept defaults, or NO_LLM=1 to disable the LLM.') from None
        if answer in ('','y','yes'):
            return True
        if answer in ('n','no'):
            return False
        print('Choose Y/Enter to enable, or N to disable.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=5732)
    parser.add_argument('--queue-port',type=int,default=DEFAULT_QUEUE_PORT)
    parser.add_argument('--no-loras',action='store_true',help='Disable LoRAs for this server')
    parser.add_argument('--no-llm',action='store_true',help='Disable local prompt expansion (default on)')
    parser.add_argument('--model', choices=PROFILES, help='Choose the default model; the generator processes jobs for all setup models')
    args=parser.parse_args()
    try:
        explicit=args.model or os.environ.get('IMAGE_MODEL') or os.environ.get('MODEL')
        ready=available_models()
        selected=select_model(explicit,choices=ready,preferred=default_model(ROOT,ready) if ready else None)
    except (ValueError,RuntimeError) as error:
        parser.error(str(error))
    check_assets(selected)
    llm_enabled=not args.no_llm and select_llm_mode()
    if llm_enabled:
        try:
            check_enhancer_assets(ROOT)
        except (RuntimeError, OSError, ValueError) as error:
            raise SystemExit(str(error)+' To serve without the LLM, use make serve NO_LLM=1.') from error
    if args.port==args.queue_port:
        raise SystemExit('PORT and QUEUE_PORT must be different')
    ensure_queue(args.queue_port)
    os.environ['IMAGEGEN_QUEUE_PORT']=str(args.queue_port)
    if not port_free(args.port):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health',timeout=5) as response:
                state=json.load(response)
        except Exception:
            raise SystemExit(f'Port {args.port} is in use by another service. Choose PORT=5733 or stop it first.')
        if state.get('service')!='imagegen-generator' or state.get('runtime_model_switching') is not True or state.get('project_dir')!=str(ROOT):
            raise SystemExit(f'Port {args.port} belongs to another service; leaving it running.')
        found=subprocess.check_output(['/usr/sbin/lsof','-nP','-t',f'-iTCP:{args.port}','-sTCP:LISTEN'],text=True).split()
        if len(found) != 1:
            raise SystemExit('Cannot identify one existing image-server process; leaving it running.')
        pid=int(found[0])
        cwd=subprocess.check_output(['/usr/sbin/lsof','-a','-p',str(pid),'-d','cwd','-Fn'],text=True)
        accepted={str(ROOT),str(ROOT.parent/'z-image-turbo'),str(ROOT.parent/'chroma1-hd'),str(ROOT.parent/'chroma1-flash')}
        directories={line[1:] for line in cwd.splitlines() if line.startswith('n')}
        if not directories.intersection(accepted):
            raise SystemExit('The existing service is outside the managed image folders; leaving it running.')
        command_line=subprocess.check_output(['/bin/ps','-p',str(pid),'-o','command='],text=True)
        if 'server.py' not in command_line or 'python' not in command_line.lower():
            raise SystemExit('The existing listener is not a managed Python image server; leaving it running.')
        if directories == {str(ROOT)} and state.get('loras_enabled',True) == (not args.no_loras) and state.get('prompt_expansion',{}).get('enabled',False) == llm_enabled and state.get('capabilities',{}).get('queue_api_url',f'http://127.0.0.1:{DEFAULT_QUEUE_PORT}') == f'http://127.0.0.1:{args.queue_port}':
            initialize_shared_queue(ROOT).set_setting('default_model',selected)
            print(f'All setup models are already serving at http://127.0.0.1:{args.port}/; default {selected}')
            return
        if state.get('busy'):
            raise SystemExit(f"The generator is processing a batch. Wait for it to finish, then run this command again.")
        mode = "base model only" if args.no_loras else "LoRAs enabled"
        print(f"Restarting idle generator with default {selected} ({mode})...",flush=True)
        os.kill(pid,signal.SIGTERM)
        for _ in range(150):
            if port_free(args.port):
                break
            time.sleep(0.2)
        else:
            raise SystemExit('The previous server has not stopped yet. Try again shortly.')
    print(f'All setup models are eligible; default model: {PROFILES[selected]["name"]}. Models load on demand.',flush=True)
    os.chdir(ROOT)
    command=[sys.executable,str(ROOT/'server.py'),'--model',selected,'--port',str(args.port)]
    if args.no_loras:
        command.append('--no-loras')
    if not llm_enabled:
        command.append('--no-llm')
    os.execv(sys.executable,command)


if __name__=='__main__':
    main()
