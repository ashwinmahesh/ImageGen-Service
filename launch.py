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
from model_profiles import PROFILES, DEFAULT_MODEL, select_model, SHARED_ASSETS
from storage_paths import data_directory, require_storage
from enhancer_profile import check_assets as check_enhancer_assets
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
    store=data_directory(ROOT)
    folder=store/selected/'weights'
    require_storage(folder)
    profile=PROFILES[selected]
    required=[folder/'download.json']
    directories=[]
    if profile['backend']=='sd-cpp':
        required += [folder/profile['filename'],ROOT/'runtime'/'sd-cli',ROOT/'runtime'/'libstable-diffusion.dylib']
        required += [store/'chroma1-hd'/'weights'/asset[2] for asset in SHARED_ASSETS]
    else:
        directories=[folder/name for name in ['transformer','text_encoder','tokenizer','vae']]
    missing=[path for path in required if not path.is_file()]+[path for path in directories if not path.is_dir()]
    if missing:
        raise SystemExit(f'Model assets missing; run make setup MODEL={selected} first. Missing: {missing[0]}')


def available_models():
    ready=[]
    for identifier in PROFILES:
        try:
            check_assets(identifier)
        except SystemExit:
            continue
        ready.append(identifier)
    return ready


def select_llm_mode(explicit,*,interactive=None,prompt=input):
    if explicit is not None:
        return explicit
    if interactive is None:
        interactive=sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get('NONINTERACTIVE')!='1'
    if not interactive:
        return True
    while True:
        try:
            answer=prompt('Enable local LLM prompt expansion? [Y/n]: ').strip().lower()
        except EOFError:
            raise SystemExit('Startup cancelled. Use NO_LLM=1 to disable or WITH_LLM=1 to enable without asking.') from None
        if answer in ('','y','yes'):
            return True
        if answer in ('n','no'):
            return False
        print('Choose Y/Enter to enable, or N to disable.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=5732)
    parser.add_argument('--no-loras',action='store_true',help='Disable LoRAs for this server')
    llm=parser.add_mutually_exclusive_group()
    llm.add_argument('--with-llm',dest='with_llm',action='store_true',help='Enable local prompt expansion without asking (default on)')
    llm.add_argument('--no-llm',dest='with_llm',action='store_false',help='Disable local prompt expansion')
    parser.set_defaults(with_llm=None)
    parser.add_argument('--model', choices=PROFILES, help='Choose a set up backend; omitted in a terminal lists locally set up models')
    args=parser.parse_args()
    try:
        explicit=args.model or os.environ.get('IMAGE_MODEL') or os.environ.get('MODEL')
        selected=select_model(explicit,choices=None if explicit else available_models())
    except (ValueError,RuntimeError) as error:
        parser.error(str(error))
    check_assets(selected)
    args.with_llm=select_llm_mode(args.with_llm)
    if args.with_llm:
        try:
            check_enhancer_assets(ROOT)
        except (RuntimeError, OSError, ValueError) as error:
            raise SystemExit(str(error)+' To serve without the LLM, use make serve NO_LLM=1.') from error
    if not port_free(args.port):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health',timeout=5) as response:
                state=json.load(response)
        except Exception:
            raise SystemExit(f'Port {args.port} is in use by another service. Choose PORT=5733 or stop it first.')
        allowed={'z-image-turbo','chroma1-hd','chroma1-flash'}
        if state.get('model') not in allowed:
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
        if directories == {str(ROOT)} and state.get('model') == selected and state.get('loras_enabled',True) == (not args.no_loras) and state.get('prompt_expansion',{}).get('enabled',False) == args.with_llm:
            print(f'{selected} is already serving at http://127.0.0.1:{args.port}/')
            return
        if state.get('busy'):
            raise SystemExit(f"{state['model']} is generating. Wait for it to finish, then run this command again.")
        mode = "base model only" if args.no_loras else "LoRAs enabled"
        print(f"Switching idle service from {state['model']} to {selected} ({mode})...",flush=True)
        os.kill(pid,signal.SIGTERM)
        for _ in range(150):
            if port_free(args.port):
                break
            time.sleep(0.2)
        else:
            raise SystemExit('The previous server has not stopped yet. Try again shortly.')
    os.chdir(ROOT)
    command=[sys.executable,str(ROOT/'server.py'),'--model',selected,'--port',str(args.port)]
    if args.no_loras:
        command.append('--no-loras')
    command.append('--with-llm' if args.with_llm else '--no-llm')
    os.execv(sys.executable,command)


if __name__=='__main__':
    main()
