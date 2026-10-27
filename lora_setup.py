"""Add user adapters without editing tracked files or loading image models."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import shutil
import sys
import tempfile
from urllib.parse import urlsplit,unquote,quote,parse_qs

from adapter_catalog import load_entries,model_lora_directory
from adapter_validation import CUSTOM_LORA_MODELS,inspect_adapter
from model_profiles import PROFILES


class SetupCancelled(Exception):
    pass


def custom_id(value):
    value=value.removeprefix('custom-')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,72}',value):
        raise ValueError('Choose a custom ID of 1–73 lowercase letters, digits or hyphens, starting with a letter/digit')
    return 'custom-'+value


def checked_text(value,label,limit,*,empty=False):
    if not isinstance(value,str) or (not empty and not value.strip()) or len(value)>limit or any(ord(c)<32 or ord(c)==127 for c in value):
        raise ValueError(f'{label} must be {"0" if empty else "1"}–{limit} printable characters on one line')
    return value.strip()


def checked_strength(value):
    try:result=float(value)
    except (TypeError,ValueError) as error:raise ValueError('LoRA strength must be a number from 0 to 2') from error
    if not math.isfinite(result) or not 0<=result<=2:raise ValueError('LoRA strength must be a number from 0 to 2')
    return result


def checked_filename(value):
    if not isinstance(value,str) or not value or '\\' in value or any(ord(c)<32 or ord(c)==127 for c in value) or any(part in ('','..','.') for part in value.split('/')) or PurePosixPath(value).is_absolute() or not value.lower().endswith('.safetensors'):
        raise ValueError('Choose a repository-relative .safetensors filename without dot/path traversal components')
    return value


def parse_source(value):
    value=value.strip()
    path=Path(value).expanduser()
    if path.is_file() or value.startswith(('/','~/','./','../')) or value.lower().endswith('.safetensors') and '://' not in value:
        if not path.is_file() or path.suffix.lower()!='.safetensors':raise ValueError('The local source must be an existing .safetensors file')
        return {'kind':'local','path':path.resolve()}
    if '://' in value:
        url=urlsplit(value)
        if url.scheme!='https' or url.netloc!='huggingface.co' or url.fragment or set(parse_qs(url.query)) - {'download'}:
            raise ValueError('Use a Hugging Face model/file URL (without credentials), owner/repository, or a local .safetensors path')
        parts=url.path.strip('/').split('/')
        if len(parts)<2:raise ValueError('The Hugging Face URL needs an owner and model repository')
        result={'kind':'huggingface','repository':'/'.join(parts[:2])}
        if len(parts)>2:
            if len(parts)<5 or parts[2] not in ('blob','resolve'):raise ValueError('Use the model repository URL or its .safetensors file page/download URL')
            result.update(revision=unquote(parts[3]),filename=checked_filename(unquote('/'.join(parts[4:]))))
    else:result={'kind':'huggingface','repository':value}
    from huggingface_hub.utils import validate_repo_id
    try:validate_repo_id(result['repository'])
    except ValueError as error:raise ValueError('Use a Hugging Face owner/repository model ID') from error
    if '/' not in result['repository']:raise ValueError('Use a Hugging Face owner/repository model ID')
    return result


def ask(label,default=None,*,empty=False):
    while True:
        suffix=f' [{default}]' if default not in (None,'') else ''
        answer=input(label+suffix+': ').strip()
        if answer.lower() in ('q','quit','cancel'):raise SetupCancelled()
        if not answer and default is not None:answer=default
        if answer or empty:return answer
        print('Enter a value, or q to cancel.')


def choose_model(value,interactive):
    if value is not None:
        if value not in CUSTOM_LORA_MODELS:raise ValueError('Custom LoRA setup supports: '+', '.join(CUSTOM_LORA_MODELS))
        return value
    if not interactive:return 'z-image-turbo'
    print('Choose the model the LoRA was trained for:')
    for number,model in enumerate(CUSTOM_LORA_MODELS,1):print(f'  {number}. {PROFILES[model]["name"]}'+(' (default)' if number==1 else ''))
    while True:
        answer=ask('Model (number or ID; q cancels)','1')
        if answer.isdigit() and 1<=int(answer)<=len(CUSTOM_LORA_MODELS):return CUSTOM_LORA_MODELS[int(answer)-1]
        if answer in CUSTOM_LORA_MODELS:return answer
        print('Choose one of the listed models.')


def resolve_hub(source,directory,*,filename=None,revision=None,interactive=False):
    from huggingface_hub import HfApi
    for key,value in [('filename',filename),('revision',revision)]:
        if value is not None and source.get(key) is not None and value!=source[key]:raise ValueError(f'The explicit {key} conflicts with the Hugging Face file URL')
    revision=revision or source.get('revision') or 'main'
    if not revision.strip() or any(ord(c)<32 for c in revision):raise ValueError('Invalid Hugging Face revision')
    client=HfApi(endpoint='https://huggingface.co')
    try:info=client.model_info(source['repository'],revision=revision,timeout=15)
    except Exception as error:raise RuntimeError('Could not read the Hugging Face repository/revision; check the source and your Hugging Face access') from error
    commit=info.sha
    if not isinstance(commit,str) or not re.fullmatch(r'[a-fA-F0-9]{40}',commit):raise RuntimeError('Hugging Face did not resolve this source to a full immutable commit')
    filename=filename or source.get('filename')
    if filename is None:
        try:files=client.list_repo_files(source['repository'],revision=commit)
        except Exception as error:raise RuntimeError('Could not list Hugging Face files; provide LORA_FILENAME explicitly or download the file yourself') from error
        files=[name for name in files if name.lower().endswith('.safetensors')]
        if not files:raise ValueError('This repository has no .safetensors files')
        for name in files:checked_filename(name)
        if len(files)==1:filename=files[0];print('Adapter file:',filename)
        elif not interactive:raise ValueError('This repository has several .safetensors files; set LORA_FILENAME or use the full file URL')
        else:
            if len(files)>100:raise ValueError('This repository has many weight files; use its exact .safetensors file URL')
            print('Select the LoRA weights (not a full model checkpoint):')
            for number,name in enumerate(files,1):print(f'  {number}. {name}')
            while filename is None:
                answer=ask('File number (q cancels)')
                if answer.isdigit() and 1<=int(answer)<=len(files):filename=files[int(answer)-1]
                else:print('Choose a listed file number.')
    filename=checked_filename(filename)
    return {**source,'revision':commit.lower(),'filename':filename,'requested_revision':revision}


def download_hub(source,directory):
    from huggingface_hub import hf_hub_download
    folder=directory/'downloads'/source['repository'].replace('/','--')/source['revision']
    print(f'Downloading/reusing {source["repository"]}/{source["filename"]} at {source["revision"]}…',flush=True)
    try:
        path=hf_hub_download(source['repository'],source['filename'],revision=source['revision'],
                             local_dir=str(folder),endpoint='https://huggingface.co',etag_timeout=15)
    except Exception as error:raise RuntimeError('Could not download the Hugging Face adapter; rerun to resume, or import a local file. Check repository access and the filename') from error
    return Path(path)


def register(project,model,source,*,identifier,name,trigger='',strength=.8,provenance=None,expected_sha256=None):
    """Atomic local registration with immutable weight names and no source edits."""
    if model not in CUSTOM_LORA_MODELS:raise ValueError('Unsupported custom LoRA model')
    identifier=custom_id(identifier);name=checked_text(name,'Display name',120)
    trigger=checked_text(trigger,'Trigger words',1000,empty=True);strength=checked_strength(strength)
    if expected_sha256 is not None and not re.fullmatch(r'[a-fA-F0-9]{64}',expected_sha256):raise ValueError('LORA_SHA256 must be a full SHA256 hex digest')
    if set(provenance or {})-{'source','repository','revision','remote_filename'}:raise ValueError('Unsupported provenance fields')
    source=Path(source).expanduser().resolve()
    if not source.is_file() or source.suffix.lower()!='.safetensors':raise ValueError('The local source must be an existing .safetensors file')
    inspect_adapter(source,model)
    directory=model_lora_directory(project,model);directory.mkdir(parents=True,exist_ok=True)
    custom=directory/'custom.json'
    temporary=None
    with (directory/'.custom-install.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        all_entries=load_entries(project,directory)
        entries=json.loads(custom.read_text()) if custom.is_file() else []
        current=next((entry for entry in all_entries if entry['id']==identifier),None)
        if current is not None and not any(entry['id']==identifier for entry in entries):raise ValueError('That ID belongs to an existing preset; choose another custom ID')
        try:
            with tempfile.NamedTemporaryFile(dir=directory,prefix='.install-',suffix='.safetensors',delete=False) as stream:temporary=Path(stream.name)
            shutil.copyfile(source,temporary)
            inspect_adapter(temporary,model)
            with temporary.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
            if expected_sha256 and checksum!=expected_sha256.lower():raise ValueError('LoRA SHA256 mismatch; nothing was registered')
            filename=f'{identifier}-{checksum[:16]}.safetensors'
            entry={'id':identifier,'name':name,'filename':filename,'models':[model],'default_strength':strength,
                   'trigger_prompt':trigger,'sha256':checksum,'bytes':temporary.stat().st_size,
                   'source':str(source),'custom':True,**(provenance or {})}
            if current is not None:
                if any(current.get(key)!=entry[key] for key in ('sha256','name','trigger_prompt','default_strength')):
                    raise ValueError('That custom ID already has different weights/settings; choose a new ID (for example a version suffix). Existing files and presets were kept')
                if model in current['models']:
                    destination=directory/current['filename']
                    if destination.is_symlink():raise ValueError('Custom weights must be a regular file in the adapter directory')
                    if destination.is_file():
                        with destination.open('rb') as stream:existing=hashlib.file_digest(stream,'sha256').hexdigest()
                        if existing!=checksum:raise ValueError('Existing custom weights changed; keep them for inspection and install with a new ID')
                        return current,directory,False
                entry={**current,'models':list(dict.fromkeys([*current['models'],model]))}
                filename=entry['filename']
            destination=directory/filename
            if destination.is_symlink():raise ValueError('Custom weights must be a regular file in the adapter directory')
            if destination.exists():
                with destination.open('rb') as stream:existing=hashlib.file_digest(stream,'sha256').hexdigest()
                if existing!=checksum:raise ValueError('A different file already uses the destination name; nothing was overwritten')
            else:temporary.replace(destination)
            if current is None:entries.append(entry)
            else:entries=[entry if item['id']==identifier else item for item in entries]
            with tempfile.NamedTemporaryFile(dir=directory,prefix='.custom-',suffix='.json.tmp',mode='w',delete=False) as stream:
                catalog_temp=Path(stream.name)
                try:
                    json.dump(entries,stream,indent=2,ensure_ascii=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
                except BaseException:catalog_temp.unlink(missing_ok=True);raise
            try:catalog_temp.replace(custom)
            finally:catalog_temp.unlink(missing_ok=True)
            return entry,directory,True
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)


def main(argv=None,*,project=None):
    def env(name):return os.environ.get(name) or None
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=env('IMAGE_MODEL'))
    parser.add_argument('--source',default=env('LORA_SOURCE'),help='Local file, owner/repo, or Hugging Face model/file URL')
    parser.add_argument('--id',default=env('LORA_ID'),help='User ID; custom- is prepended if absent')
    parser.add_argument('--name',default=env('LORA_NAME'))
    parser.add_argument('--trigger',default=env('LORA_TRIGGER'))
    parser.add_argument('--strength',default=env('LORA_STRENGTH'))
    parser.add_argument('--filename',default=env('LORA_FILENAME'),help='Repository-relative Hugging Face weight filename')
    parser.add_argument('--revision',default=env('LORA_REVISION'))
    parser.add_argument('--sha256',default=env('LORA_SHA256'))
    parser.add_argument('--noninteractive',action='store_true',default=os.environ.get('NONINTERACTIVE')=='1')
    args=parser.parse_args(argv)
    # Keep transfers in local_dir; avoid an extra Xet cache in the user's home.
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
    interactive=sys.stdin.isatty() and not args.noninteractive
    project=Path(__file__).resolve().parent if project is None else Path(project).resolve()
    try:
        print('Custom LoRA setup · no project source files are changed. q or Ctrl+C cancels.\nChoose weights trained for the selected base model; setup checks format/family, not image quality.')
        model=choose_model(args.model,interactive)
        source_text=args.source or (ask('Local .safetensors path or Hugging Face repository/file URL') if interactive else None)
        if not source_text:raise ValueError('Provide LORA_SOURCE (or --source), or run make setup-lora in a terminal')
        source=parse_source(source_text)
        if source['kind']=='local' and (args.filename or args.revision):raise ValueError('LORA_FILENAME/REVISION apply only to Hugging Face sources')
        directory=model_lora_directory(project,model)
        if source['kind']=='huggingface':source=resolve_hub(source,directory,filename=args.filename,revision=args.revision,interactive=interactive)
        stem=(source['path'] if source['kind']=='local' else Path(source['filename'])).stem
        default_name=stem.replace('_',' ').replace('-',' ')[:120] or 'Custom LoRA'
        name=checked_text(args.name if args.name is not None else ask('Display name',default_name) if interactive else default_name,'Display name',120)
        slug=re.sub(r'[^a-z0-9]+','-',name.lower()).strip('-')[:73].rstrip('-') or 'adapter'
        identifier=custom_id(args.id if args.id is not None else ask('Custom ID (custom- prefix is automatic)',slug) if interactive else slug)
        trigger=checked_text(args.trigger if args.trigger is not None else ask('Trigger words (Enter for none)','',empty=True) if interactive else '','Trigger words',1000,empty=True)
        strength=args.strength
        while strength is None:
            value=ask('Default strength, 0–2','0.8') if interactive else .8
            try:strength=checked_strength(value)
            except ValueError:
                if not interactive:raise
                print('Enter a finite number from 0 to 2.')
        strength=checked_strength(strength)
        if args.sha256 is not None and not re.fullmatch(r'[a-fA-F0-9]{64}',args.sha256):raise ValueError('LORA_SHA256 must be a full SHA256 hex digest')
        provenance=None
        if source['kind']=='huggingface':
            provenance={'source':f'https://huggingface.co/{source["repository"]}/blob/{quote(source["revision"],safe="")}/{quote(source["filename"],safe="/")}',
                        'repository':source['repository'],'revision':source['revision'],'remote_filename':source['filename']}
            path=download_hub(source,directory)
        else:path=source['path']
        entry,directory,changed=register(project,model,path,identifier=identifier,name=name,trigger=trigger,strength=strength,provenance=provenance,expected_sha256=args.sha256)
        print(('Installed: ' if changed else 'Already installed and verified: ')+entry['name'])
        print(f'LoRA ID: {entry["id"]}\nModel: {model}\nWeights: {directory/entry["filename"]}\nUser catalog: {directory/"custom.json"}\nSHA256: {entry["sha256"]}')
        print('Reload the webpage and choose this model + Style adapter, or send lora="'+entry['id']+'" in an API request. No server restart is needed; adapters must be enabled and the base model set up.')
        return 0
    except (SetupCancelled,KeyboardInterrupt,EOFError):
        print('\nLoRA setup cancelled. Existing registrations/files were kept; any Hub download cache can be reused.',file=sys.stderr);return 130
    except (ValueError,OSError,RuntimeError) as error:
        print('LoRA setup failed: '+str(error),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
