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
from urllib.parse import urlsplit,unquote,quote,parse_qs,urljoin
import urllib.error
import urllib.request

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
    if '://' in value and urlsplit(value).netloc in ('civitai.com','www.civitai.com'):return parse_civitai(value)
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


CIVITAI='https://civitai.com'
# Civitai base-model labels, normalized, that match each setup model; 'Other' or a missing label is not checked.
CIVITAI_BASES={'z-image-turbo':('zimage',),'chroma1-hd':('chroma',),'chroma1-flash':('chroma',),
               'flux2-klein-4b':('klein4b',),'flux2-klein-9b':('klein9b',),'qwen-image-2.1-turbo':('qwen21','qwenimage21')}


def parse_civitai(value):
    """civitai.com/models/<id>[?modelVersionId=<version>] or civitai.com/api/download/models/<version>."""
    url=urlsplit(value);query=parse_qs(url.query);parts=url.path.strip('/').split('/')
    if url.scheme!='https' or url.fragment or url.username or url.password or {'token','apikey','api_key'} & {key.lower() for key in query}:
        raise ValueError('Use the Civitai model page URL without credentials; set CIVITAI_TOKEN for downloads that need sign-in')
    number=lambda text:int(text) if text and text.isdigit() else None
    if len(parts)>=2 and parts[0]=='models' and number(parts[1]):
        if set(query)-{'modelVersionId'}:raise ValueError('Use the Civitai model page URL, optionally with ?modelVersionId=')
        return {'kind':'civitai','model_id':number(parts[1]),'version_id':number((query.get('modelVersionId') or [None])[0]),'file_id':None}
    if parts[:3]==['api','download','models'] and len(parts)==4 and number(parts[3]):
        if set(query)-{'type','format','size','fp','fileId'}:raise ValueError('Unsupported Civitai download URL options')
        return {'kind':'civitai','model_id':None,'version_id':number(parts[3]),'file_id':number((query.get('fileId') or [None])[0])}
    raise ValueError('Use a Civitai model page URL such as https://civitai.com/models/123 (optionally ?modelVersionId=456)')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def civitai_request(url,*,token=None,timeout=30):
    """GET from Civitai, following redirects by hand so an API key only ever goes to civitai.com, never to its download CDN."""
    opener=urllib.request.build_opener(_NoRedirect)
    for _ in range(6):
        parts=urlsplit(url)
        if parts.scheme!='https':raise RuntimeError('Civitai redirected to a non-HTTPS URL; download the file yourself and import it')
        headers={'User-Agent':'imagegen-setup-lora','Accept':'application/json, */*'}
        if token and parts.hostname=='civitai.com':headers['Authorization']='Bearer '+token
        try:return opener.open(urllib.request.Request(url,headers=headers),timeout=timeout)
        except urllib.error.HTTPError as error:
            error.close()
            if error.code in (301,302,303,307,308) and error.headers.get('Location'):
                url=urljoin(url,error.headers['Location']);continue
            if error.code in (401,403):
                raise RuntimeError('Civitai needs you signed in for this download: create an API key at civitai.com/user/account and rerun with CIVITAI_TOKEN=<key> (never in the URL)') from None
            raise RuntimeError(f'Civitai returned HTTP {error.code} for this source') from None
        except (urllib.error.URLError,TimeoutError) as error:
            raise RuntimeError('Could not reach Civitai; check the connection and retry') from error
    raise RuntimeError('Too many Civitai redirects')


def civitai_json(path):
    with civitai_request(CIVITAI+'/api/v1/'+path,token=os.environ.get('CIVITAI_TOKEN') or None) as response:
        return json.load(response)


def resolve_civitai(source,model,*,filename=None,revision=None,interactive=False):
    if revision is not None:
        if not revision.isdigit():raise ValueError('For Civitai, LORA_REVISION is a numeric model version ID')
        if source['version_id'] and int(revision)!=source['version_id']:raise ValueError('The explicit revision conflicts with the Civitai URL version')
        source={**source,'version_id':int(revision)}
    if source['version_id']:
        version=civitai_json(f'model-versions/{source["version_id"]}')
        info=version.get('model') or {}
        if source['model_id'] and version.get('modelId')!=source['model_id']:raise ValueError('That Civitai version belongs to a different model')
    else:
        info=civitai_json(f'models/{source["model_id"]}')
        published=[item for item in info.get('modelVersions') or [] if item.get('status','Published')=='Published']
        if not published:raise ValueError('This Civitai model has no published versions')
        version=published[0];print(f'Using the newest Civitai version: {version.get("name")} ({version["id"]}); set LORA_REVISION=<version ID> to pin another')
    kind=info.get('type')
    if kind not in ('LORA','LoCon',None):raise ValueError(f'This Civitai page is a {kind}, not a LoRA, so there is no adapter to install')
    base=str(version.get('baseModel') or '')
    normalized=re.sub(r'[^a-z0-9]','',base.lower().replace('.',''))
    if normalized not in ('','other') and not any(token in normalized for token in CIVITAI_BASES[model]):
        raise ValueError(f'This Civitai version is for {base}, not {PROFILES[model]["name"]}; choose the matching model or version')
    files=[item for item in version.get('files') or [] if str(item.get('name','')).lower().endswith('.safetensors') and item.get('type','Model') in ('Model','Pruned Model')]
    if source.get('file_id'):files=[item for item in files if item.get('id')==source['file_id']]
    if filename is not None:files=[item for item in files if item.get('name')==filename]
    if not files:raise ValueError('This Civitai version has no matching .safetensors adapter file')
    chosen=files[0]
    if len(files)>1:
        if not interactive:raise ValueError('This Civitai version has several .safetensors files; set LORA_FILENAME')
        print('Select the LoRA weights:')
        for number,item in enumerate(files,1):print(f'  {number}. {item["name"]}')
        while True:
            answer=ask('File number (q cancels)')
            if answer.isdigit() and 1<=int(answer)<=len(files):chosen=files[int(answer)-1];break
            print('Choose a listed file number.')
    checksum=str((chosen.get('hashes') or {}).get('SHA256') or '').lower()
    model_id=version.get('modelId') or source['model_id']
    return {'kind':'civitai','model_id':model_id,'version_id':version['id'],'filename':checked_filename(chosen['name']),
            'url':chosen.get('downloadUrl') or f'{CIVITAI}/api/download/models/{version["id"]}?fileId={chosen.get("id")}',
            'sha256':checksum if re.fullmatch(r'[a-f0-9]{64}',checksum) else None,'name':str(info.get('name') or Path(chosen['name']).stem),
            'trained_words':[str(word) for word in version.get('trainedWords') or [] if str(word).strip()],'base_model':base,
            'page':f'{CIVITAI}/models/{model_id}?modelVersionId={version["id"]}'}


def download_civitai(source,directory):
    folder=directory/'downloads'/f'civitai-{source["model_id"]}'/str(source['version_id']);folder.mkdir(parents=True,exist_ok=True)
    path=folder/Path(source['filename']).name
    if path.is_file() and source['sha256']:
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()==source['sha256']:print('Reusing the downloaded',path.name);return path
    print(f'Downloading {source["filename"]} from Civitai (version {source["version_id"]})…',flush=True)
    partial=path.with_name(path.name+'.part')
    try:
        with civitai_request(source['url'],token=os.environ.get('CIVITAI_TOKEN') or None,timeout=60) as response,partial.open('wb') as stream:
            if 'text/html' in (response.headers.get('Content-Type') or ''):raise RuntimeError('Civitai sent a web page instead of the weights; sign in with CIVITAI_TOKEN or download the file yourself')
            digest=hashlib.sha256()
            while chunk:=response.read(1<<20):stream.write(chunk);digest.update(chunk)
        if source['sha256'] and digest.hexdigest()!=source['sha256']:raise RuntimeError('The Civitai download does not match its published SHA256; nothing was installed')
        partial.replace(path)
    finally:partial.unlink(missing_ok=True)
    return path


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
    parser.add_argument('--source',default=env('LORA_SOURCE'),help='Local file, owner/repo, Hugging Face model/file URL, or Civitai model URL')
    parser.add_argument('--id',default=env('LORA_ID'),help='User ID; custom- is prepended if absent')
    parser.add_argument('--name',default=env('LORA_NAME'))
    parser.add_argument('--trigger',default=env('LORA_TRIGGER'))
    parser.add_argument('--strength',default=env('LORA_STRENGTH'))
    parser.add_argument('--filename',default=env('LORA_FILENAME'),help='Repository-relative Hugging Face weight filename, or the Civitai file name')
    parser.add_argument('--revision',default=env('LORA_REVISION'),help='Hugging Face branch/tag/commit, or Civitai model version ID')
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
        source_text=args.source or (ask('Local .safetensors path, Hugging Face repository/file URL, or Civitai model URL') if interactive else None)
        if not source_text:raise ValueError('Provide LORA_SOURCE (or --source), or run make setup-lora in a terminal')
        source=parse_source(source_text)
        if source['kind']=='local' and (args.filename or args.revision):raise ValueError('LORA_FILENAME/REVISION apply only to Hugging Face and Civitai sources')
        directory=model_lora_directory(project,model)
        if source['kind']=='huggingface':source=resolve_hub(source,directory,filename=args.filename,revision=args.revision,interactive=interactive)
        elif source['kind']=='civitai':
            source=resolve_civitai(source,model,filename=args.filename,revision=args.revision,interactive=interactive)
            print(f'Civitai: {source["name"]} · base model {source["base_model"] or "not stated"} · {source["page"]}\nCheck the license and recommended strength on that page.')
            if args.sha256 and source['sha256'] and args.sha256.lower()!=source['sha256']:raise ValueError('LORA_SHA256 differs from the SHA256 Civitai publishes for this file')
        stem=(source['path'] if source['kind']=='local' else Path(source['filename'])).stem
        default_name=(source['name'] if source['kind']=='civitai' else stem.replace('_',' ').replace('-',' '))[:120] or 'Custom LoRA'
        default_trigger=', '.join(source.get('trained_words') or [])
        name=checked_text(args.name if args.name is not None else ask('Display name',default_name) if interactive else default_name,'Display name',120)
        slug=re.sub(r'[^a-z0-9]+','-',name.lower()).strip('-')[:73].rstrip('-') or 'adapter'
        identifier=custom_id(args.id if args.id is not None else ask('Custom ID (custom- prefix is automatic)',slug) if interactive else slug)
        trigger=checked_text(args.trigger if args.trigger is not None else ask('Trigger words (Enter for none)' if not default_trigger else 'Trigger words',default_trigger,empty=True) if interactive else default_trigger,'Trigger words',1000,empty=True)
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
        elif source['kind']=='civitai':
            provenance={'source':source['page'],'repository':f'civitai:{source["model_id"]}','revision':str(source['version_id']),'remote_filename':source['filename']}
            path=download_civitai(source,directory)
        else:path=source['path']
        entry,directory,changed=register(project,model,path,identifier=identifier,name=name,trigger=trigger,strength=strength,provenance=provenance,expected_sha256=args.sha256 or source.get('sha256'))
        print(('Installed: ' if changed else 'Already installed and verified: ')+entry['name'])
        print(f'LoRA ID: {entry["id"]}\nModel: {model}\nWeights: {directory/entry["filename"]}\nUser catalog: {directory/"custom.json"}\nSHA256: {entry["sha256"]}')
        print('Reload the webpage and choose this model + Style adapter, or send lora="'+entry['id']+'" in an API request. No server restart is needed; adapters must be enabled and the base model set up.')
        return 0
    except (SetupCancelled,KeyboardInterrupt,EOFError):
        print('\nLoRA setup cancelled. Existing registrations/files were kept; any Hub download cache can be reused.',file=sys.stderr);return 130
    except (ValueError,OSError,RuntimeError) as error:
        print('LoRA setup failed: '+str(error),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
