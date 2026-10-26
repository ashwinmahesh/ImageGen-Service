"""Cheap local preflight for pinned, pre-quantized MLX editing checkpoints."""
import json
from pathlib import Path
from model_profiles import PROFILES, SHARED_ASSETS
from storage_paths import data_directory, require_storage


def check_model_assets(project, selected):
    """Read-only readiness shared by setup labels and the serving menu."""
    project=Path(project)
    store=data_directory(project)
    folder=store/selected/'weights'
    require_storage(folder)
    profile=PROFILES[selected]
    required=[folder/'download.json']
    directories=[]
    if profile['backend']=='sd-cpp':
        required += [folder/profile['filename'],project/'runtime'/'sd-cli',project/'runtime'/'libstable-diffusion.dylib']
        required += [store/'chroma1-hd'/'weights'/asset[2] for asset in SHARED_ASSETS]
    elif profile.get('cli_module'):
        try:
            validate_snapshot(folder,profile,completed=True)
        except (OSError,ValueError,TypeError) as error:
            raise SystemExit(f'Model assets incomplete; run make setup MODEL={selected} first. {error}') from error
    else:
        directories=[folder/name for name in ['transformer','text_encoder','tokenizer','vae']]
    missing=[path for path in required if not path.is_file()]+[path for path in directories if not path.is_dir()]
    if missing:
        raise SystemExit(f'Model assets missing; run make setup MODEL={selected} first. Missing: {missing[0]}')


def available_models(project):
    ready=[]
    for identifier in PROFILES:
        try:
            check_model_assets(project,identifier)
        except SystemExit:
            continue
        ready.append(identifier)
    return ready


def validate_snapshot(folder, profile, *, completed=False):
    folder=Path(folder)
    if completed:
        try:
            record=json.loads((folder/'download.json').read_text())
            if record.get('repository')!=profile['repository'] or record.get('revision')!=profile['revision']:
                raise ValueError('completion marker does not match the pinned model')
        except (OSError,ValueError,TypeError,AttributeError) as error:
            raise ValueError(f'Invalid checkpoint completion marker: {error}') from error
    for relative in profile['required_files']:
        path=folder/relative
        if not path.is_file():
            raise ValueError(f'Missing checkpoint file: {path}')
        if not relative.endswith('.safetensors.index.json'):
            continue
        try:
            manifest=json.loads(path.read_text())
            tensors=manifest['weight_map']
            if not isinstance(tensors,dict) or not tensors:
                raise ValueError('empty weight map')
            for shard in set(tensors.values()):
                if not isinstance(shard,str) or Path(shard).name!=shard or not shard.endswith('.safetensors'):
                    raise ValueError('invalid shard filename')
                if not (path.parent/shard).is_file():
                    raise ValueError(f'missing shard {shard}')
            if path.parent.name in ('text_encoder','transformer'):
                if str(manifest['metadata']['quantization_level'])!=str(profile['quantization']):
                    raise ValueError('unexpected checkpoint quantization')
            if profile['base_model']=='qwen-image-2.1-turbo' and path.parent.name=='text_encoder':
                if not any(key.startswith('visual.') for key in tensors):
                    raise ValueError('editing requires the Qwen vision encoder; text-only exports cannot edit')
        except (OSError,KeyError,TypeError,ValueError) as error:
            raise ValueError(f'Invalid checkpoint index {path}: {error}') from error
