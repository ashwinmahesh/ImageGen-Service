"""Read adapter metadata for any model without importing generator configuration."""
import json
from pathlib import Path
import re
from model_profiles import PROFILES
from storage_paths import data_directory, require_storage


def load_entries(project,directory,*,local=True):
    """Public/legacy presets plus a separate user catalog; public setup stays isolated."""
    project,directory=Path(project),Path(directory)
    path=directory/'catalog.json' if local else project/'lora-catalog.json'
    if not path.is_file():path=project/'lora-catalog.json'
    entries=json.loads(path.read_text()) if path.is_file() else []
    if not isinstance(entries,list):raise ValueError('LoRA catalog must contain a JSON list')
    custom=directory/'custom.json'
    if local and custom.is_file():
        additions=json.loads(custom.read_text())
        if not isinstance(additions,list):raise ValueError('LoRA custom.json must contain a JSON list')
        if any(not isinstance(entry,dict) or not str(entry.get('id','')).startswith('custom-') for entry in additions):raise ValueError('LoRA custom.json entries need IDs in the custom- namespace')
        entries=entries+additions
    if not isinstance(entries,list):raise ValueError('LoRA catalog must contain a JSON list')
    seen=set()
    for entry in entries:
        if not isinstance(entry,dict):raise ValueError('LoRA catalog entries must be objects')
        identifier,filename=entry.get('id',''),entry.get('filename','')
        if not isinstance(identifier,str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}',identifier) or identifier in seen:
            raise ValueError('Invalid or duplicate LoRA ID in the local catalogs')
        if not isinstance(filename,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+\.safetensors',filename):
            raise ValueError('LoRA filenames must be plain .safetensors filenames')
        if not isinstance(entry.get('models'),list) or not all(isinstance(model,str) for model in entry['models']):
            raise ValueError('LoRA models must be a list of model IDs')
        seen.add(identifier)
    return entries


def read_catalog(project,directory,model,*,local=True):
    return [entry for entry in load_entries(project,directory,local=local) if model in entry['models']]


def adapter_path(directory,entry):
    directory=Path(directory)
    path=(directory/entry['filename']).resolve()
    if path.parent!=directory.resolve():
        raise ValueError('LoRA must be inside the configured adapter directory')
    return path


def list_entries(directory,entries):
    # Cheap presence/metadata check. Inference verifies the full checksum before use.
    return [{**entry,'available':adapter_path(directory,entry).is_file() and bool(entry.get('sha256'))}
            for entry in entries]


def model_lora_directory(project,model):
    store=data_directory(project)
    # Both native Chroma backends already share the HD adapter directory.
    directory=store/('chroma1-hd' if PROFILES[model]['backend']=='sd-cpp' else model)/'loras'
    require_storage(directory.resolve())
    return directory.resolve()


def model_loras(project,model):
    directory=model_lora_directory(project,model)
    return directory,list_entries(directory,read_catalog(project,directory,model))
