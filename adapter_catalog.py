"""Read adapter metadata for any model without importing generator configuration."""
import json
from pathlib import Path
import re
from model_profiles import PROFILES
from storage_paths import data_directory, require_storage


def read_catalog(project,directory,model,*,local=True):
    project,directory=Path(project),Path(directory)
    path=directory/'catalog.json' if local else project/'lora-catalog.json'
    if not path.is_file():
        path=project/'lora-catalog.json'
    if not path.is_file():
        return []
    entries=json.loads(path.read_text())
    seen=set()
    for entry in entries:
        identifier,filename=entry['id'],entry['filename']
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}',identifier) or identifier in seen:
            raise ValueError('Invalid or duplicate LoRA ID in catalog.json')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+\.safetensors',filename):
            raise ValueError('LoRA filenames must be plain .safetensors filenames')
        seen.add(identifier)
    return [entry for entry in entries if model in entry['models']]


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


def model_loras(project,model):
    store=data_directory(project)
    # Both native Chroma backends already share the HD adapter directory.
    directory=store/('chroma1-hd' if PROFILES[model]['backend']=='sd-cpp' else model)/'loras'
    require_storage(directory.resolve())
    return directory.resolve(),list_entries(directory,read_catalog(project,directory,model))
