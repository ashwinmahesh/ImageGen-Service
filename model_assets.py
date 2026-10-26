"""Cheap local preflight for pinned, pre-quantized MLX editing checkpoints."""
import json
from pathlib import Path


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
