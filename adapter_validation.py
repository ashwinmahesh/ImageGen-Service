"""Inspect data-only adapter headers without loading tensors or a model runtime."""
import json
from pathlib import Path
from safetensors import safe_open

CUSTOM_LORA_MODELS=('z-image-turbo','chroma1-hd','chroma1-flash')


def inspect_adapter(source,model):
    if model not in CUSTOM_LORA_MODELS:
        raise ValueError('Custom LoRA setup currently supports Z-Image Turbo and Chroma HD/Flash; Klein/Qwen formats are not validated by this importer')
    source=Path(source)
    try:
        with safe_open(str(source),framework='numpy') as weights:
            keys=[key.lower() for key in weights.keys()]
            info=weights.metadata() or {}
    except Exception as error:
        raise ValueError('Invalid or incomplete .safetensors file; download the adapter weights, not an HTML page or a checkpoint in another format') from error
    if not keys or not any('lora' in key for key in keys):
        raise ValueError('This file does not contain LoRA tensors')
    metadata=json.dumps({key:info.get(key) for key in ('ss_base_model_version','ss_model_type','modelspec.architecture')}).lower()
    if model=='z-image-turbo':
        if any(word in metadata for word in ('chroma','illustrious','sdxl','flux','qwen')):
            raise ValueError('Wrong model family: choose the Z-Image Turbo adapter version')
        if not any('layers.' in key and '.attention.' in key for key in keys):
            raise ValueError('Unrecognized Z-Image transformer adapter tensors')
    elif any(word in metadata for word in ('zimage','z_image','illustrious','sdxl','qwen','flux2','klein')) or not any('double_blocks' in key or 'single_blocks' in key or 'transformer_blocks' in key for key in keys):
        raise ValueError('Unrecognized Chroma transformer adapter tensors')
    return {'tensor_count':len(keys),'metadata':info}
