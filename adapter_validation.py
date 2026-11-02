"""Inspect data-only adapter headers without loading tensors or a model runtime."""
import json
from pathlib import Path
from safetensors import safe_open

CUSTOM_LORA_MODELS=('z-image-turbo','chroma1-hd','chroma1-flash','flux2-klein-4b','flux2-klein-9b','qwen-image-2.1-turbo')
MFLUX_NAMES={'z-image-turbo':'Z-Image Turbo','flux2-klein-4b':'FLUX.2 Klein 4B','flux2-klein-9b':'FLUX.2 Klein 9B','qwen-image-2.1-turbo':'Qwen-Image 2.1'}
# Transformer width of each MFLUX model; an adapter's attention input size must match it.
HIDDEN_WIDTH={'z-image-turbo':3840,'flux2-klein-4b':3072,'flux2-klein-9b':4096,'qwen-image-2.1-turbo':4096}
WIDTH_KEYS=('attention.to_q.','attn.to_q.','attn_to_q.','img_attn.qkv.','img_attn_qkv.')
# Dual-stream Qwen-Image 1.x/Edit names; 2.1 is single-stream and cannot use those adapters.
QWEN1_KEYS=('add_q_proj','add_k_proj','txt_mlp','txt_mod','img_mod')


def adapter_width(weights,names):
    """Input width of the first attention projection: a LoRA down matrix, or LoKr factors multiplied."""
    for key,name in names.items():
        if not any(part in key for part in WIDTH_KEYS):continue
        if key.endswith(('lora_a.weight','lora_a.default.weight','lora_down.weight')):
            return weights.get_slice(name).get_shape()[-1]
        if key.endswith('.lokr_w1') and key[:-1]+'2' in names:
            return weights.get_slice(name).get_shape()[-1]*weights.get_slice(names[key[:-1]+'2']).get_shape()[-1]
    return None


def inspect_adapter(source,model):
    if model not in CUSTOM_LORA_MODELS:
        raise ValueError('Custom LoRA setup supports: '+', '.join(CUSTOM_LORA_MODELS))
    source=Path(source)
    try:
        with safe_open(str(source),framework='numpy') as weights:
            names={key.lower():key for key in weights.keys()}
            keys=list(names)
            info=weights.metadata() or {}
            width=adapter_width(weights,names) if model in HIDDEN_WIDTH else None
    except Exception as error:
        raise ValueError('Invalid or incomplete .safetensors file; download the adapter weights, not an HTML page or a checkpoint in another format') from error
    # Z-Image and FLUX.2 also accept LyCORIS LoKr (lokr_w1/lokr_w2), applied through the same adapter path.
    lokr=model in ('z-image-turbo','flux2-klein-4b','flux2-klein-9b') and any('.lokr_w1' in key for key in keys)
    if not keys or not (lokr or any('lora' in key for key in keys)):
        raise ValueError('This file does not contain LoRA tensors')
    has=lambda *parts:any(part in key for key in keys for part in parts)
    if model in MFLUX_NAMES and has('lora_magnitude_vector','dora_scale'):
        raise ValueError('DoRA adapters are not supported here; choose a standard LoRA version')
    metadata=json.dumps({key:info.get(key) for key in ('ss_base_model_version','ss_model_type','modelspec.architecture')}).lower()
    mentions=lambda *words:any(word in metadata for word in words)
    if model=='z-image-turbo':
        if mentions('chroma','illustrious','sdxl','flux','qwen'):
            raise ValueError('Wrong model family: choose the Z-Image Turbo adapter version')
        if not any('layers.' in key and '.attention.' in key for key in keys):
            raise ValueError('Unrecognized Z-Image transformer adapter tensors')
    elif model.startswith('flux2-klein'):
        if mentions('zimage','z_image','chroma','illustrious','sdxl','qwen','flux1','flux.1','flux-1') or has('img_mod','add_q_proj','txt_mod','gate_layer'):
            raise ValueError('Wrong model family: choose the FLUX.2 Klein adapter version (FLUX.1, Chroma and Qwen adapters do not fit)')
        if not has('double_blocks','single_blocks','transformer_blocks','double_stream_modulation','single_stream_modulation'):
            raise ValueError('Unrecognized FLUX.2 Klein transformer adapter tensors')
    elif model=='qwen-image-2.1-turbo':
        if mentions('zimage','z_image','chroma','illustrious','sdxl','flux','klein'):
            raise ValueError('Wrong model family: choose the Qwen-Image 2.1 adapter version')
        if has(*QWEN1_KEYS):
            raise ValueError('This adapter is for the original Qwen-Image or Qwen-Image-Edit; Qwen-Image 2.1 needs a LoRA trained for 2.1')
        if not has('transformer_blocks.') or has('single_transformer_blocks','double_blocks','single_blocks'):
            raise ValueError('Unrecognized Qwen-Image 2.1 transformer adapter tensors')
        if has('lora_down','lora_up','lokr_'):
            raise ValueError('Qwen-Image 2.1 adapters must use the PEFT/diffusers lora_A/lora_B format')
    elif mentions('zimage','z_image','illustrious','sdxl','qwen','flux2','klein') or not has('double_blocks','single_blocks','transformer_blocks'):
        raise ValueError('Unrecognized Chroma transformer adapter tensors')
    if width is not None and width!=HIDDEN_WIDTH[model]:
        raise ValueError(f'Wrong model size: this adapter targets a {width}-wide transformer, but {MFLUX_NAMES[model]} is {HIDDEN_WIDTH[model]}-wide. Choose the version trained for {MFLUX_NAMES[model]}')
    return {'tensor_count':len(keys),'metadata':info}
