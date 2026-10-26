"""Pinned model identities and capabilities; no heavyweight imports."""
import os
import sys

PROFILES = {
    'z-image-turbo': dict(name='Z-Image Turbo', backend='mflux', steps=9, max_steps=12,
        repository='filipstrand/Z-Image-Turbo-mflux-4bit', revision='b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b',
        prompt_note='Only the first 512 tokens including chat formatting are processed; excess text is truncated.'),
    'chroma1-hd': dict(name='Chroma1-HD', backend='sd-cpp', steps=40, max_steps=80,
        repository='silveroxides/Chroma1-HD-GGUF', revision='d3b77bf4eb5b84c0b50e9a1e83f6e8074ed85f82',
        filename='Chroma1-HD-Q4_K_M.gguf', sha256='086f561e371d442b82b2c3689149da75046754c62a35293932cda397fc2bf9d2',
        sampling=['--cfg-scale','3','--sampling-method','euler'],
        applied=dict(cfg_scale=3.0, sampler='euler'), prompt_note='Prompts are encoded by the bundled T5 text encoder.'),
    'chroma1-flash': dict(name='Chroma1-Flash', backend='sd-cpp', steps=8, max_steps=40,
        repository='silveroxides/Chroma1-Flash-GGUF', revision='b0a50e822dbd7ade978a76808a395cfb60c8f66f',
        filename='Chroma1-HD-Flash-Q4_K_M.gguf', sha256='59b7ecbc09e8ebe6b260b01da93c9edf465289ef8650e510a637b58a144eaa49',
        sampling=['--cfg-scale','1','--sampling-method','heun','--scheduler','beta',
                  '--extra-sample-args','alpha=0.6,beta=0.6','--flow-shift','1'],
        applied=dict(cfg_scale=1.0, sampler='heun', scheduler='beta', flow_shift=1.0),
        prompt_note='Prompts are encoded by the bundled T5 text encoder.'),
    'flux2-klein-4b': dict(name='FLUX.2 Klein 4B', backend='mflux', steps=4, max_steps=4,
        repository='mflux-community/flux2-klein-4b-mflux-q8', revision='ca49409e4ce21c6717ee87caadd9488e0b15208f',
        base_model='flux2-klein-4b', quantization=8, reference_mode='edit',
        cli_module='mflux.models.flux2.cli.flux2_edit_generate',
        text_cli_module='mflux.models.flux2.cli.flux2_generate',
        required_files=['transformer/model.safetensors.index.json', 'text_encoder/model.safetensors.index.json',
                        'vae/model.safetensors.index.json', 'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', 'tokenizer/chat_template.jinja'],
        prompt_note='Use natural-language descriptions or editing instructions. The text encoder processes up to 512 tokens.',
        license=dict(id='Apache-2.0', url='https://huggingface.co/black-forest-labs/FLUX.2-klein-4B', notice=None)),
    'flux2-klein-9b': dict(name='FLUX.2 Klein 9B', backend='mflux', steps=4, max_steps=4,
        repository='mflux-community/flux2-klein-9b-mflux-q4', revision='fcbc7528d6b4193cca919390a2a8b5b59e601f65',
        base_model='flux2-klein-9b', quantization=4, reference_mode='edit',
        cli_module='mflux.models.flux2.cli.flux2_edit_generate',
        text_cli_module='mflux.models.flux2.cli.flux2_generate',
        required_files=['transformer/model.safetensors.index.json', 'text_encoder/model.safetensors.index.json',
                        'vae/model.safetensors.index.json', 'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', 'tokenizer/chat_template.jinja'],
        prompt_note='Use natural-language descriptions or editing instructions. The text encoder processes up to 512 tokens.',
        license=dict(id='FLUX-Non-Commercial', url='https://huggingface.co/black-forest-labs/FLUX.2-klein-9B',
                     notice='Commercial use of these model weights requires a separate license from Black Forest Labs.')),
    'qwen-image-2.1-turbo': dict(name='Qwen-Image 2.1 Turbo', backend='mflux', steps=8, max_steps=8,
        repository='mflux-community/qwen-image-2-1-turbo-mflux-q4', revision='bda8951ec6b8cc288a961432fe15c22433521a1d',
        base_model='qwen-image-2.1-turbo', quantization=4, reference_mode='edit', vae_tiling=True, mask_editing=True,
        cli_module='mflux.models.qwen21.cli.qwen21_edit_generate',
        required_files=['transformer/model.safetensors.index.json', 'text_encoder/model.safetensors.index.json',
                        'vae/model.safetensors.index.json', 'text_encoder/config.json', 'model_index.json',
                        'scheduler/scheduler_config.json', 'processor/tokenizer.json', 'processor/tokenizer_config.json',
                        'processor/preprocessor_config.json', 'processor/chat_template.jinja'],
        prompt_note='Describe the picture to generate or the change to the reference photo. Use 1024 px or larger for Turbo detail.',
        license=dict(id='Qwen-Research', url='https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE',
                     notice='The Qwen Research License permits research/evaluation use only; commercial model use requires a separate license from Qwen.')),
}
DEFAULT_MODEL='z-image-turbo'
SHARED_ASSETS=[
    ('city96/t5-v1_1-xxl-encoder-gguf','005a6ea51a7d0b84d677b3e633bb52a8c85a83d9',
     't5-v1_1-xxl-encoder-Q4_K_M.gguf','6be2b0b7e2de7cf2919340c88cb802a103a997ce46c53131cec91958c1db1af4'),
    ('lodestones/Chroma1-HD','0e0c60ece1e82b17cb7f77342d765ba5024c40c0',
     'vae/diffusion_pytorch_model.safetensors','f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3'),
]
RUNTIME_RELEASE='master-1005-ea209db'
RUNTIME_URL='https://github.com/leejet/stable-diffusion.cpp/releases/download/master-1005-ea209db/sd-master-ea209db-bin-Darwin-macOS-26.6.2-arm64.zip'
RUNTIME_SHA256='d183e8d87a1d1b26f376ff68f4d15f3322648805687cb01c25a97c9ac5acdc50'


def model_details(identifier):
    """Model-specific public metadata, kept under the exact model ID by callers."""
    profile=PROFILES[identifier]
    return {key:profile[key] for key in ('quantization','license') if key in profile}


def _editing_label(profile):
    if profile.get('mask_editing'):
        return ' (edit selected area enabled)'
    if profile.get('reference_mode')=='edit':
        return ' (edits enabled)'
    return ''


def _show_unavailable(identifiers):
    missing=[name for name in PROFILES if name not in identifiers]
    if missing:
        print('\nNot set up (not selectable):')
        for name in missing:
            profile=PROFILES[name]
            print(f"  - {profile['name']}{_editing_label(profile)} ({name}) — make setup MODEL={name}")
        print('Run a model\'s setup command to add it to the numbered options, then run make serve again.')


def select_model(explicit=None, *, interactive=None, prompt=input, choices=None, setup_complete=None, allow_cancel=False):
    identifiers=tuple(PROFILES if choices is None else (name for name in PROFILES if name in choices))
    selected=explicit or os.environ.get('IMAGE_MODEL') or os.environ.get('MODEL')
    if interactive is None:
        interactive=sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get('NONINTERACTIVE') != '1'
    if not identifiers:
        if interactive:
            print('No image models are set up.')
            _show_unavailable(identifiers)
        raise ValueError('No models are set up. Run make setup first, then make serve.')
    default=DEFAULT_MODEL if DEFAULT_MODEL in identifiers else identifiers[0]
    if not selected and interactive:
        print('Choose an image model:' if choices is None else 'Choose a set up image model:')
        completed=setup_complete() if choices is None and setup_complete else ()
        for index,identifier in enumerate(identifiers,1):
            profile=PROFILES[identifier]
            label=_editing_label(profile) if choices is not None else ''
            if identifier in completed:
                label += ' [SETUP COMPLETE]'
            print(f"  {index}. {profile['name']}{label} ({identifier})")
        if choices is not None:
            _show_unavailable(identifiers)
        if allow_cancel:
            print('  q. Cancel setup')
        answer=prompt(f"Model [{identifiers.index(default)+1} / {PROFILES[default]['name']}]: ").strip()
        if allow_cancel and answer.lower() in ('q','quit','cancel'):
            raise KeyboardInterrupt
        selected=dict(enumerate(identifiers,1)).get(int(answer)) if answer.isdigit() else answer
        if answer and not selected:
            raise ValueError('Choose a listed model number or ID')
    selected=selected or default
    if selected not in PROFILES:
        raise ValueError('Unknown model; choose '+', '.join(PROFILES))
    if selected not in identifiers:
        raise ValueError(f'{selected} is not set up; run make setup MODEL={selected} first')
    return selected
