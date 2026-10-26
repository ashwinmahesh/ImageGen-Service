"""Tiny readiness fixtures; never downloaded, executed or loaded as model weights."""
import json
from pathlib import Path
from model_profiles import PROFILES,SHARED_ASSETS


def setup_model(project,identifier):
    project=Path(project);profile=PROFILES[identifier];folder=project/identifier/'weights';folder.mkdir(parents=True,exist_ok=True)
    (folder/'download.json').write_text(json.dumps({'repository':profile['repository'],'revision':profile['revision']}))
    if profile['backend']=='sd-cpp':
        paths=[folder/profile['filename'],project/'runtime/sd-cli',project/'runtime/libstable-diffusion.dylib',*[project/'chroma1-hd/weights'/asset[2] for asset in SHARED_ASSETS]]
        for path in paths:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture only')
    elif profile.get('cli_module'):
        for name in profile['required_files']:
            path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
            value={}
            if name.endswith('.safetensors.index.json'):
                value={'weight_map':{'visual.fixture':'weights.safetensors'},'metadata':{'quantization_level':profile['quantization']}}
                (path.parent/'weights.safetensors').write_bytes(b'fixture only')
            path.write_text(json.dumps(value))
    else:
        for name in ('transformer','text_encoder','tokenizer','vae'):(folder/name).mkdir(exist_ok=True)
