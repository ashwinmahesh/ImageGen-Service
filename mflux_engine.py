"""Run MLX editing pipelines in an offline child that releases all weights on exit."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from PIL import Image
from config import ROOT, MODEL_DIR, PROFILE
from lora_catalog import effective_prompt
from model_assets import validate_snapshot


class MfluxEngine:
    def __init__(self):
        validate_snapshot(MODEL_DIR,PROFILE,completed=True)
        self.active_log=None

    def generate_image(self, **settings):
        return MfluxResult(settings)

    def progress(self):
        if self.active_log is None or not self.active_log.is_file():
            return None
        try:
            with self.active_log.open('rb') as stream:
                stream.seek(0,2)
                stream.seek(max(0,stream.tell()-65536))
                tail=stream.read().decode('utf-8',errors='replace')
        except OSError:
            return None
        matches=list(re.finditer(r'(\d+)/(\d+)\s+\[[^\]]*?,\s*([0-9.]+)s/it\]',tail))
        if not matches:
            return {'stage':'loading'}
        step,total,seconds=matches[-1].groups()
        step,total,seconds=int(step),int(total),float(seconds)
        return {'stage':'finishing' if step>=total else 'sampling','step':step,'total_steps':total,
                'estimated_remaining_seconds':round(max(0,total-step)*seconds)}


class MfluxResult:
    def __init__(self, settings):
        self.settings=settings

    def save(self, path, overwrite=False):
        path=Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        settings=self.settings
        reference=settings.get('image_path')
        # Neither pipeline exposes our variation-strength convention. References
        # condition the instruction edit; the source is not a partially noised latent.
        if settings.get('image_strength') is not None:
            raise ValueError('image_strength is unsupported for instruction editing')
        module=PROFILE['cli_module'] if reference else PROFILE.get('text_cli_module',PROFILE['cli_module'])
        logs=ROOT/'logs'; logs.mkdir(parents=True,exist_ok=True)
        log_path=logs/(settings.get('job_id',path.stem)+'.log')
        scratch=ROOT/'cache'/'jobs'; scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=path.stem+'-',dir=scratch) as temporary:
            temporary=Path(temporary)
            prompt=temporary/'prompt.txt'
            selection=settings.get('lora')
            prompt.write_text(effective_prompt(settings['prompt'],selection))
            destination=temporary/'result.png'
            command=[sys.executable,'-m',module,'--model',str(MODEL_DIR),
                     '--base-model',PROFILE['base_model'],'--low-ram',
                     '--steps',str(settings['num_inference_steps']),
                     '--width',str(settings['width']),'--height',str(settings['height']),
                     '--seed',str(settings['seed']),'--prompt-file',str(prompt),
                     '--output',str(destination),'--no-exif']
            if PROFILE.get('vae_tiling'):
                command.append('--vae-tiling')
            if reference:
                command+=['--image-paths',str(reference)]
            if settings.get('mask_path'):
                if not PROFILE.get('mask_editing') or not reference:
                    raise ValueError('Masks require a Qwen photo edit and source reference')
                command+=['--mask-image',str(settings['mask_path'])]
            if selection:
                command+=['--lora',selection['file_path'],str(selection['strength']),'--no-bake-lora']
            environment={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                         'TMPDIR':str(scratch),'PYTHONUNBUFFERED':'1'}
            with log_path.open('w') as stream:
                process=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,
                                       timeout=3600,pass_fds=tuple(settings.get('inherited_fds',())))
            if process.returncode:
                raise RuntimeError(f"{PROFILE['name']} exited {process.returncode}; see {log_path}")
            if not destination.is_file():
                raise RuntimeError(f"{PROFILE['name']} produced no image; see {log_path}")
            with Image.open(destination) as result:
                if result.size!=(settings['width'],settings['height']):
                    raise RuntimeError(f'Unexpected output size {result.size}; see {log_path}')
                result.verify()
            destination.replace(path)
