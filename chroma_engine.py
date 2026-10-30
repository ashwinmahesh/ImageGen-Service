"""Bridge the same image API to the pinned Metal stable-diffusion.cpp CLI."""
import os
import re
from pathlib import Path
import subprocess
import tempfile
from PIL import Image, ImageOps
from lora_catalog import effective_prompt
from config import ROOT, MODEL_FILE, T5_FILE, VAE_FILE, RUNTIME_DIR, RUNTIME_RELEASE, PROFILE


class ChromaEngine:
    """Loads GPU weights only during a request, then releases them on CLI exit."""
    def __init__(self):
        self.binary = RUNTIME_DIR / 'sd-cli'
        self.active_log = None
        for path in [self.binary,MODEL_FILE,T5_FILE,VAE_FILE]:
            if not path.is_file():
                raise RuntimeError(f'Missing {path}; run make download')

    def generate_image(self, **settings):
        return ChromaResult(self.binary, settings)

    def progress(self):
        path = self.active_log
        if path is None or not path.is_file():
            return None
        try:
            with path.open('rb') as source:
                source.seek(0,2)
                source.seek(max(0,source.tell()-65536))
                tail=source.read().decode('utf-8',errors='replace')
        except OSError:
            return None
        matches=list(re.finditer(r'(\d+)/(\d+)\s+-\s+([0-9.]+)s/it',tail))
        if not matches:
            return {'stage':'loading'}
        last=matches[-1]
        step, total, seconds=int(last[1]),int(last[2]),float(last[3])
        return {'stage':'finishing' if step >= total else 'sampling',
                'step':step,'total_steps':total,
                'estimated_remaining_seconds':round(max(0,total-step)*seconds)}


class ChromaResult:
    def __init__(self,binary,settings):
        self.binary=binary
        self.settings=settings

    def save(self,path,overwrite=False):
        path=Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        logs=ROOT/'logs'
        logs.mkdir(exist_ok=True)
        log_path=logs/(self.settings.get('job_id',path.stem)+'.log')
        scratch=ROOT/'cache'/'jobs'
        scratch.mkdir(parents=True,exist_ok=True)
        s=self.settings
        with tempfile.TemporaryDirectory(prefix=path.stem+'-',dir=scratch) as temporary:
            temporary=Path(temporary)
            prompt=temporary/'prompt.txt'
            selection = s.get('lora')
            text = effective_prompt(s['prompt'], selection)
            if selection:
                name = Path(selection['file_path']).stem
                text += f" <lora:{name}:{selection['strength']}>"
            prompt.write_text(text)
            destination=temporary/'result.png'
            command=[str(self.binary), '--diffusion-model',str(MODEL_FILE),
                     '--t5xxl',str(T5_FILE),'--vae',str(VAE_FILE),
                     '--backend','metal','--max-vram','14','--flash-attn','on',
                     '--batched-cfg','off','--mmap',
                     '--vae-tiling','--vae-tile-size','512x512',
                     *PROFILE['sampling'],
                     '--steps',str(s['num_inference_steps']),
                     '--width',str(s['width']),'--height',str(s['height']),
                     '--seed',str(s['seed']),'--prompt-file',str(prompt),
                     '--output',str(destination),'--disable-image-metadata']
            if selection:
                command += ['--lora-model-dir', str(Path(selection['file_path']).parent),
                            '--lora-apply-mode', 'at_runtime']
            if s.get('preview_dir'):
                # About eight decoded previews per image, whatever the step count.
                command += ['--preview', 'vae', '--preview-path', str(Path(s['preview_dir'])/'preview.png'),
                            '--preview-interval', str(max(1, s['num_inference_steps']//8))]
            if s.get('image_path') and s.get('image_strength',0)>0:
                # Keep the API convention: higher means MORE original preserved.
                # sd.cpp uses the inverse (denoising strength), so translate it.
                reference=temporary/'reference.png'
                with Image.open(s['image_path']) as source:
                    normalized=ImageOps.exif_transpose(source).convert('RGB')
                    normalized.resize((s['width'],s['height']),Image.Resampling.LANCZOS).save(reference)
                command += ['--init-img',str(reference),'--strength',str(1-s['image_strength'])]
            env=os.environ.copy()
            env['TMPDIR']=str(scratch)
            with log_path.open('w') as log:
                log.write(f'Runtime: {RUNTIME_RELEASE}\n')
                log.flush()
                process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600,pass_fds=tuple(s.get('inherited_fds',())))
            if process.returncode != 0:
                raise RuntimeError(f'Chroma runtime exited {process.returncode}; see {log_path}')
            if not destination.is_file():
                raise RuntimeError(f'Chroma produced no image; see {log_path}')
            with Image.open(destination) as result:
                if result.size != (s['width'],s['height']):
                    raise RuntimeError(f'Unexpected output size {result.size}; see {log_path}')
                result.verify()
            destination.replace(path)
