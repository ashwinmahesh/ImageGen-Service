"""Download only the pinned files used by this installation; verify SHA256."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from storage_paths import require_storage
import urllib.request
import zipfile
from config import ASSETS, MODEL_DIR, RUNTIME_DIR, RUNTIME_RELEASE, RUNTIME_URL, RUNTIME_SHA256, PROFILE, MODEL_ID, MODEL_REVISION, SHARED_ROOT, SHARED_ASSETS
from huggingface_hub import hf_hub_download
from model_assets import validate_snapshot


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024),b''):
            value.update(chunk)
    return value.hexdigest()


def download_asset(asset, directory=None):
    directory = MODEL_DIR if directory is None else directory
    require_storage(directory)
    directory.mkdir(parents=True, exist_ok=True)
    repo, revision, filename, expected = asset
    path = hf_hub_download(repo,filename,revision=revision,local_dir=str(directory))
    if digest(path) != expected:
        raise RuntimeError(f'Checksum mismatch: {filename}')
    print(f'Verified {filename}',flush=True)
    return {'repository':repo,'revision':revision,'file':filename,'sha256':expected}


def download_runtime():
    RUNTIME_DIR.mkdir(parents=True,exist_ok=True)
    archive = RUNTIME_DIR / f'{RUNTIME_RELEASE}.zip'
    if not archive.exists() or digest(archive) != RUNTIME_SHA256:
        temporary = archive.with_suffix('.partial')
        with urllib.request.urlopen(RUNTIME_URL) as source, temporary.open('wb') as target:
            while chunk := source.read(1024*1024):
                target.write(chunk)
        if digest(temporary) != RUNTIME_SHA256:
            raise RuntimeError('Runtime archive checksum mismatch')
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (RUNTIME_DIR/member.filename).resolve()
            if not target.is_relative_to(RUNTIME_DIR.resolve()):
                raise RuntimeError('Unsafe runtime archive path')
        bundle.extractall(RUNTIME_DIR)
    for path in RUNTIME_DIR.rglob('*'):
        if path.is_file() and (path.name.startswith('sd-') or path.suffix in ['.dylib','.so']):
            path.chmod(0o755)
    print('Verified and installed runtime '+RUNTIME_RELEASE,flush=True)


def main():
    require_storage(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    if PROFILE['backend']=='mflux':
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_ID,revision=MODEL_REVISION,local_dir=str(MODEL_DIR),
            allow_patterns=['text_encoder/*','tokenizer/*','transformer/*','vae/*','processor/*',
                            'scheduler/*','model_index.json','manifest.json','README.md','LICENSE*'],max_workers=3)
        if PROFILE.get('cli_module'):
            validate_snapshot(MODEL_DIR,PROFILE)
        record={'repository':MODEL_ID,'revision':MODEL_REVISION}
    else:
        # Flash needs the shared encoder/VAE, but does not download the HD diffusion model.
        require_storage(SHARED_ROOT)
        with ThreadPoolExecutor(max_workers=4) as pool:
            runtime=pool.submit(download_runtime)
            own=list(pool.map(download_asset,ASSETS))
            shared=list(pool.map(lambda asset: download_asset(asset,SHARED_ROOT/'weights'),SHARED_ASSETS))
            runtime.result()
        record={'assets':own,'shared_assets':shared,'runtime_release':RUNTIME_RELEASE,'runtime_sha256':RUNTIME_SHA256}
    marker=MODEL_DIR/'download.json'
    temporary=marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(marker)
    print(f'Model ready: {MODEL_DIR}')


if __name__=='__main__':
    main()
