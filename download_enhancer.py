"""Download the shared prompt LLM once; never called implicitly while serving."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from download import digest
from enhancer_profile import REPOSITORY, REVISION, FILES, WEIGHTS_SHA256, weights_directory
from storage_paths import require_storage


def main():
    folder = weights_directory(Path(__file__).resolve().parent)
    require_storage(folder)
    folder.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPOSITORY, revision=REVISION, local_dir=str(folder),
                      allow_patterns=list(FILES), max_workers=3)
    for name in FILES:
        if not (folder/name).is_file():
            raise RuntimeError('Incomplete prompt model download: ' + name)
    if digest(folder/'model.safetensors') != WEIGHTS_SHA256:
        raise RuntimeError('Prompt model checksum mismatch; remove the corrupt file and retry make download-llm')
    record = {'repository': REPOSITORY, 'revision': REVISION, 'license': 'Apache-2.0',
              'weights_sha256': WEIGHTS_SHA256, 'runtime': 'mlx-lm', 'task': 'text-only prompt expansion'}
    temporary = folder/'download.tmp'
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(folder/'download.json')
    print(f'Prompt expander ready: {folder}')


if __name__ == '__main__':
    main()
