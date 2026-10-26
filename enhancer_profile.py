"""Pinned, shared, text-only prompt-expansion assets (Apache-2.0)."""
from pathlib import Path
from storage_paths import data_directory, require_storage

REPOSITORY = 'mlx-community/Qwen3.5-2B-4bit'
REVISION = '674aaa7240b91e8012fcad5d791b7dfe5ba90207'
WEIGHTS_SHA256 = '713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5'
FILES = ('config.json', 'model.safetensors', 'model.safetensors.index.json',
         'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'vocab.json')


def weights_directory(project):
    return data_directory(project) / 'prompt-enhancer' / 'weights'


def check_assets(project):
    import json
    folder = weights_directory(project)
    require_storage(folder)
    missing = [name for name in (*FILES, 'download.json') if not (folder/name).is_file()]
    if missing:
        raise RuntimeError('Prompt expander assets missing; run make download-llm (or make setup). Missing: ' + missing[0])
    record = json.loads((folder/'download.json').read_text())
    if record.get('revision') != REVISION or record.get('repository') != REPOSITORY:
        raise RuntimeError('Prompt expander revision differs; run make download-llm')
    return folder
