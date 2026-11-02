"""Pinned, shared gallery search assets: SigLIP 2 image–text embeddings (Apache-2.0)."""
from pathlib import Path
from storage_paths import data_directory, require_storage

REPOSITORY = 'google/siglip2-base-patch16-256'
REVISION = '3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab'
WEIGHTS_SHA256 = '6125cacc01fa93bdc98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863'
FILES = ('config.json', 'model.safetensors', 'preprocessor_config.json', 'special_tokens_map.json',
         'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json')
# Stored with every vector, so a different model or revision re-embeds the library instead of mixing spaces.
EMBEDDING_MODEL = f'{REPOSITORY}@{REVISION[:12]}'


def weights_directory(project):
    return data_directory(project) / 'gallery-search' / 'weights'


def check_assets(project):
    """The weights folder, or RuntimeError with the command that fixes it. Never downloads."""
    import json
    folder = weights_directory(project)
    require_storage(folder)
    missing = [name for name in (*FILES, 'download.json') if not (folder / name).is_file()]
    if missing:
        raise RuntimeError('Gallery search model missing; run make download-search. Missing: ' + missing[0])
    record = json.loads((folder / 'download.json').read_text())
    if record.get('revision') != REVISION or record.get('repository') != REPOSITORY:
        raise RuntimeError('Gallery search model revision differs; run make download-search')
    return folder
