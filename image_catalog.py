"""Read saved generations and cache small previews without loading a GPU model."""
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import threading

from PIL import Image, ImageOps

IDENTIFIER = re.compile(r'[0-9a-f]{32}')
MAX_METADATA_BYTES = 256 * 1024
_thumbnail_slots = threading.BoundedSemaphore(2)


def valid_identifier(value):
    return bool(IDENTIFIER.fullmatch(value))


def read_metadata(path):
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            return None
        def reject_constant(value):
            raise ValueError('Non-finite JSON number')
        def finite_float(value):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError('Non-finite JSON number')
            return number
        data = json.loads(path.read_text(), parse_constant=reject_constant, parse_float=finite_float)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def resolve_image(root, image_dir, identifier):
    if not valid_identifier(identifier):
        raise FileNotFoundError('Invalid image ID')
    record = read_metadata(root / 'image-index' / f'{identifier}.json')
    candidates = []
    if record and isinstance(record.get('file_path'), str):
        candidates.append(Path(record['file_path']))
    candidates.append(image_dir / f'{identifier}.png')
    for candidate in candidates:
        try:
            if not candidate.is_absolute():
                continue
            path = candidate.resolve()
            if path.stem == identifier and path.suffix.lower() == '.png' and path.is_file() and path.stat().st_size:
                return path
        except (OSError, ValueError):
            continue
    raise FileNotFoundError('Saved image is unavailable')


def _number(value, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        finite = math.isfinite(value)
    except OverflowError:
        return None
    if not finite or (integer and not isinstance(value, int)):
        return None
    return value


def image_record(root, image_dir, identifier):
    path = resolve_image(root, image_dir, identifier)
    stat = path.stat()
    metadata_path = path.with_suffix('.json')
    metadata = read_metadata(metadata_path)
    data = metadata or {}
    size = data.get('size')
    width = height = None
    if isinstance(size, str) and re.fullmatch(r'[1-9][0-9]{0,4}x[1-9][0-9]{0,4}', size):
        width, height = map(int, size.split('x'))
    else:
        try:
            with Image.open(path) as photo:
                width, height = photo.size
            size = f'{width}x{height}'
        except (OSError, ValueError, Image.DecompressionBombError):
            size = None
    created = _number(data.get('created'))
    if created is None or not 0 < created <= 253402300799:
        created = stat.st_mtime
    adapter = data.get('lora')
    if isinstance(adapter, dict) and isinstance(adapter.get('id'), str):
        adapter = {'id': adapter['id'], 'name': str(adapter.get('name', adapter['id'])),
                   'strength': _number(adapter.get('strength'))}
    else:
        adapter = None
    prompt = data.get('prompt') if isinstance(data.get('prompt'), str) else None
    return {
        'id': identifier, 'file_path': str(path),
        'metadata_path': str(metadata_path) if metadata is not None else None,
        'metadata_available': metadata is not None, 'metadata': metadata,
        'created': int(created), 'prompt': prompt,
        'model': data.get('model') if isinstance(data.get('model'), str) else None,
        'size': size, 'width': width, 'height': height,
        'seed': _number(data.get('seed'), integer=True),
        'steps': _number(data.get('steps'), integer=True),
        'generation_seconds': _number(data.get('seconds')), 'lora': adapter,
        'file_size': stat.st_size, '_version': f'{stat.st_mtime_ns}-{stat.st_size}',
    }


def list_images(root, image_dir, query='', order='newest', limit=24, offset=0):
    if not root.is_dir():
        raise OSError('Project storage is unavailable')
    identifiers = set()
    for folder, pattern in [(image_dir, '*.png'), (root / 'image-index', '*.json')]:
        if folder.is_dir():
            identifiers.update(path.stem for path in folder.glob(pattern) if valid_identifier(path.stem))
    records = []
    query = query.casefold().strip()
    for identifier in identifiers:
        try:
            record = image_record(root, image_dir, identifier)
        except (OSError, ValueError):
            continue
        searchable = ' '.join(str(record[key]) for key in ['prompt', 'seed', 'size', 'model'] if record[key] is not None)
        if record['lora']:
            searchable += ' ' + record['lora']['id'] + ' ' + record['lora']['name']
        if query and query not in searchable.casefold():
            continue
        records.append(record)
    records.sort(key=lambda item: (item['created'], item['id']), reverse=order == 'newest')
    return records[offset:offset + limit], len(records)


def thumbnail(root, image_dir, identifier):
    path = resolve_image(root, image_dir, identifier)
    stat = path.stat()
    version = hashlib.sha256(f'{path}:{stat.st_mtime_ns}:{stat.st_size}'.encode()).hexdigest()[:20]
    folder = root / 'cache' / 'thumbnails'
    destination = folder / f'{identifier}-{version}.webp'
    if destination.is_file():
        return destination
    with _thumbnail_slots:
        if destination.is_file():
            return destination
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(path) as source:
                if source.width * source.height > 40_000_000:
                    raise ValueError('Image is too large to preview')
                preview = ImageOps.exif_transpose(source).convert('RGB')
                preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
        except Image.DecompressionBombError as error:
            raise ValueError('Image is too large to preview') from error
        with tempfile.NamedTemporaryFile(dir=folder, prefix='thumbnail-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            preview.save(temporary, format='WEBP', quality=82)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination
