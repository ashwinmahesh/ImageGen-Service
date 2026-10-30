"""Read saved generations and cache small previews without loading a GPU model."""
import hashlib
import json
import math
import stat
from pathlib import Path
import re
import tempfile
import threading
import time
import unicodedata
import uuid

from PIL import Image, ImageOps
from storage_paths import require_storage

IDENTIFIER = re.compile(r'[0-9a-f]{32}')
MAX_METADATA_BYTES = 256 * 1024
_thumbnail_slots = threading.BoundedSemaphore(2)
_index_lock = threading.Lock()


class ImageDeleteConflict(Exception):
    pass


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


def normalize_tags(values):
    if not isinstance(values, list) or len(values) > 32:
        raise ValueError('Use a list of at most 32 tags')
    tags, seen = [], set()
    for value in values:
        if not isinstance(value, str) or any(unicodedata.category(c).startswith('C') for c in value) or ',' in value:
            raise ValueError('Tags must be text without commas or control characters')
        tag = ' '.join(unicodedata.normalize('NFC', value).split())
        if not 1 <= len(tag) <= 64:
            raise ValueError('Each tag must contain 1–64 characters')
        if tag.casefold() not in seen:
            seen.add(tag.casefold())
            tags.append(tag)
    return tags


def read_tags(record, metadata):
    values = record.get('tags', metadata.get('tags', [])) if record else metadata.get('tags', [])
    try:
        return normalize_tags(values)
    except ValueError:
        return []


def write_index(root, identifier, path, tags):
    tags = normalize_tags(tags)
    if not valid_identifier(identifier):
        raise ValueError('Invalid image ID')
    folder = root / 'image-index'
    require_storage(root)
    with _index_lock:
        if not Path(path).is_file():
            raise FileNotFoundError('Saved image is unavailable')
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / f'{identifier}.json'
        record = read_metadata(destination) or {}
        record.update(file_path=str(path), tags=tags, tags_updated=int(time.time()))
        with tempfile.NamedTemporaryFile(dir=folder, prefix='index-', suffix='.tmp', mode='w', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(record, stream, ensure_ascii=False)
                stream.write('\n')
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def update_tags(root, image_dir, identifier, tags):
    path = resolve_image(root, image_dir, identifier)
    write_index(root, identifier, path, tags)
    return image_record(root, image_dir, identifier)


def deletion_path(root, image_dir, identifier):
    """Resolve the same gallery ID without following a leaf file symlink."""
    path = resolve_image(root, image_dir, identifier)
    default = image_dir / f'{identifier}.png'
    if default.is_file() and default.resolve() != path:
        raise ImageDeleteConflict('Several files use this image ID. Resolve the duplicate before deleting it.')
    record = read_metadata(root / 'image-index' / f'{identifier}.json') or {}
    candidates = [Path(record['file_path'])] if isinstance(record.get('file_path'), str) else []
    candidates.append(image_dir / f'{identifier}.png')
    for candidate in candidates:
        if candidate.is_absolute() and candidate.resolve() == path:
            if candidate.is_symlink():
                raise ImageDeleteConflict('Cannot delete a linked image; its target was left unchanged.')
            return path
    raise FileNotFoundError('Saved image is unavailable')


def delete_image(root, image_dir, identifier, *, expected_path):
    """Hide a validated image's own files, then erase them; never accept caller paths."""
    require_storage(root)
    with _index_lock:
        path = deletion_path(root, image_dir, identifier)
        if path != expected_path:
            raise ImageDeleteConflict('The saved image location changed; refresh before deleting it.')
        targets = [path, path.with_suffix('.json'), root / 'image-index' / f'{identifier}.json']
        targets.extend((root / 'cache' / 'thumbnails').glob(f'{identifier}-*.webp'))
        targets = list(dict.fromkeys(targets))
        existing = []
        for target in targets:
            require_storage(target.parent)
            try:
                info = target.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ImageDeleteConflict('An image file or companion is not a regular file; nothing was deleted.')
            existing.append(target)
        # Stage in each original directory, including custom destinations on another
        # volume. If staging fails, all originals are still available to restore.
        staged = []
        token = uuid.uuid4().hex
        try:
            for target in existing:
                temporary = target.with_name(f'.delete-{token}-{target.name}')
                target.rename(temporary)
                staged.append((target, temporary))
        except OSError:
            for original, temporary in reversed(staged):
                temporary.replace(original)
            raise
        pending = []
        for _, temporary in staged:
            try:
                temporary.unlink()
            except OSError:
                pending.append(str(temporary))
        return {'id': identifier, 'deleted': True, 'cleanup_pending': pending}


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
        'tags': read_tags(read_metadata(root / 'image-index' / f'{identifier}.json'), data),
        'file_size': stat.st_size, '_version': f'{stat.st_mtime_ns}-{stat.st_size}',
    }


def list_images(root, image_dir, query='', order='newest', limit=24, offset=0, tags=None):
    if not root.is_dir():
        raise OSError('Project storage is unavailable')
    identifiers = set()
    for folder, pattern in [(image_dir, '*.png'), (root / 'image-index', '*.json')]:
        if folder.is_dir():
            identifiers.update(path.stem for path in folder.glob(pattern) if valid_identifier(path.stem))
    records = []
    query = query.casefold().strip()
    required_tags = {tag.casefold() for tag in normalize_tags(tags or [])}
    for identifier in identifiers:
        try:
            record = image_record(root, image_dir, identifier)
        except (OSError, ValueError):
            continue
        searchable = ' '.join(str(record[key]) for key in ['prompt', 'seed', 'size', 'model'] if record[key] is not None)
        searchable += ' ' + ' '.join(record['tags'])
        if not required_tags.issubset({tag.casefold() for tag in record['tags']}):
            continue
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
            with _index_lock:
                # A thumbnail started before deletion must not republish a cache
                # file after the PNG and its old previews have been removed.
                current = resolve_image(root, image_dir, identifier)
                now = current.stat()
                if current != path or (now.st_mtime_ns, now.st_size) != (stat.st_mtime_ns, stat.st_size):
                    raise FileNotFoundError('Saved image changed while preparing its preview')
                temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination
