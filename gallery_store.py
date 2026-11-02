"""The gallery's source of truth: each saved image's settings, tags and location in SQLite.

Generation registers every image here as it is saved. The JSON sidecar beside each PNG is still
written as a portable copy for people and tools, but listing, search and tags read only this
database. Search uses SQLite FTS5: every word in the query must match the start of a word in the
prompt, tags, model, adapter names, seed or size. Images removed outside the service are hidden
when a page finds their file missing, and shown again if the file comes back.

Each image can also have a SigLIP 2 vector (gallery_search.py) for finding similar images:
search() then lists the exact matches first and the given ranked similar images after them.

python gallery_store.py imports images saved before the database existed (or copied into a
model's images/ folder by hand); the generator also does this once per model folder at startup.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import threading
import time

import numpy as np

import image_catalog
from storage_paths import require_storage

_initialized = set()
_initialize_lock = threading.Lock()


def gallery_database(project):
    """Project-local, next to the queue database, so gallery queries never wait on the external image drive."""
    return Path(project).resolve() / 'state' / 'gallery' / 'images.sqlite3'


def search_terms(query):
    """FTS5 query for the words in query: each must prefix-match a word, in any order."""
    words = [word for word in query.casefold().split() if re.search(r'\w', word)]
    return ' '.join('"' + word.replace('"', '""') + '"*' for word in words)


def search_text(record, tags):
    parts = [record.get('prompt'), record.get('seed'), record.get('size'), record.get('model'), *tags]
    for adapter in record.get('loras') or ([record['lora']] if record.get('lora') else []):
        parts += [adapter['id'], adapter['name']]
    return ' '.join(str(part) for part in parts if part is not None)


def public_record(row, info, *, detail=False):
    """API fields for one database row; info is the PNG's current stat, read only for rows being shown."""
    path = Path(row['file_path'])
    metadata = json.loads(row['metadata']) if row['metadata'] else None
    record = {'id': row['id'], 'file_path': str(path), 'model': row['model'],
              'metadata_path': str(path.with_suffix('.json')) if metadata is not None else None,
              'metadata_available': metadata is not None, 'created': int(row['created']),
              'prompt': row['prompt'], 'size': row['size'], 'width': row['width'], 'height': row['height'],
              'seed': row['seed'], 'steps': row['steps'], 'generation_seconds': row['generation_seconds'],
              'lora': json.loads(row['lora']) if row['lora'] else None,
              'loras': json.loads(row['loras']) if row['loras'] else None,
              'tags': json.loads(row['tags']), 'file_size': info.st_size,
              '_version': f'{info.st_mtime_ns}-{info.st_size}'}
    if row.get('_match'):
        record['match'] = row['_match']
    if row.get('_similarity') is not None:
        record['similarity'] = round(float(row['_similarity']), 4)
    if detail:
        record['metadata'] = metadata
    return record


class GalleryStore:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def connection(self, write=False):
        try:
            require_storage(self.path)
        except RuntimeError as error:
            raise sqlite3.OperationalError('Gallery storage is unavailable') from error
        self._initialize()
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            if write:
                connection.execute('BEGIN IMMEDIATE')
            yield connection
            if write:
                connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self):
        key = str(self.path)
        if key in _initialized:
            return
        with _initialize_lock:
            if key in _initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=10)
            try:
                if connection.execute('PRAGMA user_version').fetchone()[0] not in (0, 1, 2):
                    raise RuntimeError('Unsupported gallery database version; use a compatible server')
                connection.executescript('''
                    PRAGMA journal_mode=DELETE;
                    CREATE TABLE IF NOT EXISTS images (
                        number INTEGER PRIMARY KEY,
                        id TEXT NOT NULL UNIQUE,
                        root TEXT NOT NULL,
                        model TEXT,
                        file_path TEXT NOT NULL,
                        created REAL NOT NULL,
                        prompt TEXT, seed INTEGER, steps INTEGER, size TEXT, width INTEGER, height INTEGER,
                        generation_seconds REAL, lora TEXT, loras TEXT,
                        metadata TEXT,
                        tags TEXT NOT NULL DEFAULT '[]',
                        missing INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE INDEX IF NOT EXISTS images_order ON images(missing, created, id);
                    CREATE TABLE IF NOT EXISTS image_tags (
                        image INTEGER NOT NULL, tag TEXT NOT NULL, PRIMARY KEY(image, tag)
                    ) WITHOUT ROWID;
                    CREATE INDEX IF NOT EXISTS image_tags_by_tag ON image_tags(tag, image);
                    CREATE VIRTUAL TABLE IF NOT EXISTS images_search USING fts5(text, tokenize='unicode61 remove_diacritics 2');
                    CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    -- One embedding per image and model; a NULL vector marks a file that could not be read.
                    CREATE TABLE IF NOT EXISTS image_vectors (
                        image INTEGER PRIMARY KEY, model TEXT NOT NULL, vector BLOB
                    );
                    PRAGMA user_version=2;
                ''')
            finally:
                connection.close()
            _initialized.add(key)

    @staticmethod
    def _index(db, number, record, tags):
        db.execute('DELETE FROM image_tags WHERE image=?', (number,))
        db.executemany('INSERT OR IGNORE INTO image_tags(image, tag) VALUES (?, ?)', [(number, tag.casefold()) for tag in tags])
        db.execute('DELETE FROM images_search WHERE rowid=?', (number,))
        db.execute('INSERT INTO images_search(rowid, text) VALUES (?, ?)', (number, search_text(record, tags)))

    def register(self, identifier, *, root, origin, path, metadata, tags, replace=True):
        """Add or update one saved image. replace=False keeps an existing row (used by imports)."""
        path = Path(path)
        record = image_catalog.describe(path, metadata)
        record['model'] = record['model'] or origin
        tags = image_catalog.normalize_tags(tags or [])
        values = {'id': identifier, 'root': str(root), 'model': record['model'], 'file_path': str(path),
                  'created': record['created'], 'prompt': record['prompt'], 'seed': record['seed'],
                  'steps': record['steps'], 'size': record['size'], 'width': record['width'],
                  'height': record['height'], 'generation_seconds': record['generation_seconds'],
                  'lora': json.dumps(record['lora']) if record['lora'] else None,
                  'loras': json.dumps(record['loras']) if record['loras'] else None,
                  'metadata': json.dumps(metadata, ensure_ascii=False) if metadata is not None else None,
                  'tags': json.dumps(tags, ensure_ascii=False)}
        with self.connection(write=True) as db:
            existing = db.execute('SELECT number FROM images WHERE id=?', (identifier,)).fetchone()
            if existing and not replace:
                return False
            columns = ', '.join(values)
            if existing:
                db.execute(f'UPDATE images SET {", ".join(f"{key}=:{key}" for key in values)}, missing=0 WHERE id=:id', values)
                number = existing['number']
            else:
                number = db.execute(f'INSERT INTO images({columns}) VALUES ({", ".join(":" + key for key in values)})', values).lastrowid
            self._index(db, number, record, tags)
        return True

    def get(self, identifier):
        with self.connection() as db:
            row = db.execute('SELECT * FROM images WHERE id=?', (identifier,)).fetchone()
        return dict(row) if row else None

    def known(self, identifiers):
        with self.connection() as db:
            return {row['id'] for row in db.execute('SELECT id FROM images')} & set(identifiers)

    @staticmethod
    def _filters(query, tags):
        """(visible-and-tagged condition, its params, FTS terms) for a listing."""
        where, params = ['missing=0'], []
        wanted = sorted({tag.casefold() for tag in image_catalog.normalize_tags(tags or [])})
        if wanted:
            where.append(f'number IN (SELECT image FROM image_tags WHERE tag IN ({", ".join("?" * len(wanted))}) '
                         'GROUP BY image HAVING count(*)=?)')
            params += [*wanted, len(wanted)]
        return ' AND '.join(where), params, search_terms(query)

    def search(self, query='', tags=(), order='newest', limit=24, offset=0):
        """One page of visible images, newest (or oldest) first, plus how many match in total."""
        condition, params, terms = self._filters(query, tags)
        if terms:
            condition += ' AND number IN (SELECT rowid FROM images_search WHERE images_search MATCH ?)'
            params.append(terms)
        direction = 'DESC' if order == 'newest' else 'ASC'
        with self.connection() as db:
            total = db.execute(f'SELECT count(*) FROM images WHERE {condition}', params).fetchone()[0]
            rows = db.execute(f'SELECT * FROM images WHERE {condition} ORDER BY created {direction}, id {direction} '
                              'LIMIT ? OFFSET ?', [*params, limit, offset]).fetchall()
        return [{**dict(row), '_match': 'exact' if terms else None} for row in rows], total

    def search_similar(self, query='', tags=(), order='newest', limit=24, offset=0, related=()):
        """Like search(), with similar images after the exact matches: (page rows, total, similar total).

        related lists (image number, similarity) pairs, best first. Those images follow the exact
        matches in that order (marked match='similar'), skipping exact matches, hidden images and
        any outside the tag filter. With no query, only the similar images are listed.
        """
        visible, params, terms = self._filters(query, tags)
        direction = 'DESC' if order == 'newest' else 'ASC'
        with self.connection() as db:
            exact = [row[0] for row in db.execute(
                f'SELECT number FROM images WHERE {visible} AND number IN (SELECT rowid FROM images_search WHERE images_search MATCH ?) '
                f'ORDER BY created {direction}, id {direction}', [*params, terms])] if terms else []
            allowed = {row[0] for row in db.execute(f'SELECT number FROM images WHERE {visible}', params)}
            known = set(exact)
            similar = [(number, score) for number, score in related if number in allowed and number not in known]
            page = [(number, 'exact', None) for number in exact[offset:offset + limit]]
            start = max(0, offset - len(exact))
            page += [(number, 'similar', score) for number, score in similar[start:start + limit - len(page)]]
            found = {}
            for chunk in range(0, len(page), 500):
                numbers = [number for number, _, _ in page[chunk:chunk + 500]]
                found.update((row['number'], dict(row)) for row in db.execute(
                    f'SELECT * FROM images WHERE number IN ({", ".join("?" * len(numbers))})', numbers))
        rows = [{**found[number], '_match': match, '_similarity': score} for number, match, score in page if number in found]
        return rows, len(exact) + len(similar), len(similar)

    def unembedded(self, model, limit=16):
        """Visible images with no vector from this model yet, newest first."""
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                'SELECT images.number, images.id, images.file_path FROM images LEFT JOIN image_vectors '
                'ON image_vectors.image=images.number AND image_vectors.model=? '
                'WHERE images.missing=0 AND image_vectors.image IS NULL ORDER BY images.created DESC LIMIT ?', (model, limit))]

    def save_vectors(self, model, entries):
        """Store (number, id, float32 vector or None) entries; rows deleted meanwhile are skipped."""
        with self.connection(write=True) as db:
            db.executemany('INSERT OR REPLACE INTO image_vectors(image, model, vector) '
                           'SELECT ?, ?, ? WHERE EXISTS (SELECT 1 FROM images WHERE number=? AND id=?)',
                           [(number, model, None if vector is None else np.asarray(vector, '<f4').tobytes(), number, identifier)
                            for number, identifier, vector in entries])
            self._bump_vectors(db)

    @staticmethod
    def _bump_vectors(db):
        db.execute("INSERT INTO settings(key, value) VALUES ('vectors_revision', '1') "
                   "ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1")

    def vectors_revision(self):
        with self.connection() as db:
            row = db.execute("SELECT value FROM settings WHERE key='vectors_revision'").fetchone()
        return int(row['value']) if row else 0

    def vectors(self, model):
        """(image numbers, matrix of their unit vectors) for every stored vector of this model."""
        with self.connection() as db:
            rows = db.execute('SELECT image, vector FROM image_vectors WHERE model=? AND vector IS NOT NULL ORDER BY image', (model,)).fetchall()
        if not rows:
            return np.zeros(0, np.int64), np.zeros((0, 0), np.float32)
        return (np.array([row['image'] for row in rows], np.int64),
                np.stack([np.frombuffer(row['vector'], '<f4') for row in rows]))

    def vector_counts(self, model):
        """(images with a vector from this model, visible images still waiting for one)."""
        with self.connection() as db:
            done = db.execute('SELECT count(*) FROM image_vectors WHERE model=?', (model,)).fetchone()[0]
            waiting = db.execute('SELECT count(*) FROM images LEFT JOIN image_vectors ON image_vectors.image=images.number '
                                 'AND image_vectors.model=? WHERE images.missing=0 AND image_vectors.image IS NULL', (model,)).fetchone()[0]
        return done, waiting

    def set_tags(self, identifier, tags):
        tags = image_catalog.normalize_tags(tags)
        with self.connection(write=True) as db:
            row = db.execute('SELECT * FROM images WHERE id=?', (identifier,)).fetchone()
            if row is None:
                return None
            db.execute('UPDATE images SET tags=? WHERE number=?', (json.dumps(tags, ensure_ascii=False), row['number']))
            record = {key: row[key] for key in ('prompt', 'seed', 'size', 'model')}
            record.update(lora=json.loads(row['lora']) if row['lora'] else None, loras=json.loads(row['loras']) if row['loras'] else None)
            self._index(db, row['number'], record, tags)
            return {**dict(row), 'tags': json.dumps(tags, ensure_ascii=False)}

    def remove(self, identifier):
        with self.connection(write=True) as db:
            row = db.execute('SELECT number FROM images WHERE id=?', (identifier,)).fetchone()
            if row:
                for statement in ('DELETE FROM image_tags WHERE image=?', 'DELETE FROM images_search WHERE rowid=?',
                                  'DELETE FROM image_vectors WHERE image=?', 'DELETE FROM images WHERE number=?'):
                    db.execute(statement, (row['number'],))
                self._bump_vectors(db)

    def mark_missing(self, identifiers, missing=True):
        with self.connection(write=True) as db:
            db.executemany('UPDATE images SET missing=? WHERE id=?', [(int(missing), identifier) for identifier in identifiers])

    def recheck_missing(self):
        """Show hidden images again whose files are back (for example a drive that was unplugged)."""
        with self.connection() as db:
            hidden = [(row['id'], row['file_path']) for row in db.execute('SELECT id, file_path FROM images WHERE missing=1')]
        back = [identifier for identifier, path in hidden if Path(path).is_file()]
        if back:
            self.mark_missing(back, missing=False)
        return len(back)

    def setting(self, key):
        with self.connection() as db:
            row = db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
        return json.loads(row['value']) if row else None

    def set_setting(self, key, value):
        with self.connection(write=True) as db:
            db.execute('INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                       (key, json.dumps(value)))


def import_library(store, roots, *, rescan=False):
    """Register images saved before the gallery database existed; existing rows are kept.

    roots maps a model ID to (model data folder, its images folder). Each folder's default
    images and image-index/ entries (custom output locations and edited tags) are imported once;
    afterwards image-index/ is renamed to image-index.imported as a backup. rescan=True looks
    at every images folder again, for files copied in by hand.
    """
    added = 0
    for origin, (root, image_dir) in roots.items():
        root, image_dir = Path(root), Path(image_dir)
        index = root / 'image-index'
        key = f'imported:{root}'
        if not root.is_dir() or (not rescan and not index.is_dir() and store.setting(key)):
            continue
        identifiers = set()
        for folder, pattern in [(image_dir, '*.png'), (index, '*.json')]:
            if folder.is_dir():
                identifiers.update(path.stem for path in folder.glob(pattern) if image_catalog.valid_identifier(path.stem))
        for identifier in sorted(identifiers - store.known(identifiers)):
            try:
                path = image_catalog.resolve_image(root, image_dir, identifier)
            except (OSError, ValueError):
                continue
            entry = image_catalog.read_metadata(index / f'{identifier}.json')
            metadata = image_catalog.read_metadata(path.with_suffix('.json'))
            if store.register(identifier, root=root, origin=origin, path=path, metadata=metadata,
                              tags=image_catalog.read_tags(entry, metadata), replace=False):
                added += 1
        if index.is_dir():
            backup = root / 'image-index.imported'
            number = 2
            while backup.exists():
                backup = root / f'image-index.imported-{number}'
                number += 1
            index.rename(backup)
        store.set_setting(key, int(time.time()))
    return added


def main(argv=None):
    parser = argparse.ArgumentParser(description='Add saved images that are not in the gallery yet (for example PNGs copied into a model images folder).')
    parser.parse_args(argv)
    from config import MODEL_ROOTS, PROJECT_ROOT
    store = GalleryStore(gallery_database(PROJECT_ROOT))
    added = import_library(store, {model: (root, root / 'images') for model, root in MODEL_ROOTS.items()}, rescan=True)
    store.recheck_missing()
    print(f'Added {added} image(s) to the gallery: {store.path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
