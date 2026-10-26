"""Project-local queue state and a locked, restart-safe legacy migration."""
from contextlib import closing
import fcntl
import hashlib
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import uuid
from storage_paths import require_storage

log = logging.getLogger('image-queue')


def queue_database(project, model):
    project = Path(project).resolve()
    path = project / 'state' / 'queues' / model / 'jobs.sqlite3'
    if not path.resolve().is_relative_to(project):
        raise RuntimeError('Queue state must stay inside the serving project; remove the external state symlink')
    return path


def _check(db):
    if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
        raise RuntimeError('Queue integrity check failed; original files were left unchanged')
    if db.execute('PRAGMA user_version').fetchone()[0] not in (1,2,3,4):
        raise RuntimeError('Unsupported legacy queue database version; original files were left unchanged')


def _digest(db):
    digest = hashlib.sha256()
    for line in db.iterdump():
        digest.update(line.encode())
        digest.update(b'\n')
    return digest.digest()


def _sync(path):
    with Path(path).open('rb') as stream:
        os.fsync(stream.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def initialize_queue(store, legacy_path):
    """Call while holding the destination worker lease, before recovery/inference.

    The old worker lease prevents copying a live queue. SQLite's backup API also
    captures committed WAL content. Keep a verified project-local backup before
    retiring the old database; neither an absent SSD nor a conflict creates a
    second empty queue. Identical copies after an interrupted migration converge.
    """
    legacy = Path(legacy_path)
    require_storage(legacy)
    if legacy.resolve() == store.path.resolve():
        raise RuntimeError('Legacy and project-local queue paths must be different')
    if legacy.is_file():
        with (legacy.parent / 'worker.lock').open('a+') as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('The legacy queue still has a worker; stop its server and let native inference finish before migrating') from None
            _migrate(store.path, legacy)
    store.initialize()


def _migrate(destination, legacy):
    # rw forbids SQLite from silently creating an empty source if it disappeared.
    with closing(sqlite3.connect(legacy.as_uri() + '?mode=rw', uri=True)) as source:
        _check(source)
        if destination.exists():
            with closing(sqlite3.connect(destination.as_uri() + '?mode=ro', uri=True)) as current:
                _check(current)
                if _digest(source) != _digest(current):
                    raise RuntimeError('Legacy and project-local queues differ; refusing to merge or overwrite either database')
        descriptor, name = tempfile.mkstemp(prefix='.queue-migration-', suffix='.sqlite3', dir=destination.parent)
        os.close(descriptor)
        temporary = Path(name)
        backup = destination.parent / f'legacy-backup-{uuid.uuid4().hex}.sqlite3'
        try:
            with closing(sqlite3.connect(temporary)) as snapshot:
                source.backup(snapshot)
                _check(snapshot)
                if _digest(source) != _digest(snapshot):
                    raise RuntimeError('Legacy queue changed during migration; original files were left unchanged')
            _sync(temporary)
            temporary.replace(backup)
            _sync_directory(destination.parent)
            if not destination.exists():
                shutil.copyfile(backup, temporary)
                temporary.chmod(0o600)
                _sync(temporary)
                temporary.replace(destination)
                _sync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
    # All connections and both worker leases are controlled here. The verified
    # standalone backup includes WAL rows, so obsolete sidecars are safe to retire.
    for suffix in ['-wal', '-shm', '-journal']:
        legacy.with_name(legacy.name + suffix).unlink(missing_ok=True)
    legacy.unlink()
    _sync_directory(legacy.parent)
    log.info('Moved legacy queue to %s; retained backup %s', destination, backup)
