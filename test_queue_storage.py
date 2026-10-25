"""Local SQLite placement and migration preserve real queue state without a GPU."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from job_queue import JobStore, QueueWorker, fingerprint
from queue_storage import initialize_queue, queue_database
from storage_paths import configure_storage


class QueueStorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.project = self.root / 'project'
        self.project.mkdir()
        self.data = self.root / 'external'
        configure_storage(self.project, self.data)
        self.legacy = JobStore(self.data / 'z-image-turbo' / 'queue' / 'jobs.sqlite3')
        self.store = JobStore(queue_database(self.project, 'z-image-turbo'))
        self.worker = QueueWorker(self.store, lambda job: None)

    def tearDown(self):
        self.worker.release()
        self.temp.cleanup()

    def add(self, name, immediate=False):
        payload = dict(prompt=name, seed=123, generate_immediately=immediate)
        return self.legacy.enqueue(payload, fingerprint(payload), key=name)

    def migrate(self):
        self.worker.acquire(prepare=lambda: initialize_queue(self.store, self.legacy.path))

    def test_external_weights_and_local_databases_are_independent(self):
        self.assertTrue((self.project / 'z-image-turbo').is_symlink())
        self.assertFalse(self.legacy.path.parent.exists())
        self.migrate()
        self.assertTrue(self.store.path.is_relative_to(self.project / 'state'))
        self.assertTrue(self.store.path.is_file())
        self.assertFalse(self.legacy.path.exists())
        self.assertEqual(self.store.summary()['database_path'], str(self.store.path))
        with patch('queue_storage.require_storage', side_effect=RuntimeError('SSD absent')):
            # Once initialized, queue listing/clear need no model-store mount.
            self.assertEqual(self.store.listing()[1], 0)
            self.assertEqual(self.store.clear(), 0)

    def test_migration_preserves_jobs_ids_results_idempotency_pause_and_order(self):
        self.legacy.initialize()
        completed = self.add('complete')
        self.legacy.claim()
        self.legacy.finish(completed['id'], result={'file_path': '/external/images/result.png'})
        ordinary = self.add('ordinary')
        immediate = self.add('next', True)
        self.legacy.pause(True)
        before = self.legacy.listing(limit=100)
        self.migrate()
        self.assertEqual(self.store.listing(limit=100), before)
        self.assertTrue(self.store.summary()['paused'])
        self.assertEqual(self.store.existing('ordinary', fingerprint(ordinary['request']))['id'], ordinary['id'])
        self.assertEqual(self.store.claim()['id'], immediate['id'])
        self.assertFalse(self.legacy.path.exists())
        backups = list(self.store.path.parent.glob('legacy-backup-*.sqlite3'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(JobStore(backups[0]).listing(limit=100), before)
        self.worker.release()
        self.migrate()
        self.assertEqual(len(list(self.store.path.parent.glob('legacy-backup-*.sqlite3'))), 1)
        self.assertEqual(self.store.get(immediate['id'])['status'], 'queued')

    def test_live_legacy_worker_blocks_migration_and_releases_new_lease(self):
        self.legacy.initialize()
        original = self.add('keep')
        old_worker = QueueWorker(self.legacy, lambda job: None)
        old_worker.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'legacy queue still has a worker'):
                self.migrate()
            self.assertIsNone(self.worker.lease)
            self.assertFalse(self.store.path.exists())
            self.assertEqual(self.legacy.get(original['id'])['status'], 'queued')
        finally:
            old_worker.release()
        self.migrate()

    def test_conflicting_queues_are_never_overwritten_or_merged(self):
        self.legacy.initialize()
        old = self.add('old')
        self.store.initialize()
        new = self.store.enqueue(dict(prompt='new', generate_immediately=False), 'new')
        with self.assertRaisesRegex(RuntimeError, 'queues differ'):
            self.migrate()
        self.assertEqual(self.legacy.get(old['id'])['request']['prompt'], 'old')
        self.assertEqual(self.store.get(new['id'])['request']['prompt'], 'new')

    def test_interrupted_migration_with_identical_copies_can_finish(self):
        self.legacy.initialize()
        original = self.add('same')
        self.store.path.parent.mkdir(parents=True)
        with closing(sqlite3.connect(self.legacy.path)) as old, closing(sqlite3.connect(self.store.path)) as new:
            old.backup(new)
        self.migrate()
        self.assertFalse(self.legacy.path.exists())
        self.assertEqual(self.store.get(original['id'])['request'], original['request'])

    def test_committed_wal_records_are_included_in_backup(self):
        self.legacy.initialize()
        with closing(sqlite3.connect(self.legacy.path)) as connection:
            connection.execute('PRAGMA journal_mode=WAL').fetchall()
            connection.execute('SELECT COUNT(*) FROM jobs').fetchone()
            original = self.add('wal-row')
            self.assertTrue(self.legacy.path.with_name('jobs.sqlite3-wal').exists())
            self.migrate()
        self.assertEqual(self.store.get(original['id'])['request']['prompt'], 'wal-row')
        self.assertFalse(self.legacy.path.exists())
        self.assertFalse(self.legacy.path.with_name('jobs.sqlite3-wal').exists())

    def test_failed_copy_keeps_original_and_does_not_publish_empty_database(self):
        self.legacy.initialize()
        original = self.add('keep')
        with patch('queue_storage._sync', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.migrate()
        self.assertIsNone(self.worker.lease)
        self.assertFalse(self.store.path.exists())
        self.assertEqual(self.legacy.get(original['id'])['status'], 'queued')
        self.migrate()

    def test_missing_legacy_volume_does_not_create_a_fallback_database(self):
        with patch('queue_storage.require_storage', side_effect=RuntimeError('not mounted')):
            with self.assertRaisesRegex(RuntimeError, 'not mounted'):
                self.migrate()
        self.assertFalse(self.store.path.exists())

    def test_external_state_symlink_is_rejected(self):
        (self.project / 'state').symlink_to(self.data, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'inside the serving project'):
            queue_database(self.project, 'z-image-turbo')


if __name__ == '__main__':
    unittest.main()
