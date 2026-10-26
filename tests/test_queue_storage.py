"""Shared queue placement, isolation from model storage and global ownership."""
from pathlib import Path
import tempfile
import unittest
from queue_storage import queue_database,initialize_shared_queue
from generation_worker import QueueWorker
from storage_paths import configure_storage


class QueueStorageTest(unittest.TestCase):
    def test_shared_store_is_internal_with_external_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);project=root/'project';project.mkdir();data=root/'external'
            configure_storage(project,data)
            store=initialize_shared_queue(project)
            self.assertEqual(store.path,project.resolve()/'state/queue/jobs.sqlite3')
            self.assertTrue(store.path.is_file());self.assertFalse((data/'queue').exists())
            first=QueueWorker(store,lambda job:None);first.acquire()
            try:
                second=QueueWorker(initialize_shared_queue(project),lambda job:None)
                with self.assertRaisesRegex(RuntimeError,'live worker'):second.acquire()
            finally:first.release()
            self.assertEqual(initialize_shared_queue(project).listing()[1],0)

    def test_external_state_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);project=root/'project';project.mkdir();external=root/'external';external.mkdir()
            (project/'state').symlink_to(external,target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError,'inside the serving project'):queue_database(project)
