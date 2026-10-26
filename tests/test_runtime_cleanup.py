"""Safe cleanup without querying protected unrelated process details."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from model_process import ModelProcess,ModelProcessError


class CleanupTest(unittest.TestCase):
    def test_only_owned_group_status_is_inspected_and_zombies_are_not_live(self):
        import psutil
        for status,expected in ((psutil.STATUS_ZOMBIE,False),(psutil.STATUS_RUNNING,True)):
            with patch('psutil.pids',return_value=[10,11,12]),patch('os.getpgid',side_effect=[PermissionError(),999,123]),patch('psutil.Process',return_value=Mock(status=Mock(return_value=status))) as process:
                self.assertEqual(ModelProcess.group_has_live_members(123),expected)
                process.assert_called_once_with(12)
    def test_disappearing_owned_process_is_safe_but_inaccessible_owned_status_is_live(self):
        import psutil
        for error,expected in ((psutil.NoSuchProcess(12),False),(psutil.AccessDenied(12),True)):
            with patch('psutil.pids',return_value=[12]),patch('os.getpgid',return_value=123),patch('psutil.Process',side_effect=error):
                self.assertEqual(ModelProcess.group_has_live_members(123),expected)
    def test_reaped_group_does_not_require_group_signal_permission(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=ModelProcess(folder);manager.process=Mock(pid=123);manager.active_model='z-image-turbo'
            with patch.object(manager,'group_has_live_members',return_value=False),patch('os.killpg',side_effect=PermissionError()) as kill:
                manager.close();kill.assert_not_called()
            self.assertIsNone(manager.process)
    def test_real_cleanup_failure_remains_fatal_with_persisted_cause(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=ModelProcess(folder);manager.process=Mock(pid=123);manager.active_model='z-image-turbo'
            path=Path(folder)/'state/model-process';path.mkdir(parents=True)
            with patch.object(manager,'group_has_live_members',return_value=True),patch('os.killpg',side_effect=PermissionError('Synthetic denied signal')),self.assertLogs('model_process',level='ERROR'):
                with self.assertRaises(ModelProcessError) as error:manager.close()
            self.assertTrue(error.exception.fatal);self.assertIsNotNone(manager.process)
            self.assertIn('Synthetic denied signal',(path/'server.log').read_text())
