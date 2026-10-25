"""Portable defaults, external symlinks and missing-volume checks; no model loading."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from storage_paths import DATA_FOLDERS, configure_storage, data_directory, require_storage, setup_storage


class StoragePathsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.project = self.base/'project'
        self.project.mkdir()
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_fresh_checkout_needs_no_external_drive(self):
        self.assertEqual(data_directory(self.project), self.project)
        self.assertEqual(configure_storage(self.project), self.project)
        self.assertEqual(configure_storage(self.project), self.project)
        for name in DATA_FOLDERS:
            self.assertTrue((self.project/name).is_dir())
            self.assertFalse((self.project/name).is_symlink())

    def test_interactive_blank_uses_project_and_external_answer_links(self):
        with patch('builtins.print'):
            result=setup_storage(self.project,interactive=True,prompt=lambda question: '')
        self.assertEqual(result,self.project)
        other=self.base/'another-project'
        other.mkdir()
        external=self.base/'External Drive'/'models'
        with patch('builtins.print'):
            result=setup_storage(other,interactive=True,prompt=lambda question: str(external))
        self.assertEqual(result,external)
        self.assertTrue((other/'z-image-turbo').is_symlink())

    def test_saved_and_noninteractive_setup_never_prompt(self):
        unexpected=lambda question: self.fail('Setup must not prompt')
        self.assertEqual(setup_storage(self.project,interactive=False,prompt=unexpected),self.project)
        self.assertEqual(setup_storage(self.project,interactive=True,prompt=unexpected),self.project)
        other=self.base/'other-project'
        other.mkdir()
        external=self.base/'external'
        with patch.dict(os.environ,{'DATA_DIR':str(external)}):
            self.assertEqual(setup_storage(other,interactive=True,prompt=unexpected),external)

    def test_external_storage_links_reuse_assets_and_persist(self):
        external=self.base/'external'
        (external/'z-image-turbo').mkdir(parents=True)
        model=external/'z-image-turbo/model.bin'
        model.write_bytes(b'existing-model')
        self.assertEqual(configure_storage(self.project,str(external)),external)
        for name in DATA_FOLDERS:
            self.assertTrue((self.project/name).is_symlink())
            self.assertEqual((self.project/name).resolve(),external/name)
        self.assertEqual(configure_storage(self.project),external)
        self.assertEqual(data_directory(self.project),external)
        self.assertEqual((self.project/'z-image-turbo/model.bin').read_bytes(),b'existing-model')
        self.assertEqual(model.read_bytes(),b'existing-model')

    def test_existing_model_symlink_is_discovered_without_reconfiguration(self):
        external=self.base/'external'
        (external/'z-image-turbo').mkdir(parents=True)
        (self.project/'z-image-turbo').symlink_to(external/'z-image-turbo',target_is_directory=True)
        self.assertEqual(data_directory(self.project),external)
        self.assertEqual(configure_storage(self.project),external)

    def test_changing_storage_never_overwrites_existing_data(self):
        (self.project/'z-image-turbo').mkdir()
        original=self.project/'z-image-turbo/model.bin'
        original.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError,'already contains data'):
            configure_storage(self.project,str(self.base/'external'))
        self.assertEqual(original.read_bytes(),b'keep')
        self.assertFalse((self.base/'external').exists())
        self.assertFalse((self.project/'storage.json').exists())

    def test_missing_volume_and_broken_link_do_not_create_fallback(self):
        path=Path('/Volumes/model-test-unmounted/models/example')
        (self.project/'z-image-turbo').symlink_to(path/'z-image-turbo',target_is_directory=True)
        with patch.object(Path,'is_mount',return_value=False),patch.object(Path,'mkdir') as mkdir:
            with self.assertRaisesRegex(RuntimeError,'not mounted'):
                configure_storage(self.project)
            mkdir.assert_not_called()
        self.assertTrue((self.project/'z-image-turbo').is_symlink())
        self.assertFalse((self.project/'storage.json').exists())

    def test_data_override_requires_absolute_path_and_rejects_bad_manifest(self):
        for value in ['', 'relative/folder']:
            with patch.dict(os.environ,{'DATA_DIR':value}):
                with self.assertRaises(ValueError):
                    data_directory(self.project)
        (self.project/'storage.json').write_text('{invalid')
        with self.assertRaisesRegex(ValueError,'Invalid storage.json'):
            data_directory(self.project)

    def test_config_separates_source_from_explicit_external_data(self):
        data=self.base/'assets'
        code='''import config, json, os
print(json.dumps({'project':str(config.PROJECT_ROOT),'data':str(config.ROOT),
'weights':str(config.MODEL_DIR),'images':str(config.IMAGE_DIR),'cache':os.environ['HF_HOME'],
'runtime':str(getattr(config,'RUNTIME_DIR',''))}))'''
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,
            env={**os.environ,'DATA_DIR':str(data)},capture_output=True,text=True,check=True)
        paths=json.loads(result.stdout)
        self.assertEqual(paths['project'],str(Path(__file__).resolve().parent))
        self.assertEqual(paths['data'],str(data/'z-image-turbo'))
        self.assertEqual(paths['weights'],str(data/'z-image-turbo/weights'))
        self.assertEqual(paths['images'],str(data/'z-image-turbo/images'))
        self.assertEqual(paths['cache'],str(data/'z-image-turbo/cache/huggingface'))
        self.assertFalse(data.exists())
        if paths['runtime']:
            self.assertFalse(Path(paths['runtime']).is_relative_to(data))


if __name__ == '__main__':
    unittest.main()
