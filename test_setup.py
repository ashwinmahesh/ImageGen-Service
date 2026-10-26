"""Check fresh-checkout setup without installing packages or downloading weights."""
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import config
import lora_catalog

ROOT = Path(__file__).resolve().parent


class SetupTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_make_setup_creates_environment_then_downloads_and_can_repeat(self):
        shutil.copyfile(ROOT / 'Makefile', self.root / 'Makefile')
        for name in ['setup.py','model_profiles.py']:
            shutil.copyfile(ROOT/name,self.root/name)
        log = self.root / 'calls.jsonl'
        fake = self.root / 'bootstrap'
        fake.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys, runpy
with open(os.environ['SETUP_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1:2] == ['setup.py']:
    runpy.run_path('setup.py',run_name='__main__')
if sys.argv[1:3] == ['-m', 'venv']:
    python = pathlib.Path(sys.argv[3]) / 'bin/python'
    python.parent.mkdir(parents=True)
    python.write_text(pathlib.Path(__file__).read_text())
    python.chmod(0o755)
''')
        fake.chmod(0o755)
        command = ['make', 'setup', f'VENV={self.root / "env"}', f'BOOTSTRAP_PYTHON={fake}']
        env = {**os.environ, 'SETUP_TEST_LOG': str(log)}
        subprocess.run(command, cwd=self.root, env=env, capture_output=True, check=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(calls[0], ['setup.py'])
        calls = [call for call in calls if call not in [['storage_paths.py'],['setup.py']]]
        self.assertEqual(calls[0][:2], ['-m', 'venv'])
        self.assertEqual(calls[1][:3], ['-m', 'pip', 'install'])
        self.assertEqual(calls[2:],[['download.py'],['download_enhancer.py']])
        log.write_text('')
        subprocess.run(command + ['WITH_REALISM=1'], cwd=self.root, env=env, capture_output=True, check=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(calls[0], ['setup.py'])
        calls = [call for call in calls if call not in [['storage_paths.py'],['setup.py']]]
        self.assertEqual(calls[0][:3], ['-m', 'pip', 'install'])
        self.assertEqual(calls[1:], [['download.py'], ['download_enhancer.py'], ['download_loras.py', '--id', 'realism']])

    def test_model_downloader_uses_pinned_hugging_face_snapshot(self):
        weights = self.root / 'weights'
        snapshot = Mock()
        with patch.object(config, 'MODEL_DIR', weights), patch('huggingface_hub.snapshot_download', snapshot):
            runpy.run_path(str(ROOT / 'download.py'), run_name='__main__')
        arguments = snapshot.call_args.kwargs
        self.assertEqual(arguments['repo_id'], config.MODEL_ID)
        self.assertEqual(arguments['revision'], config.MODEL_REVISION)
        self.assertEqual(arguments['local_dir'], str(weights))
        self.assertIn('transformer/*', arguments['allow_patterns'])
        self.assertIn('text_encoder/*', arguments['allow_patterns'])
        record = json.loads((weights / 'download.json').read_text())
        self.assertEqual(record['revision'], config.MODEL_REVISION)

    def test_fresh_checkout_catalog_and_local_override(self):
        entry = {'id': 'test-style', 'name': 'Source style', 'filename': 'style.safetensors',
                 'models': [config.API_MODEL], 'default_strength': .8, 'sha256': 'abc'}
        (self.root / 'lora-catalog.json').write_text(json.dumps([entry]))
        adapters = self.root / 'loras'
        with patch.object(lora_catalog, 'ROOT', self.root), patch.object(lora_catalog, 'LORA_DIR', adapters):
            listed = lora_catalog.list_loras()
            self.assertEqual(listed[0]['name'], 'Source style')
            self.assertFalse(listed[0]['available'])
            adapters.mkdir()
            (adapters / 'catalog.json').write_text(json.dumps([{**entry, 'name': 'Local style'}]))
            self.assertEqual(lora_catalog.catalog()[0]['name'], 'Local style')

    def test_git_ignores_images_assets_and_runtime_data(self):
        shutil.copyfile(ROOT / '.gitignore', self.root / '.gitignore')
        subprocess.run(['git', 'init', '-q', str(self.root)], capture_output=True, check=True)
        ignored = ['images/output.png', 'references/source.jpg', 'image-index/id.json',
                   'weights/transformer/weights.safetensors', 'loras/catalog.json',
                   'cache/private.json', 'storage.json', 'weights', 'images', 'cache', 'ui/future.PNG', 'nested/photo.JPEG', 'photo.heic',
                   'nested/vector.svg', 'ui/._index.html', '__pycache__/server.pyc', '.env']
        result = subprocess.run(['git', 'check-ignore', '--stdin', '-z'], cwd=self.root,
                                input='\0'.join(ignored)+'\0', capture_output=True, text=True, check=True)
        self.assertEqual(set(result.stdout.rstrip('\0').split('\0')), set(ignored))
        tracked = ['README.md', 'Makefile', 'server.py', 'ui/index.html', 'lora-catalog.json']
        result = subprocess.run(['git', 'check-ignore', '--stdin'], cwd=self.root,
                                input='\n'.join(tracked)+'\n', capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
