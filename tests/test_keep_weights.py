"""Keep weights loaded: shared setting, keep/release decisions and in-process reuse, without inference."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
import mflux_engine
import queue_api
import resident_mflux
import server
from job_queue import JobStore, KEEP_WEIGHTS_SETTING
from model_profiles import PROFILES
from tests.model_fixtures import setup_model


class SettingsAPITest(unittest.TestCase):
    def test_both_services_share_one_strict_setting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); setup_model(root, 'z-image-turbo')
            queue = TestClient(queue_api.create_app(root))
            store = JobStore(root/'state'/'queue'/'jobs.sqlite3')
            generator = TestClient(server.app, base_url='http://127.0.0.1')
            try:
                self.assertEqual(queue.get('/v1/settings').json(), {'keep_weights_loaded': False})
                with patch.object(server, 'job_store', store):
                    self.assertEqual(generator.get('/v1/settings').json(), {'keep_weights_loaded': False})
                    self.assertEqual(generator.patch('/v1/settings', json={'keep_weights_loaded': True}).json(), {'keep_weights_loaded': True})
                    for bad in ({'keep_weights_loaded': 'yes'}, {'keep_weights_loaded': True, 'other': 1}, {}):
                        self.assertEqual(generator.patch('/v1/settings', json=bad).status_code, 422, bad)
                self.assertEqual(queue.get('/v1/settings').json(), {'keep_weights_loaded': True})
                self.assertEqual(queue.patch('/v1/settings', json={'keep_weights_loaded': False}).json(), {'keep_weights_loaded': False})
                self.assertFalse(store.setting(KEEP_WEIGHTS_SETTING))
            finally:
                queue.close(); generator.close()


class Store:
    def __init__(self, enabled, next_model): self.enabled, self.next_model = enabled, next_model
    def setting(self, key, default=None): return self.enabled if key == KEEP_WEIGHTS_SETTING else default
    def summary(self): return {'next_model': self.next_model}


class DecisionTest(unittest.TestCase):
    def decide(self, enabled, next_model, **job):
        job = {'request': {'n': 1}, **job}
        with patch.object(server, 'job_store', Store(enabled, next_model)), patch.object(server, 'API_MODEL', 'qwen-image-2.1-turbo'):
            return server.keep_weights_loaded(job)

    def test_weights_stay_only_while_the_next_image_uses_the_same_model(self):
        self.assertFalse(self.decide(False, 'qwen-image-2.1-turbo'))
        self.assertTrue(self.decide(True, 'qwen-image-2.1-turbo'))
        self.assertFalse(self.decide(True, 'z-image-turbo'))
        self.assertFalse(self.decide(True, None))
        # Later options of the same batch reuse the weights whatever is queued next.
        self.assertTrue(self.decide(True, None, _candidate_index=1, request={'n': 3}))
        self.assertFalse(self.decide(True, None, _candidate_index=3, request={'n': 3}))


class ResidentEngineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.built = []; self.generated = []
        def build(module, args):
            self.built.append(module); return types.SimpleNamespace(callbacks=None), 'creator'
        def generate(module, model, args):
            self.generated.append(model)
            class Image_:
                def save(_, path, export_json_metadata, overwrite): Image.new('RGB', (256, 256), 'green').save(path)
            return Image_()
        fake_mlx = types.ModuleType('mlx'); fake_mlx.core = types.SimpleNamespace(clear_cache=lambda: None)
        self.patches = [patch.object(mflux_engine, 'ROOT', self.root), patch.object(mflux_engine, 'PROFILE', PROFILES['qwen-image-2.1-turbo']),
                        patch.object(resident_mflux, 'parse', side_effect=lambda module, arguments: types.SimpleNamespace(arguments=arguments)),
                        patch.object(resident_mflux, 'key', side_effect=lambda module, args: (module, 'qwen')),
                        patch.object(resident_mflux, 'build', side_effect=build), patch.object(resident_mflux, 'generate', side_effect=generate),
                        patch.object(mflux_engine.subprocess, 'run', side_effect=self.cli),
                        patch.dict(sys.modules, {'mlx': fake_mlx, 'mlx.core': fake_mlx.core})]
        for item in self.patches: item.start()
        self.engine = mflux_engine.MfluxEngine.__new__(mflux_engine.MfluxEngine); self.engine.active_log = None; self.engine.resident = None
        self.cli_runs = 0

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        self.temp.cleanup()

    def cli(self, command, **kwargs):
        self.cli_runs += 1
        Image.new('RGB', (256, 256), 'green').save(command[command.index('--output')+1])
        return subprocess.CompletedProcess(command, 0)

    def render(self, name, **settings):
        base = dict(prompt='Edit', width=256, height=256, seed=1, num_inference_steps=8, job_id=name)
        self.engine.generate_image(**base, **settings).save(self.root/(name+'.png'))
        return (self.root/'logs'/(name+'.log')).read_text()

    def test_back_to_back_images_load_once_then_release(self):
        self.assertIn('Loaded weights in', self.render('first', keep_loaded=True))
        self.assertIn('Reusing weights kept in memory', self.render('second', keep_loaded=True))
        self.assertIsNotNone(self.engine.resident)
        self.assertIn('Reusing weights kept in memory', self.render('last', keep_loaded=False))  # Loaded weights are still used once.
        self.assertIsNone(self.engine.resident)
        self.assertEqual((len(self.built), len(self.generated), self.cli_runs), (1, 3, 0))
        self.render('separate')  # Setting off and nothing loaded: the usual per-image CLI.
        self.assertEqual((len(self.built), self.cli_runs), (1, 1))

    def test_errors_release_the_kept_weights(self):
        self.render('first', keep_loaded=True)
        with patch.object(resident_mflux, 'generate', side_effect=RuntimeError('out of memory')):
            with self.assertRaisesRegex(RuntimeError, 'out of memory'): self.render('broken', keep_loaded=True)
        self.assertIsNone(self.engine.resident)


@unittest.skipUnless(importlib.util.find_spec('mflux'), 'mflux is not installed')
class PinnedCLITest(unittest.TestCase):
    def test_engine_arguments_parse_with_each_pinned_cli(self):
        """The in-process runner reuses each CLI's parser; the arguments the engine builds must still parse."""
        for model, reference in [('qwen-image-2.1-turbo', True), ('flux2-klein-4b', True), ('flux2-klein-4b', False)]:
            profile = PROFILES[model]; captured = {}
            def run(command, **kwargs):
                captured['command'] = command
                Image.new('RGB', (256, 256), 'green').save(command[command.index('--output')+1])
                return subprocess.CompletedProcess(command, 0)
            with tempfile.TemporaryDirectory() as temporary, patch.object(mflux_engine, 'ROOT', Path(temporary)), \
                    patch.object(mflux_engine, 'PROFILE', profile), patch.object(mflux_engine.subprocess, 'run', side_effect=run):
                photo = Path(temporary)/'photo.png'; Image.new('RGB', (256, 256)).save(photo)
                settings = dict(prompt='Edit', width=256, height=256, seed=1, num_inference_steps=4, **({'image_path': str(photo)} if reference else {}))
                mflux_engine.MfluxResult(settings).save(Path(temporary)/'out.png')
                command = captured['command']; module = command[2]
                self.assertIn(module, resident_mflux.SUPPORTED)
                args = resident_mflux.parse(module, command[3:])
                self.assertEqual((args.steps, args.seed), (4, [1]))
                resident_mflux.key(module, args)
                cli = sys.modules[module]
                names = ('QwenImage21Edit', 'QwenImage21LatentCreator', 'FAMILY_MODELS') if module == resident_mflux.QWEN_EDIT else ('Flux2LatentCreator', 'FAMILY_MODELS', 'DEFAULT_MODEL', 'Flux2KleinEdit' if module == resident_mflux.FLUX2_EDIT else 'Flux2Klein')
                for name in names: self.assertTrue(hasattr(cli, name), (module, name))


if __name__ == '__main__':
    unittest.main()
