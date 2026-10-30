"""Opt-in live previews: runtime flags, newest-frame files, serving and progress, without inference."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
from PIL import Image
import chroma_engine
import engine
import live_preview
import mflux_engine
from job_progress import ProgressTracker
from model_profiles import PROFILES
from tests import test_api

JOB = 'a'*32


def until(condition, timeout=10):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if condition(): return
        time.sleep(.05)
    raise AssertionError('Condition was not reached')


class PreviewFileTest(unittest.TestCase):
    def test_only_the_newest_step_survives_as_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for step in (1, 2, 3):
                path = folder/f'seed_7_step{step}of4.png'; path.write_bytes(b'step %d' % step)
                os.utime(path, (step, step))
            (folder/'seed_9_step1of4.png').write_bytes(b'other seed')
            live_preview.keep_latest(folder, 7)
            self.assertEqual((folder/'preview.png').read_bytes(), b'step 3')
            self.assertEqual(sorted(path.name for path in folder.iterdir()), ['preview.png', 'seed_9_step1of4.png'])

    def test_preview_time_belongs_to_this_attempt_of_a_live_preview_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            job = {'id': JOB, 'started': time.time(), 'request': {'live_preview': True}}
            self.assertIsNone(live_preview.preview_updated(project, job))
            folder = live_preview.preview_folder(project, JOB); folder.mkdir(parents=True)
            preview = folder/'preview.png'; preview.write_bytes(b'png')
            self.assertAlmostEqual(live_preview.preview_updated(project, job), preview.stat().st_mtime)
            self.assertIsNone(live_preview.preview_updated(project, {**job, 'request': {}}))
            old = job['started']-60; os.utime(preview, (old, old))  # Left by an earlier attempt.
            self.assertIsNone(live_preview.preview_updated(project, job))
            with self.assertRaises(ValueError): live_preview.preview_folder(project, '../escape')


class RuntimeFlagTest(unittest.TestCase):
    def test_mflux_cli_runs_through_the_preview_wrapper_only_when_asked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); commands = []
            def run(command, **kwargs):
                commands.append(command)
                Image.new('RGB', (256, 256), 'green').save(command[command.index('--output')+1])
                return subprocess.CompletedProcess(command, 0)
            settings = dict(prompt='A fox', width=256, height=256, seed=1, num_inference_steps=4)
            with patch.object(mflux_engine, 'ROOT', root), patch.object(mflux_engine, 'PROFILE', PROFILES['flux2-klein-4b']), patch.object(mflux_engine.subprocess, 'run', side_effect=run):
                mflux_engine.MfluxResult(settings).save(root/'plain.png')
                mflux_engine.MfluxResult({**settings, 'preview_dir': '/tmp/previews/x'}).save(root/'live.png')
            plain, live = commands
            self.assertEqual(plain[1:3], ['-m', PROFILES['flux2-klein-4b']['text_cli_module']])
            self.assertNotIn('--stepwise-image-output-dir', plain)
            self.assertEqual(Path(live[1]).name, 'live_preview.py')
            self.assertEqual(live[2], PROFILES['flux2-klein-4b']['text_cli_module'])
            self.assertEqual(live[live.index('--stepwise-image-output-dir')+1], '/tmp/previews/x')

    def test_chroma_writes_about_eight_vae_previews(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); commands = []
            def run(command, **kwargs):
                commands.append(command)
                Image.new('RGB', (256, 256), 'green').save(command[command.index('--output')+1])
                return subprocess.CompletedProcess(command, 0)
            profile = PROFILES['chroma1-hd']
            with patch.object(chroma_engine, 'ROOT', root), patch.object(chroma_engine, 'PROFILE', profile), patch.object(chroma_engine.subprocess, 'run', side_effect=run):
                chroma_engine.ChromaResult(root/'sd-cli', dict(prompt='A fox', seed=1, width=256, height=256, num_inference_steps=40, preview_dir=str(root/'p'))).save(root/'out.png')
                chroma_engine.ChromaResult(root/'sd-cli', dict(prompt='A fox', seed=1, width=256, height=256, num_inference_steps=40)).save(root/'plain.png')
            live, plain = commands
            self.assertEqual(live[live.index('--preview')+1], 'vae')
            self.assertEqual(live[live.index('--preview-path')+1], str(root/'p'/'preview.png'))
            self.assertEqual(live[live.index('--preview-interval')+1], '5')
            self.assertNotIn('--preview', plain)

    def test_resident_z_image_writes_previews_only_during_that_generation(self):
        registered = []
        class Registry:
            def __init__(self): self.before_loop, self.in_loop, self.after_loop, self.interrupt = [], [], [], []
            def register(self, handler): self.in_loop.append(handler); registered.append(handler)
        class Model:
            callbacks = Registry()
            def generate_image(self, **settings):
                self.seen = list(self.callbacks.in_loop); return 'image'
        class Handler:
            def __init__(self, model, output_dir, latent_creator): self.output_dir, self.latent_creator = output_dir, latent_creator
        model = Model()
        fake_creator = types.ModuleType('mflux.models.z_image.latent_creator'); fake_creator.ZImageLatentCreator = object()
        with patch.dict(sys.modules, {'mflux.models.z_image.latent_creator': fake_creator}), patch.object(live_preview, 'handler_class', return_value=Handler):
            runtime = engine.ZImageEngine(factory=lambda selection: model, clear_cache=lambda: None)
            self.assertEqual(runtime.generate_image(prompt='A fox', seed=1), 'image')
            self.assertEqual(model.seen, [])
            self.assertEqual(runtime.generate_image(prompt='A fox', seed=1, preview_dir='/tmp/p'), 'image')
        self.assertEqual(len(model.seen), 1)
        self.assertEqual((model.seen[0].output_dir, model.seen[0].latent_creator), ('/tmp/p', fake_creator.ZImageLatentCreator))
        self.assertEqual(model.callbacks.in_loop, [])  # Removed once the image is done.


class PreviewServingTest(unittest.TestCase):
    def setUp(self):
        self.fixture = test_api.APITest(); self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_jobs_pass_a_preview_folder_and_clean_it_up(self):
        client, fake = self.fixture.client, self.fixture.fake
        plain = client.post('/v1/jobs', json=self.fixture.payload).json()
        live = client.post('/v1/jobs', json={**self.fixture.payload, 'live_preview': True}).json()
        until(lambda: client.get('/v1/jobs/'+live['id']).json()['status'] == 'succeeded')
        calls = {Path(call['preview_dir']).name if 'preview_dir' in call else None for call in fake.calls}
        self.assertEqual(calls, {None, live['id']})
        self.assertFalse(live_preview.preview_folder(self.fixture.root, live['id']).exists())
        self.assertTrue(client.get('/v1/jobs/'+live['id']).json()['request']['live_preview'])
        self.assertFalse(client.get('/v1/jobs/'+plain['id']).json()['request'].get('live_preview'))
        self.assertEqual(client.post('/v1/jobs', json={**self.fixture.payload, 'live_preview': 'yes'}).status_code, 422)

    def test_running_live_jobs_serve_their_newest_preview(self):
        self.fixture.worker.stop()  # Keep the job running without inference.
        store = self.fixture.store
        job = store.enqueue({**self.fixture.payload, 'model': 'z-image-turbo', 'live_preview': True}, 'live')
        running = store.claim()
        route = f"/v1/jobs/{job['id']}/preview"
        self.assertEqual(self.fixture.client.get(route).status_code, 404)
        folder = live_preview.preview_folder(self.fixture.root, job['id']); folder.mkdir(parents=True)
        Image.new('RGB', (32, 32), 'teal').save(folder/'preview.png')
        response = self.fixture.client.get(route)
        self.assertEqual((response.status_code, response.headers['content-type'], response.headers['cache-control']), (200, 'image/png', 'no-store'))
        tracker = ProgressTracker(store, types.SimpleNamespace(job=running, stage='generating'), self.fixture.root, steps=lambda *args: (2, 4, 1.0))
        self.assertIsNotNone(tracker.poll()['preview_updated'])
        store.finish(job['id'], result={'created': 1})
        self.assertEqual(self.fixture.client.get(route).status_code, 404)


if __name__ == '__main__':
    unittest.main()
