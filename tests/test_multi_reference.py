"""Several reference photos for FLUX.2 Klein and Qwen-Image: validation, CLI arguments and job inputs, without inference."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError
import mflux_engine
import resident_mflux
import server
from job_queue import JobStore, QueueConflict, fingerprint
from model_profiles import PROFILES
from photo_workflows import MULTI_REFERENCE_SCOPE, edit_prompt, edit_prompt_fits, MAX_PROMPT_CHARACTERS
from tests import test_api


def request(model, **fields):
    return server.GenerateRequest(model=model, prompt='Put the jacket from image 2 on the person in image 1.', **fields)


class MultiReferenceValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.photos = [self.root / f'photo-{number}.png' for number in range(1, 6)]
        for path in self.photos: Image.new('RGB', (64, 48), 'teal').save(path)
        self.extras = [str(path) for path in self.photos[1:]]

    def tearDown(self):
        self.temp.cleanup()

    def test_klein_and_qwen_take_up_to_three_more_references(self):
        for model in ['flux2-klein-4b', 'flux2-klein-9b', 'qwen-image-2.1-turbo']:
            with self.subTest(model=model):
                accepted = request(model, reference_image=str(self.photos[0]), additional_reference_images=self.extras[:3])
                self.assertEqual(server.normalized_request(accepted)['additional_reference_images'], self.extras[:3])
                self.assertEqual(server.capabilities(model, ready=[model])['max_reference_images'], 4)
                with self.assertRaises(ValidationError):
                    request(model, reference_image=str(self.photos[0]), additional_reference_images=self.extras)

    def test_requests_without_more_references_keep_their_fingerprint(self):
        plain = request('qwen-image-2.1-turbo', reference_image=str(self.photos[0]))
        self.assertNotIn('additional_reference_images', server.normalized_request(plain))

    def test_other_models_missing_image_1_and_repeated_files_are_refused(self):
        for model in ['z-image-turbo', 'chroma1-hd', 'chroma1-flash']:
            with self.subTest(model=model), self.assertRaisesRegex(ValidationError, 'FLUX.2 Klein or Qwen'):
                request(model, reference_image=str(self.photos[0]), additional_reference_images=self.extras[:1])
            self.assertEqual(server.capabilities(model, ready=[model])['max_reference_images'], 1)
        cases = [({'additional_reference_images': self.extras[:1]}, 'image 1'),
                 ({'reference_image': str(self.photos[0]), 'additional_reference_images': [self.extras[0], self.extras[0]]}, 'different file'),
                 ({'reference_image': str(self.photos[0]), 'additional_reference_images': [str(self.photos[0])]}, 'cannot also be'),
                 ({'reference_image': str(self.photos[0]), 'additional_reference_images': []}, 'at least 1'),
                 ({'reference_image': str(self.photos[0]), 'additional_reference_images': ['relative.png']}, 'absolute')]
        for fields, message in cases:
            with self.subTest(fields=fields), self.assertRaisesRegex(ValidationError, message):
                request('qwen-image-2.1-turbo', **fields)

    def test_edit_wording_names_image_1_only_when_there_are_several(self):
        self.assertNotIn(MULTI_REFERENCE_SCOPE, edit_prompt('Make the jacket green.'))
        several = edit_prompt('Make the jacket match image 2.', None, '{"bbox_2d": [1, 2, 3, 4]}', references=3)
        self.assertIn(MULTI_REFERENCE_SCOPE, several); self.assertIn('0-1000 scale of image 1', several)
        instruction = 'x' * (MAX_PROMPT_CHARACTERS - len(edit_prompt('')) )
        self.assertTrue(edit_prompt_fits(instruction)); self.assertFalse(edit_prompt_fits(instruction, references=2))


class MultiReferenceCommandTest(unittest.TestCase):
    def command(self, model, extras):
        captured = []
        def run(command, **kwargs):
            captured.append(command); Image.new('RGB', (256, 256), 'green').save(command[command.index('--output') + 1])
            return subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as temporary, patch.object(mflux_engine, 'ROOT', Path(temporary)), \
                patch.object(mflux_engine, 'PROFILE', PROFILES[model]), patch.object(mflux_engine.subprocess, 'run', side_effect=run):
            settings = dict(prompt='Combine', width=256, height=256, seed=1, num_inference_steps=4, image_path='/tmp/one.png',
                            **({'additional_image_paths': extras} if extras else {}))
            mflux_engine.MfluxResult(settings).save(Path(temporary) / 'out.png')
        return captured[0]

    @staticmethod
    def image_paths(command):
        values = command[command.index('--image-paths') + 1:]
        return values[:next((index for index, value in enumerate(values) if value.startswith('--')), len(values))]

    def test_every_reference_is_passed_in_order_after_image_1(self):
        for model in ['flux2-klein-4b', 'qwen-image-2.1-turbo']:
            with self.subTest(model=model):
                self.assertEqual(self.image_paths(self.command(model, ['/tmp/two.png', '/tmp/three.png'])), ['/tmp/one.png', '/tmp/two.png', '/tmp/three.png'])
                self.assertEqual(self.image_paths(self.command(model, None)), ['/tmp/one.png'])

    @unittest.skipUnless(importlib.util.find_spec('mflux'), 'mflux is not installed')
    def test_pinned_clis_parse_several_image_paths(self):
        for model in ['flux2-klein-4b', 'qwen-image-2.1-turbo']:
            with self.subTest(model=model):
                command = self.command(model, ['/tmp/two.png', '/tmp/three.png', '/tmp/four.png'])
                args = resident_mflux.parse(command[2], command[3:])
                self.assertEqual([str(path) for path in args.image_paths], ['/tmp/one.png', '/tmp/two.png', '/tmp/three.png', '/tmp/four.png'])


class MultiReferenceJobInputTest(unittest.TestCase):
    def setUp(self):
        self.fixture = test_api.APITest(); self.fixture.setUp(); self.fixture.store.pause(True)
        self.client = self.fixture.client; self.folder = self.fixture.root / 'elsewhere'; self.folder.mkdir()

    def tearDown(self):
        self.fixture.tearDown()

    def test_each_additional_reference_is_served_from_the_job_only(self):
        first = self.folder / 'person.png'; Image.new('RGB', (40, 30), 'teal').save(first)
        second = self.folder / 'jacket.jpg'; Image.new('RGB', (30, 30), 'navy').save(second)
        request = {'prompt': 'Combine', 'model': 'qwen-image-2.1-turbo', 'reference_image': str(first), 'additional_reference_images': [str(second)]}
        job = self.fixture.store.enqueue(request, fingerprint(request))
        response = self.client.get(f"/v1/jobs/{job['id']}/inputs/additional_reference_images/0")
        self.assertEqual((response.status_code, response.headers['content-type'], response.content), (200, 'image/jpeg', second.read_bytes()))
        self.assertEqual(self.client.get(f"/v1/jobs/{job['id']}/inputs/additional_reference_images/1").status_code, 404)
        self.assertEqual(self.client.get(f"/v1/jobs/{job['id']}/inputs/additional_reference_images/3").status_code, 422)
        with self.assertRaises(QueueConflict):
            with self.fixture.store.guard_artifact_deletion(second): pass


if __name__ == '__main__':
    unittest.main()
