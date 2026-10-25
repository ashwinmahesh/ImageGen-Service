"""API integration tests with a tiny fake generator; no weights or GPU required."""
import base64
import json
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
import server


class FakeModel:
    def __init__(self):
        self.calls = []
        self.reference_sizes = []
        self.echo_reference = False

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get('image_path'):
            with Image.open(kwargs['image_path']) as reference:
                self.reference_sizes.append(reference.size)
        echo_reference = self.echo_reference
        class Result:
            def save(self, path, overwrite=False):
                assert overwrite
                if echo_reference and kwargs.get('image_path'):
                    with Image.open(kwargs['image_path']) as reference:
                        reference.save(path)
                else:
                    Image.new('RGB', (kwargs['width'], kwargs['height']), 'green').save(path)
        return Result()


class APITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.fake = FakeModel()
        self.patches = [
            patch.object(server, 'ROOT', self.root),
            patch.object(server, 'IMAGE_DIR', self.root / 'images'),
            patch.object(server, 'model', self.fake),
            patch.dict(sys.modules, {'mlx': types.ModuleType('mlx'), 'mlx.core': types.SimpleNamespace(clear_cache=lambda: None)}),
        ]
        for p in self.patches:
            p.start()
        # No context manager: skip real model startup/lifespan.
        self.client = TestClient(server.app, base_url='http://127.0.0.1')
        self.payload = {'prompt': 'A fox', 'size': '256x256', 'seed': 42,
                        'output_dir': str(self.root / 'custom' / 'nested'), 'response_format': 'url'}

    def tearDown(self):
        self.client.close()
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def post(self, **changes):
        return self.client.post('/v1/images/generations', json={**self.payload, **changes})

    def test_custom_destination_and_persisted_url(self):
        response = self.post()
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()['data'][0]
        png = Path(item['file_path'])
        self.assertEqual(png.parent, Path(self.payload['output_dir']))
        with Image.open(png) as decoded:
            self.assertEqual(decoded.size, (256, 256))
        metadata = json.loads(Path(item['metadata_path']).read_text())
        self.assertEqual(metadata['seed'], 42)
        self.assertEqual(metadata['file_path'], str(png))
        self.assertEqual(self.client.get(item['url']).content, png.read_bytes())
        # A newly created client can resolve the URL solely from the disk index.
        other = TestClient(server.app, base_url='http://127.0.0.1')
        try:
            self.assertEqual(other.get(item['url']).content, png.read_bytes())
        finally:
            other.close()

    def test_base64_still_returns_files(self):
        response = self.post(response_format='b64_json')
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()['data'][0]
        self.assertNotIn('url', item)
        self.assertEqual(base64.b64decode(item['b64_json']), Path(item['file_path']).read_bytes())
        self.assertTrue(Path(item['metadata_path']).exists())

    def test_reference_forwarded_without_modifying_source(self):
        ref = self.root / 'reference.png'
        Image.new('RGB', (256, 256), 'red').save(ref)
        original = ref.read_bytes()
        response = self.post(reference_image=str(ref), image_strength=0.7)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.fake.calls[-1]['image_path'], str(ref))
        self.assertEqual(self.fake.calls[-1]['image_strength'], 0.7)
        self.assertEqual(ref.read_bytes(), original)
        meta = json.loads(Path(response.json()['data'][0]['metadata_path']).read_text())
        self.assertEqual(meta['reference_image'], str(ref))

    def test_unreadable_reference_and_unwritable_destination(self):
        self.assertEqual(self.post(reference_image=str(self.root / 'missing.png')).status_code, 400)
        bad = self.root / 'bad.png'
        bad.write_text('not an image')
        self.assertEqual(self.post(reference_image=str(bad)).status_code, 400)
        self.assertEqual(self.post(output_dir=str(bad)).status_code, 400)
        self.assertEqual(len(self.fake.calls), 0)

    def test_invalid_inputs(self):
        for change in [{'output_dir': 'relative'}, {'reference_image': 'relative.png'},
                       {'image_strength': 0.6}, {'reference_image': '/tmp/x', 'image_strength': 1},
                       {'reference_image': '/tmp/x', 'steps': 1}, {'size': '1000x1000'}, {'size': '2112x1024'},
                       {'prompt': ' '}, {'prompt': 'x' * 4001}, {'n': 2}]:
            with self.subTest(change=list(change)):
                self.assertEqual(self.post(**change).status_code, 422)

    def test_busy_and_missing_image(self):
        server.busy.acquire()
        try:
            self.assertEqual(self.post().status_code, 429)
        finally:
            server.busy.release()
        self.assertEqual(self.client.get('/images/invalid.png').status_code, 404)
        self.assertEqual(self.client.get('/images/' + 'a' * 32 + '.png').status_code, 404)

    def test_schema_documents_paths_and_reference(self):
        schema = self.client.get('/openapi.json').json()['components']['schemas']
        self.assertIn('reference_image', schema['GenerateRequest']['properties'])
        self.assertIn('512 tokens', schema['GenerateRequest']['properties']['prompt']['description'])
        self.assertIn('file_path', schema['ImageResult']['required'])
        self.assertIn('metadata_path', schema['ImageResult']['required'])

    def test_larger_resolution_validation(self):
        for size in ['1536x1024', '1536x1536', '2048x2048']:
            self.assertEqual(server.GenerateRequest(prompt='A fox', size=size).size, size)

    def test_upload_reference_preview_and_generation(self):
        source = io.BytesIO()
        Image.new('RGB', (320, 256), 'blue').save(source, format='JPEG')
        response = self.client.post('/v1/reference-images', content=source.getvalue(), headers={'Content-Type': 'image/jpeg'})
        self.assertEqual(response.status_code, 201, response.text)
        uploaded = response.json()
        path = Path(uploaded['file_path'])
        self.assertEqual(path.parent, self.root / 'references')
        self.assertEqual((uploaded['width'], uploaded['height']), (320, 256))
        self.assertEqual(self.client.get(uploaded['url']).content, path.read_bytes())
        generated = self.post(reference_image=str(path))
        self.assertEqual(generated.status_code, 200, generated.text)
        self.assertEqual(generated.json()['data'][0]['size'], '256x205')
        self.assertEqual(self.fake.reference_sizes[-1], (256, 256))

    def test_upload_corrects_camera_orientation(self):
        source = io.BytesIO()
        photo = Image.new('RGB', (320, 256), 'red')
        exif = Image.Exif()
        exif[274] = 6  # Camera orientation: rotate 90 degrees clockwise.
        photo.save(source, format='JPEG', exif=exif)
        response = self.client.post('/v1/reference-images', content=source.getvalue())
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual((response.json()['width'], response.json()['height']), (256, 320))

    def test_upload_invalid_and_large_files(self):
        for content in [b'', b'not an image']:
            self.assertEqual(self.client.post('/v1/reference-images', content=content).status_code, 400)
        with patch.object(server, 'MAX_REFERENCE_BYTES', 4):
            self.assertEqual(self.client.post('/v1/reference-images', content=b'12345').status_code, 413)
        source = io.BytesIO()
        Image.new('RGB', (20, 20)).save(source, format='PNG')
        with patch.object(server, 'MAX_REFERENCE_PIXELS', 100):
            self.assertEqual(self.client.post('/v1/reference-images', content=source.getvalue()).status_code, 413)
        self.assertEqual(self.client.get('/references/not-an-id.png').status_code, 404)
        self.assertEqual(self.client.get('/references/' + 'a' * 32 + '.png').status_code, 404)


if __name__ == '__main__':
    unittest.main()
