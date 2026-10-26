"""Verify reference orientation, aspect ratio, padding, and response dimensions."""
import base64
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from PIL import Image, ImageChops
from tests import test_api
import reference_geometry


class ReferenceGeometryTest(unittest.TestCase):
    setUp = test_api.APITest.setUp
    tearDown = test_api.APITest.tearDown
    post = test_api.APITest.post

    def test_reference_shapes_override_requested_shape(self):
        for source_size, requested, expected, canvas in [
            ((1600, 900), '1024x1024', (1024, 576), (1024, 576)),
            ((900, 1600), '1024x768', (576, 1024), (576, 1024)),
            ((1600, 1200), '1024x1024', (1024, 768), (1024, 768)),
            ((403, 271), '1024x1024', (1024, 689), (1024, 704)),
            ((1000, 100), '512x512', (512, 51), (512, 256)),
        ]:
            with self.subTest(source_size=source_size):
                ref = self.root / 'reference.png'
                Image.new('RGB', source_size, 'red').save(ref)
                original = ref.read_bytes()
                response = self.post(reference_image=str(ref), size=requested)
                self.assertEqual(response.status_code, 200, response.text)
                item = response.json()['data'][0]
                self.assertEqual((item['width'], item['height']), expected)
                self.assertEqual(item['size'], f'{expected[0]}x{expected[1]}')
                self.assertEqual(item['generation_size'], f'{canvas[0]}x{canvas[1]}')
                call = self.fake.calls[-1]
                self.assertEqual((call['width'], call['height']), canvas)
                self.assertEqual(self.fake.reference_sizes[-1], canvas)
                with Image.open(item['file_path']) as saved:
                    self.assertEqual(saved.size, expected)
                meta = json.loads(Path(item['metadata_path']).read_text())
                self.assertEqual(meta['size'], item['size'])
                self.assertEqual(meta['requested_size'], requested)
                self.assertEqual(meta['reference_size'], list(source_size))
                self.assertEqual(meta['reference_image'], str(ref))
                self.assertEqual(ref.read_bytes(), original)
                self.assertFalse(Path(call['image_path']).exists())

    def test_direct_camera_file_uses_display_orientation(self):
        ref = self.root / 'camera.jpg'
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (320, 256), 'red').save(ref, exif=exif)
        response = self.post(reference_image=str(ref), size='512x512')
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()['data'][0]
        self.assertEqual((item['width'], item['height']), (410, 512))
        meta = json.loads(Path(item['metadata_path']).read_text())
        self.assertEqual(meta['reference_size'], [256, 320])

    def test_padding_is_removed_without_stretching_or_losing_content(self):
        ref = self.root / 'pattern.png'
        photo = Image.new('RGB', (403, 271), 'red')
        photo.paste('blue', (0, 0, 100, 271))
        photo.paste('yellow', (300, 0, 403, 70))
        photo.save(ref)
        self.fake.echo_reference = True
        response = self.post(reference_image=str(ref), size='512x512', response_format='b64_json')
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()['data'][0]
        with Image.open(io.BytesIO(base64.b64decode(item['b64_json']))) as saved:
            expected = photo.resize(saved.size, Image.Resampling.LANCZOS)
            self.assertIsNone(ImageChops.difference(saved, expected).getbbox())
        self.assertEqual(base64.b64decode(item['b64_json']), Path(item['file_path']).read_bytes())

    def test_prepared_reference_is_cleaned_after_failure(self):
        ref = self.root / 'reference.png'
        Image.new('RGB', (403, 271)).save(ref)
        with patch.object(self.fake, 'generate_image', side_effect=RuntimeError('test failure')):
            self.assertEqual(self.post(reference_image=str(ref)).status_code, 500)
        self.assertEqual(list((self.root / 'cache/scratch').iterdir()), [])
        self.assertTrue(ref.exists())

    def test_plain_generation_keeps_requested_dimensions(self):
        response = self.post(size='512x320')
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()['data'][0]
        self.assertEqual((item['width'], item['height']), (512, 320))
        self.assertEqual(item['generation_size'], '512x320')
        self.assertIsNone(self.fake.calls[-1]['image_path'])

    def test_openapi_explains_reference_sizing(self):
        schema = self.client.get('/openapi.json').json()['components']['schemas']
        self.assertIn('aspect ratio', schema['GenerateRequest']['properties']['size']['description'])
        for field in ['size', 'width', 'height', 'generation_size']:
            self.assertIn(field, schema['ImageResult']['required'])


if __name__ == '__main__':
    unittest.main()
