"""Saved-image browsing tests; no GPU or model weights required."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import image_catalog
import server


class GalleryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.images = self.root / 'images'
        self.images.mkdir()
        self.patches = [patch.object(server, 'PROJECT_ROOT', self.root), patch.object(server, 'ROOT', self.root), patch.object(server, 'IMAGE_DIR', self.images)]
        for item in self.patches:
            item.start()
        self.client = TestClient(server.app, base_url='http://127.0.0.1')

    def tearDown(self):
        self.client.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def save(self, identifier, metadata=None, folder=None, index=False, size=(1024, 640)):
        folder = folder or self.images
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{identifier}.png'
        Image.new('RGB', size, 'forestgreen').save(path)
        if metadata is not None:
            path.with_suffix('.json').write_text(json.dumps(metadata))
        if index:
            records = self.root / 'image-index'
            records.mkdir(exist_ok=True)
            (records / f'{identifier}.json').write_text(json.dumps({'file_path': str(path)}))
        return path

    def test_discovery_custom_folders_deduplication_and_exclusions(self):
        a, b, c = 'a' * 32, 'b' * 32, 'c' * 32
        self.save(a, {'prompt': 'Misty castle', 'created': 100}, index=True)
        self.save(b, {'prompt': 'Quiet forest', 'created': 200}, folder=self.root / 'custom', index=True)
        self.save(c, folder=self.root / 'references')
        self.save('not-a-generation')
        (self.images / ('d' * 32 + '.png')).touch()
        (self.root / 'image-index' / ('e' * 32 + '.json')).write_text('{broken')
        (self.root / 'image-index' / ('f' * 32 + '.json')).write_text(json.dumps({'file_path': '/missing/' + 'f' * 32 + '.png'}))
        response = self.client.get('/v1/images')
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data['total'], 2)
        self.assertEqual([item['id'] for item in data['data']], [b, a])
        self.assertNotIn('metadata', data['data'][0])
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(self.client.get(data['data'][0]['url']).content, (self.root / 'custom' / (b + '.png')).read_bytes())

    def test_search_sort_and_pagination(self):
        self.save('a' * 32, {'prompt': 'Misty CASTLE', 'seed': 0, 'created': 10, 'size': '1024x640'})
        self.save('b' * 32, {'prompt': 'Quiet forest', 'seed': 42, 'created': 20,
                              'lora': {'id': 'test-style', 'name': 'Painterly', 'strength': 0.8}}, size=(512, 512))
        first = self.client.get('/v1/images', params={'sort': 'oldest', 'limit': 1}).json()
        self.assertEqual(first['data'][0]['id'], 'a' * 32)
        self.assertTrue(first['has_more'])
        self.assertEqual(first['next_offset'], 1)
        second = self.client.get('/v1/images?sort=oldest&limit=1&offset=1').json()
        self.assertEqual(second['data'][0]['id'], 'b' * 32)
        self.assertFalse(second['has_more'])
        for query, expected in [('castle', 'a'), ('  CASTLE  ', 'a'), ('Painterly', 'b'), ('42', 'b')]:
            with self.subTest(query=query):
                result = self.client.get('/v1/images', params={'q': query}).json()
                self.assertEqual(result['total'], 1)
                self.assertEqual(result['data'][0]['id'], expected * 32)
        # Seed zero must remain searchable even though it is falsy in Python.
        self.assertEqual(self.client.get('/v1/images?q=0').json()['total'], 1)
        for query in ['limit=0', 'limit=101', 'offset=-1', 'sort=invalid']:
            self.assertEqual(self.client.get('/v1/images?' + query).status_code, 422)

    def test_tags_persist_without_changing_originals(self):
        identifier = 'a' * 32
        path = self.save(identifier, {'prompt': 'A castle', 'seed': 42}, folder=self.root / 'custom', index=True)
        png, sidecar = path.read_bytes(), path.with_suffix('.json').read_bytes()
        response = self.client.patch('/v1/images/' + identifier + '/tags', json={'tags': ['  Dark   Fantasy ', 'CASTLE', 'castle', 'café']})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['tags'], ['Dark Fantasy', 'CASTLE', 'café'])
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(path.read_bytes(), png)
        self.assertEqual(path.with_suffix('.json').read_bytes(), sidecar)
        other = TestClient(server.app, base_url='http://127.0.0.1')
        try:
            self.assertEqual(other.get('/v1/images/' + identifier).json()['tags'], response.json()['tags'])
            self.assertEqual(other.get('/v1/images?q=fantasy').json()['total'], 1)
            self.assertEqual(other.get('/v1/images?tag=dark%20fantasy&tag=castle').json()['total'], 1)
            self.assertEqual(other.get('/v1/images?tag=cast').json()['total'], 0)
            self.assertEqual(other.get('/v1/images?q=castle&tag=missing').json()['total'], 0)
        finally:
            other.close()
        self.assertEqual(self.client.patch('/v1/images/' + identifier + '/tags', json={'tags': []}).json()['tags'], [])
        self.assertEqual(self.client.get('/v1/images?q=fantasy').json()['total'], 0)

    def test_legacy_tags_and_invalid_edits(self):
        identifier = 'a' * 32
        path = self.save(identifier)
        path.with_suffix('.json').write_text('{unreadable')
        self.assertEqual(self.client.get('/v1/images/' + identifier).json()['tags'], [])
        response = self.client.patch('/v1/images/' + identifier + '/tags', json={'tags': ['Legacy']})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['metadata_available'])
        self.assertEqual(path.with_suffix('.json').read_text(), '{unreadable')
        self.assertEqual(self.client.get('/v1/images?tag=legacy').json()['total'], 1)
        for values in [None, 'castle', [1], [''], ['x' * 65], ['a,b'], ['line\nbreak'], ['x'] * 33]:
            with self.subTest(values=values):
                response = self.client.patch('/v1/images/' + identifier + '/tags', json={'tags': values})
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.client.patch('/v1/images/' + identifier + '/tags', json={'tags': [], 'prompt': 'changed'}).status_code, 422)
        self.assertEqual(self.client.patch('/v1/images/' + 'b' * 32 + '/tags', json={'tags': ['test']}).status_code, 404)
        self.assertEqual(self.client.get('/v1/images', params={'tag': 'x' * 65}).status_code, 422)
        with patch.object(image_catalog, 'write_index', side_effect=PermissionError('read only')):
            self.assertEqual(self.client.patch('/v1/images/' + identifier + '/tags', json={'tags': ['test']}).status_code, 503)
        self.assertEqual(self.client.get('/v1/images/' + identifier).json()['tags'], ['Legacy'])

    def test_filtered_pagination_metadata_and_boundaries(self):
        for number in range(5):
            self.save(f'{number:032x}', {'prompt': 'A castle', 'created': 100, 'tags': ['campaign']})
        params = {'limit': 2, 'offset': 0, 'q': 'campaign', 'sort': 'oldest', 'tag': 'CAMPAIGN'}
        first = self.client.get('/v1/images', params=params).json()
        self.assertEqual((first['total'], first['limit'], first['offset']), (5, 2, 0))
        self.assertFalse(first['has_previous'])
        self.assertNotIn('previous_offset', first)
        self.assertEqual(first['next_offset'], 2)
        second = self.client.get('/v1/images', params={**params, 'offset': first['next_offset']}).json()
        self.assertEqual(second['previous_offset'], 0)
        self.assertTrue(second['has_previous'])
        self.assertEqual(second['next_offset'], 4)
        self.assertFalse({item['id'] for item in first['data']} & {item['id'] for item in second['data']})
        last = self.client.get('/v1/images', params={**params, 'offset': second['next_offset']}).json()
        self.assertEqual(len(last['data']), 1)
        self.assertFalse(last['has_more'])
        self.assertEqual(last['previous_offset'], 2)
        outside = self.client.get('/v1/images', params={**params, 'offset': 100}).json()
        self.assertEqual(outside['data'], [])
        self.assertEqual(outside['total'], 5)
        self.assertFalse(outside['has_more'])
        schema = self.client.get('/openapi.json').json()
        self.assertIn('patch', schema['paths']['/v1/images/{identifier}/tags'])
        self.assertIn('tags', schema['components']['schemas']['CatalogImage']['properties'])

    def test_detail_and_metadata_download(self):
        identifier = 'a' * 32
        metadata = {'prompt': '<script>alert(1)</script>', 'seed': 42, 'steps': 9, 'seconds': 12.3,
                    'requested_size': '1024x1024', 'reference_image': '/tmp/photo.png', 'created': 100}
        path = self.save(identifier, metadata, index=True)
        response = self.client.get('/v1/images/' + identifier)
        self.assertEqual(response.status_code, 200)
        detail = response.json()
        self.assertEqual(detail['metadata'], metadata)
        self.assertEqual(detail['file_path'], str(path))
        self.assertEqual(detail['size'], '1024x640')
        self.assertNotIn('_version', detail)
        download = self.client.get(detail['metadata_url'])
        self.assertEqual(download.json(), metadata)
        self.assertIn('application/json', download.headers['content-type'])
        self.assertIn('attachment', download.headers['content-disposition'])

    def test_legacy_or_corrupt_sidecar_remains_browsable(self):
        for identifier, content in [('a' * 32, None), ('b' * 32, '{invalid'), ('c' * 32, '{"created":1e999}'), ('d' * 32, '[1,2]')]:
            path = self.save(identifier)
            if content:
                path.with_suffix('.json').write_text(content)
            os.utime(path, (100, 100))
            detail = self.client.get('/v1/images/' + identifier).json()
            self.assertFalse(detail['metadata_available'])
            self.assertNotIn('metadata_url', detail)
            self.assertEqual(detail['created'], 100)
            self.assertEqual(detail['size'], '1024x640')
            self.assertEqual(self.client.get('/images/' + identifier + '.json').status_code, 404)
        self.assertEqual(self.client.get('/v1/images').json()['total'], 4)

    def test_thumbnail_preserves_original_and_invalidates_cache(self):
        identifier = 'a' * 32
        path = self.save(identifier, size=(1024, 640))
        original = path.read_bytes()
        detail = self.client.get('/v1/images/' + identifier).json()
        response = self.client.get(detail['thumbnail_url'])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['content-type'], 'image/webp')
        with Image.open(io.BytesIO(response.content)) as preview:
            self.assertEqual(preview.size, (512, 320))
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(list((self.root / 'cache/thumbnails').glob('*.webp'))), 1)
        self.assertEqual(self.client.get(detail['thumbnail_url']).content, response.content)
        Image.new('RGB', (640, 1024), 'gold').save(path)
        changed = self.client.get('/v1/images/' + identifier).json()
        self.assertNotEqual(detail['thumbnail_url'], changed['thumbnail_url'])
        with Image.open(io.BytesIO(self.client.get(changed['thumbnail_url']).content)) as preview:
            self.assertEqual(preview.size, (320, 512))
        path.unlink()
        self.assertEqual(self.client.get(detail['thumbnail_url']).status_code, 404)

    def test_missing_files_and_browsing_during_generation(self):
        self.save('a' * 32, {'prompt': 'A fox'})
        server.busy.acquire()
        try:
            self.assertEqual(self.client.get('/v1/images').json()['total'], 1)
            self.assertEqual(self.client.get('/thumbnails/' + 'a' * 32 + '.webp').status_code, 200)
        finally:
            server.busy.release()
        for identifier in ['invalid', 'b' * 32, 'A' * 32]:
            self.assertEqual(self.client.get('/v1/images/' + identifier).status_code, 404)
            self.assertEqual(self.client.get('/thumbnails/' + identifier + '.webp').status_code, 404)
        with patch.object(server, 'ROOT', self.root / 'unmounted'):
            self.assertEqual(self.client.get('/v1/images').status_code, 503)

    def test_gallery_route(self):
        with patch.object(server,'PROJECT_ROOT',Path(__file__).resolve().parents[1]):
            response = self.client.get('/gallery')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.headers['content-type'])
        self.assertEqual(response.headers['cache-control'], 'no-store')


if __name__ == '__main__':
    unittest.main()
