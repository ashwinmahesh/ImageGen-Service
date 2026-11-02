"""The SQLite gallery database: registration, FTS search, tags, imports and hidden missing files."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import gallery_store
import server


class GalleryStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.images = self.root / 'images'; self.images.mkdir()
        self.store = gallery_store.GalleryStore(gallery_store.gallery_database(self.root))

    def tearDown(self):
        self.temp.cleanup()

    def save(self, identifier, prompt, created, *, tags=(), folder=None, register=True, **extra):
        folder = folder or self.images; folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{identifier}.png'
        Image.new('RGB', (32, 24), 'teal').save(path)
        metadata = {'prompt': prompt, 'created': created, 'tags': list(tags), **extra}
        path.with_suffix('.json').write_text(json.dumps(metadata))
        if register:
            self.store.register(identifier, root=self.root, origin='z-image-turbo', path=path, metadata=metadata, tags=tags)
        return path

    def ids(self, **options):
        rows, total = self.store.search(**options)
        return [row['id'] for row in rows], total

    def test_words_match_by_prefix_in_any_order_with_tags_and_paging(self):
        a, b, c = 'a' * 32, 'b' * 32, 'c' * 32
        self.save(a, 'Misty castle at dawn', 100, tags=['Fantasy'], seed=7)
        self.save(b, 'Quiet forest café', 200, lora={'id': 'custom-ink', 'name': 'Ink Wash', 'strength': 0.6})
        self.save(c, 'Castle courtyard', 300, tags=['Fantasy', 'Night'], size='1024x768')
        self.assertEqual(self.ids(), ([c, b, a], 3))
        self.assertEqual(self.ids(order='oldest', limit=1, offset=1), ([b], 3))
        for query, expected in [('castle', [c, a]), ('dawn castle', [a]), ('cas', [c, a]), ('CAFE', [b]), ('ink', [b]),
                                ('wash', [b]), ('custom-ink', [b]), ('7', [a]), ('1024x768', [c]), ('turbo', [c, b, a]),
                                ('fantasy night', [c]), ('astle', []), ('castle forest', [])]:
            with self.subTest(query=query):
                self.assertEqual(self.ids(query=query)[0], expected)
        self.assertEqual(self.ids(tags=['fantasy']), ([c, a], 2))
        self.assertEqual(self.ids(tags=['FANTASY', 'night']), ([c], 1))
        self.assertEqual(self.ids(tags=['fan']), ([], 0))
        self.assertEqual(self.ids(query='castle', tags=['night'], order='oldest'), ([c], 1))

    def test_tags_reregistration_and_removal_keep_search_in_step(self):
        a, b = 'a' * 32, 'b' * 32
        path = self.save(a, 'Misty castle', 100, tags=['Draft'])
        self.save(b, 'Quiet forest', 200)
        row = self.store.set_tags(a, ['Keeper', 'keeper', 'Coast'])
        self.assertEqual(json.loads(row['tags']), ['Keeper', 'Coast'])
        self.assertEqual(self.ids(query='keeper'), ([a], 1))
        self.assertEqual(self.ids(tags=['draft']), ([], 0))
        self.assertIsNone(self.store.set_tags('f' * 32, ['x']))
        # Registering again updates the same row (as recovery does) rather than adding one.
        self.store.register(a, root=self.root, origin='z-image-turbo', path=path, metadata={'prompt': 'Sunny meadow', 'created': 100}, tags=['Keeper'])
        self.assertEqual(self.ids(query='meadow'), ([a], 1))
        self.assertEqual(self.ids(query='misty'), ([], 0))
        # Imports never overwrite an existing row.
        self.assertFalse(self.store.register(a, root=self.root, origin='z-image-turbo', path=path, metadata={'prompt': 'Old'}, tags=[], replace=False))
        self.assertEqual(self.ids(query='meadow'), ([a], 1))
        self.store.remove(a)
        self.assertEqual(self.ids(), ([b], 1))
        self.assertEqual(self.ids(query='meadow'), ([], 0))

    def test_concurrent_writers_lose_nothing(self):
        def write(number):
            self.save(f'{number:032x}', f'Image {number}', number)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(write, range(40)))
        self.assertEqual(self.ids(limit=100)[1], 40)

    def test_import_is_once_per_folder_keeps_a_backup_and_rescans_on_request(self):
        a, b = 'a' * 32, 'b' * 32
        self.save(a, 'Misty castle', 100, register=False)
        custom = self.save(b, 'Quiet forest', 200, folder=self.root / 'custom', register=False)
        index = self.root / 'image-index'; index.mkdir()
        (index / f'{b}.json').write_text(json.dumps({'file_path': str(custom), 'tags': ['Edited']}))
        (self.root / 'image-index.imported').mkdir()  # an earlier backup is never overwritten
        roots = {'z-image-turbo': (self.root, self.images)}
        self.assertEqual(gallery_store.import_library(self.store, roots), 2)
        self.assertEqual(json.loads(self.store.get(b)['tags']), ['Edited'])
        self.assertTrue((self.root / 'image-index.imported-2' / f'{b}.json').is_file())
        self.assertFalse(index.exists())
        self.save('c' * 32, 'Copied in', 300, register=False)
        self.assertEqual(gallery_store.import_library(self.store, roots), 0)
        self.assertEqual(gallery_store.import_library(self.store, roots, rescan=True), 1)
        self.assertEqual(self.ids()[1], 3)

    def test_newer_database_versions_are_refused(self):
        path = self.root / 'state' / 'gallery' / 'future.sqlite3'; path.parent.mkdir(parents=True)
        connection = sqlite3.connect(path); connection.execute('PRAGMA user_version=9'); connection.close()
        with self.assertRaisesRegex(RuntimeError, 'Unsupported gallery database version'):
            gallery_store.GalleryStore(path).get('a' * 32)


class MissingFilesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.images = self.root / 'images'; self.images.mkdir()
        self.patches = [patch.object(server, 'PROJECT_ROOT', self.root), patch.object(server, 'ROOT', self.root), patch.object(server, 'IMAGE_DIR', self.images)]
        for item in self.patches:
            item.start()
        self.client = TestClient(server.app, base_url='http://127.0.0.1')

    def tearDown(self):
        self.client.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_a_page_hides_images_whose_file_is_gone_refills_and_shows_them_when_back(self):
        paths = []
        for number in range(5):
            identifier = f'{number:032x}'; path = self.images / f'{identifier}.png'
            Image.new('RGB', (16, 16), 'gold').save(path); paths.append(path)
            server.gallery_db().register(identifier, root=self.root, origin=server.API_MODEL, path=path,
                                         metadata={'prompt': 'A fox', 'created': number + 1}, tags=[])
        moved = [path.rename(path.with_suffix('.moved')) for path in paths[3:]]
        page = self.client.get('/v1/images?limit=2').json()
        self.assertEqual(([item['id'] for item in page['data']], page['total']), ([f'{2:032x}', f'{1:032x}'], 3))
        self.assertEqual(self.client.get('/v1/images/' + f'{4:032x}').status_code, 404)
        for path, original in zip(moved, paths[3:]):
            path.rename(original)
        server.prepare_gallery()
        self.assertEqual(self.client.get('/v1/images?limit=2').json()['total'], 5)


if __name__ == '__main__':
    unittest.main()
