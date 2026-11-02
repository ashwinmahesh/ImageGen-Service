"""Similar-image gallery search with a fake embedder: no model download or inference."""
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import gallery_search
import gallery_store
import search_profile
import server

MODEL = gallery_search.EMBEDDING_MODEL
COLORS = {'red': (1, 0, 0), 'green': (0, 1, 0), 'blue': (0, 0, 1)}
WORDS = {'crimson': (1, 0.1, 0), 'leafy': (0, 1, 0.1)}


def unit(values):
    vector = np.asarray(values, np.float32)
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Pictures map to their dominant color; a few words map near a color."""
    def __init__(self):
        self.loaded, self.calls, self.warmed = False, [], 0

    def images(self, pictures):
        self.calls.append(len(pictures))
        return np.stack([unit(np.asarray(picture.convert('RGB').getpixel((0, 0)), np.float32) + 1e-3) for picture in pictures])

    def text(self, query, timeout=None):
        return unit(WORDS.get(query.casefold(), (1, 1, 1)))

    def warm_in_background(self):
        self.warmed += 1


class Library(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.images = self.root / 'images'; self.images.mkdir()
        self.store = gallery_store.GalleryStore(gallery_store.gallery_database(self.root))

    def tearDown(self):
        self.temp.cleanup()

    def save(self, identifier, prompt, created, color, *, tags=()):
        path = self.images / f'{identifier}.png'
        Image.new('RGB', (32, 24), color).save(path)
        metadata = {'prompt': prompt, 'created': created, 'tags': list(tags)}
        path.with_suffix('.json').write_text(json.dumps(metadata))
        self.store.register(identifier, root=self.root, origin='z-image-turbo', path=path, metadata=metadata, tags=tags)
        return path

    def index(self, embedder=None):
        indexer = gallery_search.Indexer(lambda: self.store, embedder or FakeEmbedder(), batch=2)
        while indexer.run_once():
            pass
        return indexer


class VectorStoreTest(Library):
    def test_vectors_are_stored_counted_and_removed_with_their_image(self):
        a, b = 'a' * 32, 'b' * 32
        self.save(a, 'Red barn', 1, 'red'); self.save(b, 'Green field', 2, 'green')
        self.assertEqual([row['id'] for row in self.store.unembedded(MODEL)], [b, a])
        self.assertEqual(self.store.vector_counts(MODEL), (0, 2))
        before = self.store.vectors_revision()
        self.index()
        self.assertEqual(self.store.unembedded(MODEL), [])
        self.assertEqual(self.store.vector_counts(MODEL), (2, 0))
        self.assertGreater(self.store.vectors_revision(), before)
        numbers, matrix = self.store.vectors(MODEL)
        self.assertEqual(matrix.shape, (2, 3)); self.assertEqual(matrix.dtype, np.float32)
        # Another model's vectors never mix into this one's space.
        self.assertEqual(self.store.vector_counts('other'), (0, 2))
        revision = self.store.vectors_revision()
        self.store.remove(a)
        self.assertGreater(self.store.vectors_revision(), revision)
        self.assertEqual(len(self.store.vectors(MODEL)[0]), 1)
        with self.store.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM image_vectors').fetchone()[0], 1)

    def test_vector_for_a_deleted_or_replaced_row_is_not_saved(self):
        a = 'a' * 32
        self.save(a, 'Red barn', 1, 'red')
        number = self.store.get(a)['number']
        self.store.remove(a)
        self.save('c' * 32, 'Reused number', 2, 'blue')
        self.assertEqual(self.store.get('c' * 32)['number'], number)
        self.store.save_vectors(MODEL, [(number, a, unit((1, 0, 0)))])
        self.assertEqual(self.store.vector_counts(MODEL), (0, 1))

    def test_unreadable_and_missing_files_are_not_retried(self):
        a, b, c = 'a' * 32, 'b' * 32, 'c' * 32
        self.save(a, 'Red', 1, 'red'); broken = self.save(b, 'Broken', 2, 'red'); gone = self.save(c, 'Gone', 3, 'red')
        broken.write_bytes(b'not a png'); gone.unlink()
        embedder = FakeEmbedder()
        self.index(embedder)
        self.assertEqual(self.store.unembedded(MODEL), [])
        self.assertEqual(self.store.get(c)['missing'], 1)
        self.assertEqual(len(self.store.vectors(MODEL)[0]), 1)  # the broken file is marked, not embedded
        self.assertEqual(sum(embedder.calls), 1)
        # A file that comes back is embedded then.
        Image.new('RGB', (32, 24), 'blue').save(gone)
        self.store.recheck_missing(); self.index(embedder)
        self.assertEqual(len(self.store.vectors(MODEL)[0]), 2)

    def test_schema_two_upgrades_version_one_databases(self):
        path = self.root / 'old.sqlite3'
        connection = sqlite3.connect(path)
        connection.execute('CREATE TABLE images (number INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, root TEXT NOT NULL, model TEXT, '
                           "file_path TEXT NOT NULL, created REAL NOT NULL, prompt TEXT, seed INTEGER, steps INTEGER, size TEXT, width INTEGER, "
                           "height INTEGER, generation_seconds REAL, lora TEXT, loras TEXT, metadata TEXT, tags TEXT NOT NULL DEFAULT '[]', "
                           'missing INTEGER NOT NULL DEFAULT 0)')
        connection.execute('PRAGMA user_version=1'); connection.commit(); connection.close()
        store = gallery_store.GalleryStore(path)
        self.assertEqual(store.vector_counts(MODEL), (0, 0))
        with store.connection() as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 2)

    def test_exact_matches_come_first_then_similar_by_rank_with_tags_and_paging(self):
        red, red2, green, blue = 'a' * 32, 'b' * 32, 'c' * 32, 'd' * 32
        self.save(red, 'Crimson barn', 1, 'red', tags=['farm'])
        self.save(red2, 'A barn at dusk', 2, 'red', tags=['farm'])
        self.save(green, 'Meadow', 3, 'green', tags=['farm'])
        self.save(blue, 'Ocean', 4, 'blue')
        self.index()
        numbers = {identifier: self.store.get(identifier)['number'] for identifier in (red, red2, green, blue)}
        related = [(numbers[red], 0.99), (numbers[red2], 0.98), (numbers[green], 0.2), (numbers[blue], 0.1)]
        rows, total, similar = self.store.search_similar('crimson', related=related)
        self.assertEqual([(row['id'], row['_match']) for row in rows],
                         [(red, 'exact'), (red2, 'similar'), (green, 'similar'), (blue, 'similar')])
        self.assertEqual((total, similar), (4, 3))
        # Pages run across the boundary between the groups.
        page, _, _ = self.store.search_similar('crimson', limit=2, offset=1, related=related)
        self.assertEqual([row['id'] for row in page], [red2, green])
        # Tags filter similar images too; hidden images are skipped.
        rows, total, similar = self.store.search_similar('crimson', tags=['farm'], related=related)
        self.assertEqual([row['id'] for row in rows], [red, red2, green])
        self.store.mark_missing([red2])
        rows, total, similar = self.store.search_similar('', related=related)
        self.assertEqual([row['id'] for row in rows], [red, green, blue])
        self.assertTrue(all(row['_match'] == 'similar' for row in rows))
        self.assertEqual(gallery_store.public_record(rows[0], (self.images / f'{red}.png').stat())['similarity'], 0.99)
        # Plain search keeps its contract and marks words as exact matches.
        rows, total = self.store.search('crimson')
        self.assertEqual(([row['id'] for row in rows], rows[0]['_match'], total), ([red], 'exact', 1))


class VectorIndexTest(Library):
    def test_rank_applies_cutoff_and_exclusion_and_reloads_after_writes(self):
        a, b, c = 'a' * 32, 'b' * 32, 'c' * 32
        self.save(a, 'Red', 1, 'red'); self.save(b, 'Green', 2, 'green')
        self.index()
        index = gallery_search.VectorIndex(lambda: self.store)
        number = self.store.get(a)['number']
        self.assertEqual([item[0] for item in index.rank(unit((1, 0, 0)), 0.5)], [number])
        self.assertEqual(index.rank(unit((1, 0, 0)), 0.5, exclude=number), [])
        self.assertEqual(index.rank(None, 0), [])
        np.testing.assert_allclose(index.vector_of(number), unit((1, 0, 0)), atol=1e-2)
        self.save(c, 'Also red', 3, 'red'); self.index()
        self.assertEqual(len(index.rank(unit((1, 0, 0)), 0.5)), 2)

    def test_words_keep_matches_near_the_best_and_pictures_keep_at_least_the_nearest(self):
        index = gallery_search.VectorIndex(lambda: self.store)
        index.rank = lambda vector, cutoff, exclude=None: [item for item in [(1, 0.15), (2, 0.13), (3, 0.11), (4, 0.095), (5, 0.08)]
                                                           if item[1] >= cutoff and item[0] != exclude]
        self.assertEqual([number for number, _ in gallery_search.described_by(index, None)], [1, 2])
        index.rank = lambda vector, cutoff, exclude=None: [item for item in [(1, 0.95), (2, 0.9), (3, 0.7), (4, 0.65), (5, 0.5)]
                                                           if item[1] >= cutoff and item[0] != exclude]
        with patch.object(gallery_search, 'IMAGE_AT_LEAST', 2):
            self.assertEqual([number for number, _ in gallery_search.looking_like(index, None, 9)], [1, 2])
        with patch.object(gallery_search, 'IMAGE_AT_LEAST', 3):
            self.assertEqual([number for number, _ in gallery_search.looking_like(index, None, 1)], [2, 3, 4])


class IndexerThreadTest(Library):
    def test_waits_while_generating_unless_the_model_is_already_loaded(self):
        self.save('a' * 32, 'Red', 1, 'red')
        embedder, generating = FakeEmbedder(), threading.Event(); generating.set()
        indexer = gallery_search.Indexer(lambda: self.store, embedder, busy=generating.is_set, interval=0.05)
        indexer.start()
        try:
            time.sleep(0.3)
            self.assertEqual((indexer.state, embedder.calls), ('waiting', []))
            embedder.loaded = True; indexer.wake()
            deadline = time.monotonic() + 5
            while self.store.vector_counts(MODEL) != (1, 0) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(self.store.vector_counts(MODEL), (1, 0))
        finally:
            indexer.stop(); indexer.thread.join(5)

    def test_failures_are_recorded_and_retried_without_stopping(self):
        self.save('a' * 32, 'Red', 1, 'red')
        embedder = FakeEmbedder(); real = embedder.images; failures = []

        def flaky(pictures):
            if not failures:
                failures.append(1); raise RuntimeError('model failed to load')
            return real(pictures)
        embedder.images = flaky
        indexer = gallery_search.Indexer(lambda: self.store, embedder, interval=0.05)
        with self.assertLogs('image-model-server', 'WARNING'):
            indexer.start()
            deadline = time.monotonic() + 5
            while self.store.vector_counts(MODEL) != (1, 0) and time.monotonic() < deadline:
                time.sleep(0.02)
        indexer.stop(); indexer.thread.join(5)
        self.assertEqual(self.store.vector_counts(MODEL), (1, 0))
        self.assertIsNone(indexer.error)


class EmbedderTest(unittest.TestCase):
    def test_query_vectors_are_cached_and_a_busy_load_times_out(self):
        embedder = gallery_search.Embedder('/nonexistent')
        texts = []
        embedder._load = lambda: None
        embedder._text = lambda batch: texts.extend(batch) or np.ones((1, 3), np.float32)
        embedder.text('  Red   Barn '); embedder.text('red barn')
        self.assertEqual(texts, ['red barn'])
        with embedder.lock:
            result = []
            thread = threading.Thread(target=lambda: result.append(embedder.text('new words', timeout=0.05)))
            thread.start(); thread.join(5)
        self.assertEqual(result, [None])


class SearchAssetsTest(unittest.TestCase):
    def test_missing_or_other_revision_assets_are_reported_without_downloading(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ', {'DATA_DIR': temp}):
            with self.assertRaisesRegex(RuntimeError, 'make download-search'):
                search_profile.check_assets(temp)
            folder = search_profile.weights_directory(temp); folder.mkdir(parents=True)
            for name in search_profile.FILES:
                (folder / name).write_text('x')
            (folder / 'download.json').write_text(json.dumps({'repository': search_profile.REPOSITORY, 'revision': 'old'}))
            with self.assertRaisesRegex(RuntimeError, 'revision differs'):
                search_profile.check_assets(temp)
            (folder / 'download.json').write_text(json.dumps({'repository': search_profile.REPOSITORY, 'revision': search_profile.REVISION}))
            self.assertEqual(search_profile.check_assets(temp), folder)


class SimilarSearchApiTest(Library):
    def setUp(self):
        super().setUp()
        self.embedder = FakeEmbedder()
        self.patches = [patch.object(server, 'PROJECT_ROOT', self.root), patch.object(server, 'ROOT', self.root),
                        patch.object(server, 'IMAGE_DIR', self.images)]
        for item in self.patches:
            item.start()
        self.store = server.gallery_db()
        self.client = TestClient(server.app, base_url='http://127.0.0.1')

    def enable(self):
        indexer = gallery_search.Indexer(server.gallery_db, self.embedder)
        for item in [patch.object(server, 'gallery_embedder', self.embedder), patch.object(server, 'gallery_indexer', indexer),
                     patch.object(server, 'gallery_vectors', gallery_search.VectorIndex(server.gallery_db))]:
            item.start(); self.patches.append(item)
        return indexer

    def tearDown(self):
        self.client.close()
        for item in reversed(self.patches):
            item.stop()
        super().tearDown()

    def library(self):
        self.ids = {'red': 'a' * 32, 'red2': 'b' * 32, 'green': 'c' * 32, 'blue': 'd' * 32}
        self.save(self.ids['red'], 'Crimson barn', 1, 'red')
        self.save(self.ids['red2'], 'Barn at dusk', 2, 'red')
        self.save(self.ids['green'], 'Meadow', 3, 'green')
        self.save(self.ids['blue'], 'Ocean', 4, 'blue')

    def test_word_search_lists_exact_matches_then_similar_pictures(self):
        self.library(); self.index(self.embedder); self.enable()
        response = self.client.get('/v1/images', params={'q': 'crimson'})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual([(item['id'], item['match']) for item in data['data']],
                         [(self.ids['red'], 'exact'), (self.ids['red2'], 'similar')])
        self.assertNotIn('similarity', data['data'][0]); self.assertGreater(data['data'][1]['similarity'], 0.9)
        self.assertEqual((data['total'], data['similar_total'], data['similar_search']), (2, 1, 'ready'))
        # No query: a plain listing with no search fields.
        plain = self.client.get('/v1/images').json()
        self.assertEqual(plain['total'], 4)
        self.assertNotIn('similar_search', plain); self.assertNotIn('match', plain['data'][0])

    def test_without_the_search_model_or_while_it_loads_exact_search_still_answers(self):
        self.library()
        data = self.client.get('/v1/images', params={'q': 'crimson'}).json()
        self.assertEqual(([item['id'] for item in data['data']], data['similar_search'], data['similar_total']), ([self.ids['red']], 'unavailable', 0))
        self.assertEqual(self.client.get('/health').json()['gallery_search']['available'], False)
        self.assertEqual(self.client.get('/v1/images', params={'similar_to': self.ids['red']}).status_code, 503)
        self.enable(); self.index(self.embedder)
        self.embedder.text = lambda query, timeout=None: None
        data = self.client.get('/v1/images', params={'q': 'crimson'}).json()
        self.assertEqual((data['total'], data['similar_search']), (1, 'loading'))

        def broken(query, timeout=None):
            raise RuntimeError('torch failed')
        self.embedder.text = broken
        with self.assertLogs('image-model-server', 'WARNING'):
            data = self.client.get('/v1/images', params={'q': 'crimson'}).json()
        self.assertEqual((data['total'], data['similar_search']), (1, 'unavailable'))

    def test_more_like_this_ranks_by_picture_and_embeds_an_unindexed_source(self):
        self.library(); self.enable()
        # Nothing is indexed yet: the source image is embedded on demand.
        data = self.client.get('/v1/images', params={'similar_to': self.ids['red']}).json()
        self.assertEqual((data['total'], data['similar_search']), (0, 'ready'))
        self.assertEqual(self.store.vector_counts(MODEL)[0], 1)
        self.index(self.embedder)
        data = self.client.get('/v1/images', params={'similar_to': self.ids['red'], 'sort': 'oldest'}).json()
        self.assertEqual([(item['id'], item['match']) for item in data['data']], [(self.ids['red2'], 'similar')])
        self.assertEqual(data['similar_total'], 1)
        self.assertEqual(self.client.get('/v1/images', params={'similar_to': 'f' * 32}).status_code, 404)
        self.assertEqual(self.client.get('/v1/images', params={'similar_to': self.ids['red'], 'q': 'barn'}).status_code, 422)
        self.assertEqual(self.client.get('/v1/images', params={'similar_to': '../x'}).status_code, 422)

    def test_health_reports_indexing_and_the_gallery_page_warms_the_model(self):
        self.library(); self.enable()
        status = self.client.get('/health').json()['gallery_search']
        self.assertEqual((status['available'], status['indexed'], status['waiting']), (True, 0, 4))
        with patch.object(server, 'render_page', return_value='<html></html>'):
            self.assertEqual(self.client.get('/gallery').status_code, 200)
        self.assertEqual(self.embedder.warmed, 1)

    def test_deleting_an_image_removes_its_vector(self):
        self.library(); self.index(self.embedder); self.enable()
        with patch.object(server, 'job_store') as job_store, patch.object(server, 'queue_ready'):
            job_store.guard_artifact_deletion.return_value.__enter__.return_value = None
            response = self.client.delete('/v1/images/' + self.ids['red2'])
        self.assertEqual(response.status_code, 200, response.text)
        data = self.client.get('/v1/images', params={'similar_to': self.ids['red']}).json()
        self.assertEqual(data['total'], 0)

    def test_startup_without_the_model_leaves_search_off(self):
        with patch.dict('os.environ', {'DATA_DIR': str(self.root)}), patch.object(server, 'gallery_indexer', None), \
                patch.object(server, 'gallery_search_error', None), self.assertLogs('image-model-server', 'INFO'):
            server.start_gallery_search()
            self.assertIsNone(server.gallery_indexer)
            self.assertIn('make download-search', server.gallery_search_status()['reason'])


if __name__ == '__main__':
    unittest.main()
