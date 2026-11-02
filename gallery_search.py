"""Find similar gallery images: SigLIP 2 vectors for every saved picture, ranked against words or another picture.

SigLIP 2 (search_profile.py) maps a picture and a phrase to unit vectors in one space, so their
dot product says how well the words describe the picture, and two pictures' dot product says
how alike they look. This works for edited images too, whose prompt is an instruction rather
than a description. The model runs on the CPU only, so it never competes with an image model
for the GPU; it loads on first use and unloads after IDLE_SECONDS. Ranking is a brute-force
matrix product over the cached vectors, well under a millisecond for thousands of images.

The Indexer embeds images that have no vector yet (new saves, the existing library, files that
came back) in small batches, newest first. While a job is generating it waits, unless the model
is already loaded for a search.
"""
from collections import OrderedDict
import logging
import threading
import time

import numpy as np

from search_profile import EMBEDDING_MODEL

log = logging.getLogger('image-model-server')

# Cosine similarity cut-offs, tuned on a real 1,478-image library. SigLIP text–image scores are low
# in absolute terms: pictures the words describe scored 0.09–0.15, unrelated ones under 0.07, and how
# high the best match goes depends on the words. So words keep images within TEXT_MARGIN of their best
# match and at least TEXT_FLOOR. Alike pictures score 0.75 or more (the median pair is about 0.58);
# More like this always shows at least IMAGE_AT_LEAST of the nearest ones above IMAGE_FLOOR.
TEXT_FLOOR = 0.09
TEXT_MARGIN = 0.03
IMAGE_CUTOFF = 0.75
IMAGE_FLOOR = 0.6
IMAGE_AT_LEAST = 8
IDLE_SECONDS = 600
THREADS = 4


class Embedder:
    """Lazily loaded SigLIP 2 on the CPU; every call is serialized by one lock."""

    def __init__(self, folder, *, idle_seconds=IDLE_SECONDS, threads=THREADS):
        self.folder = folder
        self.idle_seconds = idle_seconds
        self.threads = threads
        self.lock = threading.Lock()
        self.model = self.processor = None
        self.last_used = 0.0
        self.queries = OrderedDict()
        self.queries_lock = threading.Lock()
        self.warming = False
        self._reaper = None

    @property
    def loaded(self):
        return self.model is not None

    def _load(self):
        if self.model is not None:
            return
        started = time.monotonic()
        import torch
        from transformers import AutoModel, AutoProcessor
        torch.set_num_threads(self.threads)
        model = AutoModel.from_pretrained(self.folder, dtype=torch.float32, local_files_only=True).eval()
        self.processor = AutoProcessor.from_pretrained(self.folder, local_files_only=True)
        self.model = model
        # The first call through each tower is several seconds slower; pay it here, not in a search.
        self._text(['a photo'])
        log.info('Gallery search model loaded on the CPU in %.1fs', time.monotonic() - started)
        if self._reaper is None or not self._reaper.is_alive():
            self._reaper = threading.Thread(target=self._unload_when_idle, name='gallery-search-idle', daemon=True)
            self._reaper.start()

    def _unload_when_idle(self):
        while True:
            time.sleep(min(30, self.idle_seconds))
            with self.lock:
                if self.model is not None and time.monotonic() - self.last_used >= self.idle_seconds:
                    self.model = self.processor = None
                    import gc
                    gc.collect()
                    log.info('Gallery search model unloaded after %ds idle', self.idle_seconds)
                    self._reaper = None
                    return

    @staticmethod
    def _unit(features):
        vectors = features.detach().float().numpy()
        return (vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)).astype(np.float32)

    @staticmethod
    def _pooled(output):
        return getattr(output, 'pooler_output', output)

    def _text(self, texts):
        import torch
        # SigLIP was trained on lowercase text padded to 64 tokens.
        inputs = self.processor(text=[text.casefold() for text in texts], padding='max_length', max_length=64,
                                truncation=True, return_tensors='pt')
        with torch.inference_mode():
            return self._unit(self._pooled(self.model.get_text_features(**inputs)))

    def warm(self):
        with self.lock:
            self._load()
            self.last_used = time.monotonic()

    def warm_in_background(self):
        """Load ahead of a likely search (the gallery page opened) without making anyone wait."""
        if self.loaded or self.warming:
            return
        self.warming = True

        def run():
            try:
                self.warm()
            except Exception:  # noqa: BLE001 - a failed warm-up reappears as an unavailable search
                log.warning('Could not load the gallery search model', exc_info=True)
            finally:
                self.warming = False
        threading.Thread(target=run, name='gallery-search-warm', daemon=True).start()

    def text(self, query, timeout=None):
        """Unit vector for a search phrase, or None if the model is busy loading past timeout."""
        key = ' '.join(query.casefold().split())
        with self.queries_lock:
            if key in self.queries:
                self.queries.move_to_end(key)
                return self.queries[key]
        if not self.lock.acquire(timeout=-1 if timeout is None else timeout):
            return None
        try:
            self._load()
            vector = self._text([key])[0]
            self.last_used = time.monotonic()
        finally:
            self.lock.release()
        with self.queries_lock:
            self.queries[key] = vector
            while len(self.queries) > 64:
                self.queries.popitem(last=False)
        return vector

    def images(self, pictures):
        """Unit vectors for PIL images (any mode), one row each."""
        import torch
        with self.lock:
            self._load()
            inputs = self.processor(images=[picture.convert('RGB') for picture in pictures], return_tensors='pt')
            with torch.inference_mode():
                vectors = self._unit(self._pooled(self.model.get_image_features(**inputs)))
            self.last_used = time.monotonic()
        return vectors


class VectorIndex:
    """Every stored vector of one model as a matrix, reloaded only when the database's vectors change."""

    def __init__(self, store_factory, model=EMBEDDING_MODEL):
        self.store_factory = store_factory
        self.model = model
        self.lock = threading.Lock()
        self.revision = None
        self.numbers = np.zeros(0, np.int64)
        self.matrix = np.zeros((0, 0), np.float32)

    def _current(self):
        store = self.store_factory()
        revision = store.vectors_revision()
        with self.lock:
            if revision != self.revision:
                self.numbers, self.matrix = store.vectors(self.model)
                self.revision = revision
            return self.numbers, self.matrix

    def rank(self, vector, cutoff, exclude=None):
        """[(image number, similarity)] at or above cutoff, best first."""
        numbers, matrix = self._current()
        if not len(numbers) or vector is None:
            return []
        scores = matrix @ np.asarray(vector, np.float32)
        order = np.argsort(-scores, kind='stable')
        return [(int(numbers[index]), float(scores[index])) for index in order
                if scores[index] >= cutoff and numbers[index] != exclude]

    def vector_of(self, number):
        numbers, matrix = self._current()
        found = np.flatnonzero(numbers == number)
        return matrix[found[0]] if len(found) else None


def described_by(index, vector):
    """[(image number, similarity)] for pictures the search words describe, best first."""
    ranked = index.rank(vector, TEXT_FLOOR)
    return [item for item in ranked if item[1] >= ranked[0][1] - TEXT_MARGIN]


def looking_like(index, vector, number):
    """[(image number, similarity)] for pictures that look like image number, best first."""
    ranked = index.rank(vector, IMAGE_FLOOR, exclude=number)
    close = [item for item in ranked if item[1] >= IMAGE_CUTOFF]
    return close if len(close) >= IMAGE_AT_LEAST else ranked[:IMAGE_AT_LEAST]


class Indexer:
    """Background thread that gives every visible gallery image a vector."""

    def __init__(self, store_factory, embedder, *, busy=lambda: False, model=EMBEDDING_MODEL, batch=8, interval=30, reader=None):
        self.store_factory = store_factory
        self.embedder = embedder
        self.busy = busy
        self.model = model
        self.batch = batch
        self.interval = interval
        self.read_image = reader or read_image
        self.wakeup = threading.Event()
        self.stopping = threading.Event()
        self.thread = None
        self.state = 'starting'
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self._loop, name='gallery-indexer', daemon=True)
        self.thread.start()

    def wake(self):
        self.wakeup.set()

    def stop(self):
        self.stopping.set()
        self.wakeup.set()

    def run_once(self):
        """Embed one batch of waiting images; returns how many were handled."""
        store = self.store_factory()
        pending = store.unembedded(self.model, self.batch)
        if not pending:
            return 0
        pictures, readable, entries, missing = [], [], [], []
        for row in pending:
            try:
                pictures.append(self.read_image(row['file_path']))
                readable.append(row)
            except FileNotFoundError:
                missing.append(row['id'])
            except OSError:
                # Unreadable or not an image: remember that, so it is not retried forever.
                entries.append((row['number'], row['id'], None))
        if missing:
            store.mark_missing(missing)
        if pictures:
            vectors = self.embedder.images(pictures)
            entries += [(row['number'], row['id'], vector) for row, vector in zip(readable, vectors)]
        if entries:
            store.save_vectors(self.model, entries)
        return len(pending)

    def _loop(self):
        while not self.stopping.is_set():
            if self.busy() and not self.embedder.loaded:
                self.state = 'waiting'
                self.wakeup.wait(self.interval)
                self.wakeup.clear()
                continue
            try:
                self.state = 'indexing'
                handled = self.run_once()
                self.error = None
            except Exception as error:  # noqa: BLE001 - indexing must never stop the generator
                message = str(error) or type(error).__name__
                if message != self.error:
                    log.warning('Gallery search indexing failed; retrying every %ds', self.interval, exc_info=True)
                self.error, handled = message, 0
                self.stopping.wait(self.interval)
            if not handled:
                self.state = 'idle'
                self.wakeup.wait(self.interval)
                self.wakeup.clear()

    def status(self):
        indexed, waiting = self.store_factory().vector_counts(self.model)
        return {'state': self.state, 'indexed': indexed, 'waiting': waiting, 'model_loaded': self.embedder.loaded, 'error': self.error}


def read_image(path):
    from PIL import Image
    with Image.open(path) as picture:
        picture.draft('RGB', (512, 512))
        return picture.convert('RGB')


def main(argv=None):
    import argparse
    argparse.ArgumentParser(description='Give every gallery image a search vector now (the running generator also does this in the background).').parse_args(argv)
    import gallery_store
    from config import PROJECT_ROOT
    from search_profile import check_assets
    store = gallery_store.GalleryStore(gallery_store.gallery_database(PROJECT_ROOT))
    indexer = Indexer(lambda: store, Embedder(check_assets(PROJECT_ROOT)), batch=16)
    started, done = time.monotonic(), 0
    while handled := indexer.run_once():
        done += handled
        print(f'Indexed {done} image(s)…', flush=True)
    indexed, waiting = store.vector_counts(EMBEDDING_MODEL)
    print(f'Gallery search index ready: {indexed} image(s), {waiting} waiting, {time.monotonic() - started:.0f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
