"""The gallery's in-memory library index: pages and searches without rereading unchanged sidecars."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
import image_catalog


class GalleryIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.images = self.root / 'images'; self.images.mkdir()
        self.custom = self.root / 'custom'

    def tearDown(self):
        self.temp.cleanup()

    def save(self, identifier, prompt, created, *, folder=None, tags=()):
        folder = folder or self.images; folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{identifier}.png'
        Image.new('RGB', (64, 48), 'teal').save(path)
        temporary = path.with_name('.sidecar.tmp')
        temporary.write_text(json.dumps({'prompt': prompt, 'created': created, 'tags': list(tags), 'size': '64x48'}))
        temporary.replace(path.with_suffix('.json'))
        if folder != self.images:
            image_catalog.write_index(self.root, identifier, path, list(tags))
        return path

    def settle(self):
        """Age every folder past the settle window, as on a library nobody is writing to."""
        old = time.time() - 60
        for folder in [self.images, self.root / 'image-index', self.custom]:
            if folder.is_dir():
                os.utime(folder, (old, old))

    def listing(self, **options):
        items, total = image_catalog.list_images(self.root, self.images, **options)
        return [item['id'] for item in items], total

    def test_pages_search_tags_and_order_come_from_the_index(self):
        for index, (prompt, tags) in enumerate([('Misty castle', ['fantasy']), ('Quiet forest', []), ('Castle at dawn', ['fantasy', 'dawn'])]):
            self.save(f'{index:032x}', prompt, 100 + index, folder=self.custom if index == 1 else None, tags=tags)
        ids = [f'{index:032x}' for index in range(3)]
        self.assertEqual(self.listing(), (ids[::-1], 3))
        self.assertEqual(self.listing(limit=1, offset=1), ([ids[1]], 3))
        self.assertEqual(self.listing(order='oldest'), (ids, 3))
        self.assertEqual(self.listing(query='CASTLE'), ([ids[2], ids[0]], 2))
        self.assertEqual(self.listing(tags=['Fantasy', 'dawn']), ([ids[2]], 1))
        self.assertEqual(self.listing(query='castle', order='oldest', limit=1), ([ids[0]], 2))

    def test_unchanged_library_reads_no_files_and_changes_reread_only_what_changed(self):
        a, b, c = 'a' * 32, 'b' * 32, 'c' * 32
        self.save(a, 'Misty castle', 100); self.save(b, 'Quiet forest', 200, folder=self.custom)
        self.assertEqual(self.listing(), ([b, a], 2))
        self.settle()
        self.listing()  # Records the settled folder times.
        with patch.object(image_catalog, 'image_record', wraps=image_catalog.image_record) as reads, \
             patch.object(image_catalog, '_record_stamp', wraps=image_catalog._record_stamp) as stats:
            self.assertEqual(self.listing(query='forest'), ([b], 1))
            self.assertEqual((reads.call_count, stats.call_count), (0, 0))
            # A new image and a rewritten sidecar elsewhere: only the new or changed ones are read again.
            self.save(c, 'Lantern', 300)
            path = self.custom / f'{b}.json'; temporary = self.custom / 'next.tmp'
            temporary.write_text(json.dumps({'prompt': 'Snowy forest', 'created': 200, 'size': '64x48'})); temporary.replace(path)
            self.assertEqual(self.listing(query='snowy'), ([b], 1))
            self.assertEqual(sorted(call.args[2] for call in reads.call_args_list), [b, c])
        # Removing a file outside the server drops it from the next listing.
        (self.images / f'{a}.png').unlink()
        self.assertEqual(self.listing(), ([c, b], 2))

    def test_recently_changed_folders_are_rechecked_file_by_file(self):
        a = 'a' * 32
        self.save(a, 'Misty castle', 100)
        self.assertEqual(self.listing(query='castle'), ([a], 1))
        # Rewrite in place and put the folder time back, as a coarse 10 ms timestamp could:
        # the folder was fresh at the last listing, so its signature was not trusted.
        folder_time = self.images.stat().st_mtime_ns
        (self.images / f'{a}.json').write_text(json.dumps({'prompt': 'Sunny meadow with flowers', 'created': 100, 'size': '64x48'}))
        os.utime(self.images, ns=(folder_time, folder_time))
        self.assertEqual(self.listing(query='meadow'), ([a], 1))

    def test_tag_edits_and_deletes_in_this_process_show_immediately(self):
        a, b = 'a' * 32, 'b' * 32
        path = self.save(a, 'Misty castle', 100); self.save(b, 'Quiet forest', 200)
        self.listing(); self.settle(); self.listing()
        image_catalog.update_tags(self.root, self.images, a, ['keeper'])
        self.assertEqual(self.listing(tags=['keeper']), ([a], 1))
        image_catalog.delete_image(self.root, self.images, a, expected_path=path)
        self.assertEqual(self.listing(), ([b], 1))

    def test_missing_storage_still_fails(self):
        with self.assertRaises(OSError):
            image_catalog.list_images(self.root / 'missing', self.images)


if __name__ == '__main__':
    unittest.main()
