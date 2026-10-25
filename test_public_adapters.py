"""Public setup must not inspect or download personal serving presets."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import config
import download_loras
import lora_catalog


class PublicAdaptersTest(unittest.TestCase):
    def test_shipped_catalog_contains_only_optional_realism(self):
        entries=json.loads((Path(__file__).parent/'lora-catalog.json').read_text())
        self.assertEqual([entry['id'] for entry in entries], ['realism'])
        self.assertEqual(entries[0]['repository'], 'suayptalha/Z-Image-Turbo-Realism-LoRA')

    def test_public_download_ignores_even_an_unreadable_local_catalog(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            adapters=root/'loras'
            adapters.mkdir()
            (adapters/'catalog.json').write_text('Personal serving configuration must not be read by setup')
            payload=b'public adapter test bytes'
            fetched=root/'fetched.safetensors'
            fetched.write_bytes(payload)
            entry={'id':'realism','name':'Realism','filename':'realism.safetensors',
                   'models':[config.API_MODEL],'repository':'test-publisher/realism',
                   'revision':'pinned-test-revision','remote_filename':'weights.safetensors',
                   'sha256':hashlib.sha256(payload).hexdigest()}
            (root/'lora-catalog.json').write_text(json.dumps([entry]))
            with patch.object(lora_catalog,'ROOT',root), patch.object(lora_catalog,'LORA_DIR',adapters), \
                 patch.object(download_loras,'LORA_DIR',adapters), \
                 patch.object(download_loras,'hf_hub_download',return_value=str(fetched)) as fetch:
                download_loras.main('realism')
                fetch.assert_called_once_with('test-publisher/realism','weights.safetensors',
                    revision='pinned-test-revision',local_dir=str(adapters/'downloads'))
                self.assertEqual((adapters/'realism.safetensors').read_bytes(),payload)
                download_loras.main()
                fetch.assert_called_once()  # Reuse verified public weights.
                with self.assertRaisesRegex(ValueError,'Unknown shipped adapter'):
                    download_loras.main('personal-style')
                fetch.assert_called_once()

    def test_personal_catalog_still_applies_to_serving_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            adapters=root/'loras'
            adapters.mkdir()
            shared={'filename':'style.safetensors','models':[config.API_MODEL]}
            (root/'lora-catalog.json').write_text(json.dumps([{**shared,'id':'realism'}]))
            (adapters/'catalog.json').write_text(json.dumps([{**shared,'id':'personal-style'}]))
            with patch.object(lora_catalog,'ROOT',root), patch.object(lora_catalog,'LORA_DIR',adapters):
                self.assertEqual([entry['id'] for entry in lora_catalog.catalog()],['personal-style'])
                self.assertEqual([entry['id'] for entry in lora_catalog.catalog(local=False)],['realism'])


if __name__=='__main__':
    unittest.main()
