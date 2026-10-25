"""Exercise adapter selection and reset without loading a GPU model."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import sys
import types
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import install_lora
import lora_catalog
import server
from test_api import FakeModel


class LoRATest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.folder = self.root / "loras"
        self.folder.mkdir()
        self.file = self.folder / "style.safetensors"
        header = json.dumps({"__metadata__": {"ss_base_model_version": "zimage"},
            "diffusion_model.layers.0.attention.to_k.lora_A.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        self.file.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
        self.entry = {"id": "test-style", "name": "Test style", "filename": self.file.name,
            "models": [server.API_MODEL], "default_strength": 0.8,
            "sha256": hashlib.sha256(self.file.read_bytes()).hexdigest(),
            "trigger_prompt": "retro", "source": "https://example.com/style"}
        self.write_catalog()
        self.fake = FakeModel()
        self.patches = [patch.dict(sys.modules, {"mlx": types.ModuleType("mlx"), "mlx.core": types.SimpleNamespace(clear_cache=lambda: None)}),
            patch.object(lora_catalog, "LORA_DIR", self.folder),
            patch.object(install_lora, "LORA_DIR", self.folder),
            patch.object(server, "ROOT", self.root), patch.object(server, "model", self.fake)]
        for item in self.patches:
            item.start()
        self.client = TestClient(server.app, base_url="http://127.0.0.1")

    def write_catalog(self):
        (self.folder / "catalog.json").write_text(json.dumps([self.entry]))

    def tearDown(self):
        self.client.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def post(self, **settings):
        return self.client.post("/v1/images/generations", json={"prompt": "A haunted castle", "size": "256x256",
            "output_dir": str(self.root / "images"), "response_format": "url", **settings})

    def test_catalog_selection_default_strength_and_reset(self):
        listed = self.client.get("/v1/loras").json()["data"]
        self.assertTrue(listed[0]["available"])
        response = self.post(lora="test-style")
        self.assertEqual(response.status_code, 200, response.text)
        selection = self.fake.calls[-1]["lora"]
        self.assertEqual(selection["strength"], 0.8)
        self.assertEqual(selection["file_path"], str(self.file))
        metadata = json.loads(Path(response.json()["data"][0]["metadata_path"]).read_text())
        self.assertEqual(metadata["lora"]["sha256"], self.entry["sha256"])
        self.assertEqual(metadata["effective_prompt"], "retro, A haunted castle")
        self.assertEqual(self.post(lora="test-style", lora_strength=0.5).status_code, 200)
        self.assertEqual(self.fake.calls[-1]["lora"]["strength"], 0.5)
        self.assertEqual(self.post(lora=None).status_code, 200)
        self.assertIsNone(self.fake.calls[-1]["lora"])

    def test_missing_unknown_and_changed_weights_fail_before_inference(self):
        self.assertEqual(self.post(lora="missing").status_code, 400)
        self.file.unlink()
        self.assertFalse(self.client.get("/v1/loras").json()["data"][0]["available"])
        self.assertEqual(self.post(lora="test-style").status_code, 400)
        self.file.write_bytes(b"changed")
        self.assertEqual(self.post(lora="test-style").status_code, 400)
        self.assertEqual(self.fake.calls, [])

    def test_invalid_selection_fields_and_prompt_tags(self):
        for settings in [{"lora_strength": 1}, {"lora": "../style"},
                         {"lora": "test-style", "lora_strength": -0.1},
                         {"lora": "test-style", "lora_strength": 2.1},
                         {"prompt": "a castle <LoRA:other:1>"}]:
            with self.subTest(settings=settings):
                self.assertEqual(self.post(**settings).status_code, 422)

    def test_catalog_cannot_escape_adapter_directory_or_model_family(self):
        self.entry["filename"] = "../style.safetensors"
        self.write_catalog()
        with self.assertRaises(ValueError):
            lora_catalog.list_loras()
        self.entry["filename"] = self.file.name
        self.entry["models"] = ["unrelated-model"]
        self.write_catalog()
        self.assertEqual(lora_catalog.list_loras(), [])

    def test_import_preserves_original_and_records_checksum(self):
        source = self.root / "ZImage-test.safetensors"
        original = self.file.read_bytes()
        source.write_bytes(original)
        self.file.unlink()
        install_lora.install(source, "test-style")
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(self.file.read_bytes(), original)
        self.assertEqual(lora_catalog.resolve_lora("test-style")["sha256"], hashlib.sha256(original).hexdigest())


if __name__ == "__main__":
    unittest.main()
