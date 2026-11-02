"""Civitai sources for make setup-lora; Civitai is mocked, nothing touches the network."""
from contextlib import redirect_stdout, redirect_stderr
import email.message
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch
import numpy as np
from safetensors.numpy import save_file
import lora_setup


def qwen_adapter(path):
    save_file({"transformer_blocks.0.attn.to_q.lora_A.default.weight": np.zeros((4, 4096), np.float32),
               "transformer_blocks.0.attn.to_q.lora_B.default.weight": np.zeros((4096, 4), np.float32)}, str(path))
    return path


class Response(io.BytesIO):
    def __init__(self, data, content_type="application/octet-stream"):
        super().__init__(data)
        self.headers = email.message.Message()
        self.headers["Content-Type"] = content_type


def version(identifier, *, base="Qwen 2.1", sha256="0" * 64, words=()):
    return {"id": identifier, "modelId": 77, "name": f"v{identifier}", "status": "Published", "baseModel": base, "trainedWords": list(words),
            "files": [{"id": 1, "name": "workflow.json", "type": "Config"},
                      {"id": 2, "name": "restore.safetensors", "type": "Model", "hashes": {"SHA256": sha256.upper()},
                       "downloadUrl": f"https://civitai.com/api/download/models/{identifier}?fileId=2"}]}


class CivitaiSourceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.project = self.root / "project"; self.project.mkdir(); (self.project / "lora-catalog.json").write_text("[]")
        self.store = self.root / "store"
        self.environment = patch.dict(os.environ, {"DATA_DIR": str(self.store)}, clear=True); self.environment.start()
        self.weights = qwen_adapter(self.root / "restore.safetensors").read_bytes()
        self.digest = hashlib.sha256(self.weights).hexdigest()

    def tearDown(self):
        self.environment.stop(); self.temp.cleanup()

    def api(self, *, kind="LORA", versions=None):
        versions = versions or [version(12, sha256=self.digest, words=["restored photo"]), version(11, sha256=self.digest)]
        pages = {"models/77": {"id": 77, "name": "Qwen Restore", "type": kind, "modelVersions": versions},
                 **{f"model-versions/{item['id']}": {**item, "model": {"name": "Qwen Restore", "type": kind}} for item in versions}}
        return lambda path: pages[path]

    def run_cli(self, args):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            code = lora_setup.main(["--noninteractive", "--model", "qwen-image-2.1-turbo", *args], project=self.project)
        return code, output.getvalue()

    def test_page_and_download_urls_parse_and_credentials_are_refused(self):
        self.assertEqual(lora_setup.parse_source("https://civitai.com/models/77/qwen-restore?modelVersionId=12"),
                         {"kind": "civitai", "model_id": 77, "version_id": 12, "file_id": None})
        self.assertEqual(lora_setup.parse_source("https://civitai.com/api/download/models/12?type=Model&fileId=2"),
                         {"kind": "civitai", "model_id": None, "version_id": 12, "file_id": 2})
        for value in ["https://civitai.com/models/77?token=secret", "http://civitai.com/models/77", "https://user:pw@civitai.com/models/77",
                      "https://civitai.com/images/5", "https://civitai.com/models/77?sort=new"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                lora_setup.parse_source(value)

    def test_api_key_goes_only_to_civitai_and_redirects_are_followed_by_hand(self):
        seen = []
        class Opener:
            def open(self, request, timeout):
                seen.append((request.full_url, request.get_header("Authorization")))
                if len(seen) == 1:
                    headers = email.message.Message(); headers["Location"] = "https://cdn.example.net/file?signed=1"
                    raise urllib.error.HTTPError(request.full_url, 307, "redirect", headers, None)
                return Response(b"weights")
        with patch.object(lora_setup.urllib.request, "build_opener", return_value=Opener()):
            self.assertEqual(lora_setup.civitai_request("https://civitai.com/api/download/models/12", token="key").read(), b"weights")
        self.assertEqual(seen, [("https://civitai.com/api/download/models/12", "Bearer key"), ("https://cdn.example.net/file?signed=1", None)])
        class Denied:
            def open(self, request, timeout):
                raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", email.message.Message(), None)
        with patch.object(lora_setup.urllib.request, "build_opener", return_value=Denied()), self.assertRaisesRegex(RuntimeError, "CIVITAI_TOKEN"):
            lora_setup.civitai_request("https://civitai.com/api/download/models/12")

    def test_newest_version_installs_with_civitai_name_trigger_checksum_and_provenance(self):
        with patch.object(lora_setup, "civitai_json", side_effect=self.api()), \
             patch.object(lora_setup, "civitai_request", side_effect=lambda url, **kwargs: Response(self.weights)) as fetch:
            code, output = self.run_cli(["--source", "https://civitai.com/models/77/qwen-restore", "--id", "restore", "--strength", "1"])
        self.assertEqual(code, 0, output)
        self.assertEqual(fetch.call_args.args[0], "https://civitai.com/api/download/models/12?fileId=2")
        entry = json.loads((self.store / "qwen-image-2.1-turbo/loras/custom.json").read_text())[0]
        self.assertEqual((entry["id"], entry["name"], entry["trigger_prompt"], entry["default_strength"]), ("custom-restore", "Qwen Restore", "restored photo", 1.0))
        self.assertEqual((entry["sha256"], entry["repository"], entry["revision"], entry["remote_filename"]), (self.digest, "civitai:77", "12", "restore.safetensors"))
        self.assertEqual(entry["source"], "https://civitai.com/models/77?modelVersionId=12")
        # A second run reuses the verified download.
        with patch.object(lora_setup, "civitai_json", side_effect=self.api()), patch.object(lora_setup, "civitai_request") as fetch:
            code, output = self.run_cli(["--source", "https://civitai.com/models/77", "--id", "restore", "--strength", "1"])
        self.assertEqual(code, 0, output); fetch.assert_not_called()

    def test_pinned_version_wrong_type_base_model_and_checksum_install_nothing(self):
        with patch.object(lora_setup, "civitai_json", side_effect=self.api()), \
             patch.object(lora_setup, "civitai_request", side_effect=lambda url, **kwargs: Response(self.weights)) as fetch:
            self.assertEqual(self.run_cli(["--source", "https://civitai.com/models/77", "--revision", "11", "--id", "old"])[0], 0)
        self.assertEqual(fetch.call_args.args[0], "https://civitai.com/api/download/models/11?fileId=2")
        before = (self.store / "qwen-image-2.1-turbo/loras/custom.json").read_bytes()
        cases = [(self.api(kind="Workflows"), "is a Workflows, not a LoRA"),
                 (self.api(versions=[version(13, base="ZImageTurbo", sha256=self.digest)]), "is for ZImageTurbo"),
                 (self.api(versions=[version(14, sha256="f" * 64)]), "does not match its published SHA256")]
        for pages, message in cases:
            with self.subTest(message=message), patch.object(lora_setup, "civitai_json", side_effect=pages), \
                 patch.object(lora_setup, "civitai_request", side_effect=lambda url, **kwargs: Response(self.weights)):
                code, output = self.run_cli(["--source", "https://civitai.com/models/77", "--id", "other"])
                self.assertEqual(code, 1); self.assertIn(message, output)
        self.assertEqual((self.store / "qwen-image-2.1-turbo/loras/custom.json").read_bytes(), before)
        self.assertEqual(list((self.store / "qwen-image-2.1-turbo/loras/downloads/civitai-77/14").glob("*")), [])


if __name__ == "__main__":
    unittest.main()
