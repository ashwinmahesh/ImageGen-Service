"""Stacked LoRAs through the API, runtimes and gallery, and importer support for Klein/Qwen adapters."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from pydantic import ValidationError
from safetensors.numpy import save_file
import adapter_validation
import chroma_engine
import engine
import image_catalog
import lora_setup
import mflux_engine
import server
from lora_catalog import adapter_identity, effective_prompt
from model_profiles import PROFILES
from tests import test_loras


def header_file(path, key, metadata):
    header = json.dumps({"__metadata__": metadata, key: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    return path


class StackedLoRAAPITest(unittest.TestCase):
    # The single-adapter fixture, without re-running its tests here.
    tearDown = test_loras.LoRATest.tearDown
    post = test_loras.LoRATest.post

    def write_catalog(self):
        (self.folder / "catalog.json").write_text(json.dumps([self.entry, *getattr(self, "extra", [])]))

    def setUp(self):
        test_loras.LoRATest.setUp(self)
        ink = header_file(self.folder / "ink.safetensors", "diffusion_model.layers.0.attention.to_q.lora_A.weight", {"ss_base_model_version": "zimage"})
        self.extra = [{"id": "test-ink", "name": "Ink", "filename": ink.name, "models": [server.API_MODEL],
                       "default_strength": 0.6, "sha256": hashlib.sha256(ink.read_bytes()).hexdigest(),
                       "trigger_prompt": "inked", "source": "https://example.com/ink"}]
        self.write_catalog()

    def test_stack_renders_in_order_with_frozen_strengths_metadata_and_gallery(self):
        response = self.post(loras=[{"id": "test-ink"}, {"id": "test-style", "strength": 0.5}])
        self.assertEqual(response.status_code, 200, response.text)
        stack = self.fake.calls[-1]["lora"]
        self.assertEqual([(item["id"], item["strength"]) for item in stack], [("test-ink", 0.6), ("test-style", 0.5)])
        image = response.json()["data"][0]
        metadata = json.loads(Path(image["metadata_path"]).read_text())
        self.assertIsNone(metadata["lora"])
        self.assertEqual([item["id"] for item in metadata["loras"]], ["test-ink", "test-style"])
        self.assertEqual(metadata["effective_prompt"], "inked, retro, A haunted castle")
        # Catalog defaults are frozen on the job so a retry renders the same stack.
        job = self.client.get(f"/v1/jobs/{metadata['job_id']}").json()
        self.assertEqual(job["request"]["loras"], [{"id": "test-ink", "strength": 0.6}, {"id": "test-style", "strength": 0.5}])
        self.assertIsNone(job["request"]["lora_strength"])
        listed = self.client.get("/v1/images", params={"q": "Ink"}).json()["data"]
        self.assertEqual(listed[0]["loras"], [{"id": "test-ink", "name": "Ink", "strength": 0.6},
                                              {"id": "test-style", "name": "Test style", "strength": 0.5}])
        self.assertNotIn("file_path", json.dumps(listed[0]["loras"]))

    def test_one_item_stack_saves_like_a_single_lora(self):
        response = self.post(loras=[{"id": "test-style"}])
        self.assertEqual(response.status_code, 200, response.text)
        metadata = json.loads(Path(response.json()["data"][0]["metadata_path"]).read_text())
        self.assertEqual(metadata["lora"]["id"], "test-style")
        self.assertNotIn("loras", metadata)

    def test_invalid_stacks_are_rejected_before_rendering(self):
        calls = len(self.fake.calls)
        for settings in [{"lora": "test-style", "loras": [{"id": "test-ink"}]},
                         {"loras": [{"id": "test-ink"}, {"id": "test-ink"}]},
                         {"loras": [{"id": f"style-{index}"} for index in range(5)]},
                         {"loras": []},
                         {"loras": [{"id": "test-ink", "strength": 2.5}]},
                         {"loras": [{"id": "test-ink", "weight": 1}]},
                         {"loras": [{"id": "../ink"}]}]:
            with self.subTest(settings=settings):
                self.assertEqual(self.post(**settings).status_code, 422)
        self.assertNotEqual(self.post(loras=[{"id": "test-ink"}, {"id": "missing"}]).status_code, 200)
        with patch.object(server.app.state, "loras_enabled", False):
            self.assertEqual(self.post(loras=[{"id": "test-ink"}, {"id": "test-style"}]).status_code, 422)
        self.assertEqual(len(self.fake.calls), calls)
        with self.assertRaises(ValidationError):
            server.ExpandPromptRequest(prompt="A fox", loras=[{"id": "test-ink"}, {"id": "test-ink"}])

    def test_queued_stack_pins_one_checksum_in_order(self):
        self.store.pause(True)
        response = self.client.post("/v1/jobs", json={"prompt": "A castle", "output_dir": str(self.root / "images"),
                                                      "loras": [{"id": "test-style"}, {"id": "test-ink"}]})
        self.assertEqual(response.status_code, 202, response.text)
        with self.store.connection() as db:
            pinned = db.execute("SELECT adapter_sha FROM jobs WHERE id=?", (response.json()["id"],)).fetchone()[0]
        forward = adapter_identity([{"sha256": self.entry["sha256"]}, {"sha256": self.extra[0]["sha256"]}])
        backward = adapter_identity([{"sha256": self.extra[0]["sha256"]}, {"sha256": self.entry["sha256"]}])
        self.assertEqual(pinned, forward)
        self.assertNotEqual(forward, backward)


class StackedRuntimeTest(unittest.TestCase):
    stack = [{"id": "ink", "file_path": "/tmp/loras/ink.safetensors", "strength": 0.6, "sha256": "a" * 64, "trigger_prompt": "inked"},
             {"id": "style", "file_path": "/tmp/loras/style.safetensors", "strength": 0.5, "sha256": "b" * 64, "trigger_prompt": "retro"}]

    def test_identity_and_triggers(self):
        self.assertIsNone(adapter_identity(None))
        self.assertEqual(adapter_identity(self.stack[0]), "a" * 64)
        self.assertEqual(adapter_identity(self.stack[:1]), "a" * 64)
        self.assertEqual(effective_prompt("A fox", self.stack), "inked, retro, A fox")
        self.assertEqual(effective_prompt("A Retro fox", self.stack), "inked, A Retro fox")
        self.assertEqual(effective_prompt("A fox", [self.stack[0], {**self.stack[1], "trigger_prompt": "inked"}]), "inked, A fox")

    def test_mflux_edit_command_passes_each_adapter_once_then_runtime_mode(self):
        for identifier in ("flux2-klein-4b", "qwen-image-2.1-turbo"):
            with self.subTest(model=identifier), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); commands = []
                def run(command, **kwargs):
                    commands.append(command)
                    self.assertEqual(Path(command[command.index("--prompt-file") + 1]).read_text(), "inked, retro, Make it night.")
                    Image.new("RGB", (320, 256), "green").save(command[command.index("--output") + 1])
                    return subprocess.CompletedProcess(command, 0)
                with patch.object(mflux_engine, "ROOT", root), patch.object(mflux_engine, "MODEL_DIR", root / "weights"), \
                     patch.object(mflux_engine, "PROFILE", PROFILES[identifier]), patch.object(mflux_engine.subprocess, "run", side_effect=run):
                    mflux_engine.MfluxResult(dict(prompt="Make it night.", width=320, height=256, seed=7, num_inference_steps=4,
                                                  image_path="reference.png", lora=self.stack)).save(root / "out.png")
                command = commands[0]
                groups = [command[index + 1:index + 3] for index, value in enumerate(command) if value == "--lora"]
                self.assertEqual(groups, [["/tmp/loras/ink.safetensors", "0.6"], ["/tmp/loras/style.safetensors", "0.5"]])
                self.assertEqual(command.count("--no-bake-lora"), 1)

    def test_chroma_tags_each_adapter_and_needs_one_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); commands = []
            def run(command, **kwargs):
                commands.append((command, Path(command[command.index("--prompt-file") + 1]).read_text()))
                Image.new("RGB", (256, 256), "green").save(command[command.index("--output") + 1])
                return subprocess.CompletedProcess(command, 0)
            with patch.object(chroma_engine, "ROOT", root), patch.object(chroma_engine, "PROFILE", PROFILES["chroma1-hd"]), \
                 patch.object(chroma_engine.subprocess, "run", side_effect=run):
                settings = dict(prompt="A fox", seed=1, width=256, height=256, num_inference_steps=40)
                chroma_engine.ChromaResult(root / "sd-cli", {**settings, "lora": self.stack}).save(root / "out.png")
                split = [self.stack[0], {**self.stack[1], "file_path": "/elsewhere/style.safetensors"}]
                with self.assertRaisesRegex(ValueError, "same LoRA folder"):
                    chroma_engine.ChromaResult(root / "sd-cli", {**settings, "lora": split}).save(root / "split.png")
            command, prompt = commands[0]
            self.assertEqual(prompt, "inked, retro, A fox <lora:ink:0.6> <lora:style:0.5>")
            self.assertEqual(command[command.index("--lora-model-dir") + 1], "/tmp/loras")

    def test_z_image_reloads_when_the_stack_order_or_strength_changes(self):
        loads = []
        class Model:
            tiling_config = None
            def generate_image(self, **settings): return settings
        def factory(selection):
            loads.append([adapter["id"] for adapter in selection or []]); return Model()
        runtime = engine.ZImageEngine(factory=factory, clear_cache=lambda: None)
        runtime.generate_image(lora=self.stack, prompt="A fox")
        runtime.generate_image(lora=list(self.stack), prompt="A fox")
        runtime.generate_image(lora=self.stack[::-1], prompt="A fox")
        runtime.generate_image(lora=[self.stack[1], {**self.stack[0], "strength": 0.7}], prompt="A fox")
        self.assertEqual(loads, [[], ["ink", "style"], ["style", "ink"], ["style", "ink"]])

    def test_gallery_record_keeps_only_public_stack_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); images = root / "images"; images.mkdir()
            identifier = "c" * 32
            Image.new("RGB", (64, 64), "green").save(images / f"{identifier}.png")
            (images / f"{identifier}.json").write_text(json.dumps({"prompt": "A fox", "lora": None, "loras": self.stack + ["junk"]}))
            record = image_catalog.describe(images / f"{identifier}.png", image_catalog.read_metadata(images / f"{identifier}.json"))
            self.assertEqual(record["loras"], [{"id": "ink", "name": "ink", "strength": 0.6}, {"id": "style", "name": "style", "strength": 0.5}])
            self.assertIsNone(record["lora"])


def adapter(path, tensors, metadata=None):
    save_file({key: np.zeros(shape, np.float32) for key, shape in tensors.items()}, str(path), metadata=metadata or {})
    return path


class EditLoRAImportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def check(self, model, tensors, metadata=None):
        return adapter_validation.inspect_adapter(adapter(self.root / "adapter.safetensors", tensors, metadata), model)

    def test_klein_accepts_bfl_kohya_diffusers_and_lokr_of_the_right_size(self):
        kohya = {"lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": (8, 3072), "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": (9216, 8)}
        diffusers = {"transformer.transformer_blocks.0.attn.to_q.lora_A.weight": (8, 4096), "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": (4096, 8)}
        lokr = {"diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1": (8, 8), "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2": (1152, 384)}
        self.check("flux2-klein-4b", kohya)
        self.check("flux2-klein-4b", lokr)
        self.check("flux2-klein-9b", diffusers, {"modelspec.architecture": "flux-2-klein-9b/lora"})
        with self.assertRaisesRegex(ValueError, "Wrong model size.*4096-wide.*FLUX.2 Klein 4B is 3072"):
            self.check("flux2-klein-4b", diffusers)
        with self.assertRaisesRegex(ValueError, "Wrong model family"):
            self.check("flux2-klein-4b", {"lora_unet_double_blocks_0_img_mod_lin.lora_down.weight": (8, 3072)}, {"ss_base_model_version": "flux1"})
        with self.assertRaisesRegex(ValueError, "Wrong model family"):
            self.check("flux2-klein-9b", {"transformer_blocks.0.img_mlp.gate_layer.lora_A.default.weight": (8, 4096)})

    def test_qwen_21_accepts_peft_and_rejects_original_qwen_and_other_formats(self):
        peft = {"transformer_blocks.0.attn.to_q.lora_A.default.weight": (16, 4096), "transformer_blocks.0.attn.to_q.lora_B.default.weight": (4096, 16)}
        self.check("qwen-image-2.1-turbo", peft)
        self.check("qwen-image-2.1-turbo", {"diffusion_model.transformer_blocks.0.img_mlp.out.lora_A.weight": (16, 12288)})
        with self.assertRaisesRegex(ValueError, "original Qwen-Image"):
            self.check("qwen-image-2.1-turbo", {"transformer_blocks.0.attn.add_q_proj.lora_A.weight": (16, 3072)})
        with self.assertRaisesRegex(ValueError, "lora_A/lora_B"):
            self.check("qwen-image-2.1-turbo", {"lora_unet_transformer_blocks.0.attn.to_q.lora_down.weight": (16, 4096)})
        with self.assertRaisesRegex(ValueError, "Wrong model family"):
            self.check("qwen-image-2.1-turbo", peft, {"ss_base_model_version": "flux2_klein"})
        with self.assertRaisesRegex(ValueError, "Unrecognized Qwen-Image 2.1"):
            self.check("qwen-image-2.1-turbo", {"single_transformer_blocks.0.attn.to_q.lora_A.weight": (16, 4096)})
        with self.assertRaisesRegex(ValueError, "Wrong model size"):
            self.check("qwen-image-2.1-turbo", {"transformer.transformer_blocks.0.attn.to_q.lora_A.weight": (16, 3072)})

    def test_dora_is_rejected_for_mflux_models(self):
        with self.assertRaisesRegex(ValueError, "DoRA"):
            self.check("qwen-image-2.1-turbo", {"transformer_blocks.0.attn.to_q.lora_A.default.weight": (16, 4096),
                                               "transformer_blocks.0.attn.to_q.lora_magnitude_vector.default.weight": (4096,)})

    def test_editor_adapters_install_into_their_model_folder_and_are_listed(self):
        project = self.root / "project"; project.mkdir(); store = self.root / "store"
        (project / "lora-catalog.json").write_text("[]")
        source = adapter(self.root / "Night.safetensors", {"transformer_blocks.0.attn.to_q.lora_A.default.weight": (16, 4096),
                                                            "transformer_blocks.0.attn.to_q.lora_B.default.weight": (4096, 16)})
        with patch.dict(os.environ, {"DATA_DIR": str(store)}, clear=True):
            entry, directory, changed = lora_setup.register(project, "qwen-image-2.1-turbo", source, identifier="night", name="Night")
            self.assertTrue(changed)
            self.assertEqual(directory, (store / "qwen-image-2.1-turbo" / "loras").resolve())
            listed = server.model_loras(project, "qwen-image-2.1-turbo")[1]
        self.assertEqual([(item["id"], item["available"]) for item in listed], [(entry["id"], True)])


if __name__ == "__main__":
    unittest.main()
