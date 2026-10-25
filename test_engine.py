"""Adapter changes must start from clean weights rather than accumulate updates."""
import unittest
import weakref
from engine import ZImageEngine


class EngineTest(unittest.TestCase):
    def test_adapter_strength_switch_and_none_reload_clean_model(self):
        created = []
        previous = []
        calls = []
        class Fake:
            def generate_image(self, **settings):
                calls.append(settings)
                return "image"
        def factory(selection):
            if previous:
                self.assertIsNone(previous[-1](), "Old GPU model must be released first")
            model = Fake()
            previous.append(weakref.ref(model))
            created.append(selection)
            return model
        engine = ZImageEngine(factory=factory, clear_cache=lambda: None)
        selection = {"file_path": "/tmp/style.safetensors", "sha256": "abc", "strength": 0.8, "trigger_prompt": "Realism"}
        self.assertEqual(engine.generate_image(prompt="A fox", lora=selection), "image")
        self.assertEqual(calls[-1]["prompt"], "Realism, A fox")
        engine.generate_image(prompt="photorealism", lora=selection)
        self.assertEqual(calls[-1]["prompt"], "Realism, photorealism")
        self.assertEqual(len(created), 2)  # Base startup + selected adapter, reused.
        engine.generate_image(prompt="A fox", lora={**selection, "strength": 0.5})
        self.assertEqual(len(created), 3)
        engine.generate_image(prompt="A fox", lora=None)
        self.assertEqual(len(created), 4)
        self.assertIsNone(created[-1])
        self.assertEqual(calls[-1]["prompt"], "A fox")


if __name__ == "__main__":
    unittest.main()
