"""Z-Image loads LyCORIS LoKr adapters through mflux's shared adapter layers (tiny modules, no model weights)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from adapter_validation import inspect_adapter
from engine import enable_z_image_lokr


def lokr_tensors(prefix='diffusion_model.layers.0.attention.to_q', rows=(2, 4), cols=(2, 32)):
    generator = np.random.default_rng(7)
    return {f'{prefix}.lokr_w1': generator.normal(size=(rows[0], cols[0])).astype(np.float32),
            f'{prefix}.lokr_w2': generator.normal(size=(rows[1], cols[1])).astype(np.float32),
            f'{prefix}.alpha': np.array(4.0, dtype=np.float32)}


class LoKrTest(unittest.TestCase):
    def apply(self, quantize):
        import mlx.core as mx
        import mlx.nn as nn
        from mflux.models.common.lora.mapping.lora_loader import LoRALoader
        from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 8, bias=False)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = Attention()

        class Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [Block()]

        model = Transformer()
        if quantize:
            nn.quantize(model, group_size=32, bits=4)
        enable_z_image_lokr()
        enable_z_image_lokr()  # repeat model loads must not stack patterns
        tensors = lokr_tensors()
        weights = {key: mx.array(value) for key, value in tensors.items()}
        x = mx.array(np.random.default_rng(3).normal(size=(3, 64)).astype(np.float32))
        before = model.layers[0].attention.to_q(x)
        applied, matched, failed = LoRALoader._apply_lora_with_mapping(
            model, weights, 0.5, LoRALoader._build_pattern_mappings(ZImageLoRAMapping.get_mapping()),
            role=None, bake_lora=False)
        self.assertEqual((applied, matched, failed), (1, set(weights), []))
        delta = np.kron(tensors['diffusion_model.layers.0.attention.to_q.lokr_w1'],
                        tensors['diffusion_model.layers.0.attention.to_q.lokr_w2'])
        change = 0.5 * np.array(x) @ delta.T
        self.assertGreater(np.abs(change).max(), 1)  # far above the tolerance below
        # The adapter path runs in reduced precision; allow ~1% drift.
        np.testing.assert_allclose(np.array(model.layers[0].attention.to_q(x)), np.array(before) + change, rtol=2e-2, atol=2e-2)

    def test_lokr_adds_scaled_kronecker_delta_to_full_and_quantized_layers(self):
        self.apply(quantize=False)
        self.apply(quantize=True)

    def test_patch_is_idempotent_and_leaves_fused_qkv_targets_alone(self):
        from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping
        enable_z_image_lokr()
        enable_z_image_lokr()
        targets = ZImageLoRAMapping.get_mapping()
        to_q = next(target for target in targets if target.model_path == 'layers.{block}.attention.to_q' and target.up_transform is None)
        self.assertEqual(to_q.possible_lokr_w1_patterns.count('diffusion_model.layers.{block}.attention.to_q.lokr_w1'), 1)
        self.assertTrue(all(not target.possible_lokr_w1_patterns for target in targets if target.up_transform is not None))

    def test_importer_accepts_z_image_lokr_but_not_for_chroma(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'lighting.safetensors'
            save_file(lokr_tensors(), str(path), metadata={'ss_base_model_version': 'zimage'})
            self.assertEqual(inspect_adapter(path, 'z-image-turbo')['tensor_count'], 3)
            with self.assertRaisesRegex(ValueError, 'does not contain LoRA tensors'):
                inspect_adapter(path, 'chroma1-hd')


if __name__ == '__main__':
    unittest.main()
