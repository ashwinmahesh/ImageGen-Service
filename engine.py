"""Keep one MLX model resident; reload clean weights when adapter selection changes."""
import gc
from config import MODEL_DIR
from lora_catalog import effective_prompt


def load_z_image(selection):
    from mflux.models.z_image import ZImageTurbo
    return ZImageTurbo(model_path=str(MODEL_DIR), quantize=4, bake_lora=False,
        lora_paths=[selection["file_path"]] if selection else None,
        lora_scales=[selection["strength"]] if selection else None)


def release_cache():
    import mlx.core as mx
    mx.clear_cache()


class ZImageEngine:
    def __init__(self, factory=load_z_image, clear_cache=release_cache):
        self._factory = factory
        self._clear_cache = clear_cache
        self._model = None
        self._selection_key = None
        self.tiling_config = None
        self._ensure_model(None)

    def _ensure_model(self, selection):
        key = (selection["file_path"], selection["sha256"], selection["strength"]) if selection else ()
        if self._model is not None and self._selection_key == key:
            return
        # Release the old model before creating the replacement on this 24 GB Mac.
        self._model = None
        self._selection_key = None
        gc.collect()
        self._clear_cache()
        self._model = self._factory(selection)
        self._selection_key = key

    def generate_image(self, *, lora=None, **settings):
        self._ensure_model(lora)
        self._model.tiling_config = self.tiling_config
        settings["prompt"] = effective_prompt(settings["prompt"], lora)
        return self._model.generate_image(**settings)
