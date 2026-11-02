"""Keep one MLX model resident; reload clean weights when adapter selection changes."""
import gc
from model_lifecycle import operation
from config import MODEL_DIR
from lora_catalog import adapter_list, effective_prompt


def enable_z_image_lokr():
    """Let Z-Image load LyCORIS LoKr adapters (such as ai-toolkit exports).

    mflux applies LoKr through its shared adapter layers but maps LoKr keys only
    for FLUX. Each unfused Z-Image target gets `<module>.lokr_w1` / `.lokr_w2`
    names beside the `<module>.alpha` names it already accepts.
    """
    from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping
    if getattr(ZImageLoRAMapping, "_imagegen_lokr", False):
        return
    original = ZImageLoRAMapping.get_mapping

    def get_mapping():
        targets = original()
        for target in targets:
            # Fused-qkv splitting is LoRA-specific; leave those targets as they are.
            if target.up_transform is not None:
                continue
            for pattern in target.possible_alpha_patterns:
                base = pattern.removesuffix(".alpha")
                for name, patterns in (("lokr_w1", target.possible_lokr_w1_patterns), ("lokr_w2", target.possible_lokr_w2_patterns)):
                    if f"{base}.{name}" not in patterns:
                        patterns.append(f"{base}.{name}")
        return targets

    ZImageLoRAMapping.get_mapping = staticmethod(get_mapping)
    ZImageLoRAMapping._imagegen_lokr = True


def load_z_image(selection):
    from mflux.models.z_image import ZImageTurbo
    enable_z_image_lokr()
    adapters = adapter_list(selection)
    return ZImageTurbo(model_path=str(MODEL_DIR), quantize=4, bake_lora=False,
        lora_paths=[adapter["file_path"] for adapter in adapters] or None,
        lora_scales=[adapter["strength"] for adapter in adapters] or None)


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
        # Any change to the stack, its order or a strength reloads clean weights.
        key = tuple((adapter["file_path"], adapter["sha256"], adapter["strength"]) for adapter in adapter_list(selection))
        if self._model is not None and self._selection_key == key:
            return
        # Release the old model before creating the replacement on this 24 GB Mac.
        previous=self._model is not None
        if previous:
            with operation('unload','weights','adapter_change'):
                self._model = None
                self._selection_key = None
                gc.collect()
                self._clear_cache()
        else:
            gc.collect()
            self._clear_cache()
        with operation('load','weights','adapter_change' if previous else 'initial_load'):
            self._model = self._factory(selection)
            self._selection_key = key

    def generate_image(self, *, lora=None, preview_dir=None, **settings):
        self._ensure_model(lora)
        self._model.tiling_config = self.tiling_config
        settings["prompt"] = effective_prompt(settings["prompt"], lora)
        if preview_dir is None:
            return self._model.generate_image(**settings)
        import live_preview
        from mflux.models.z_image.latent_creator import ZImageLatentCreator
        detach = live_preview.attach(self._model, ZImageLatentCreator, preview_dir)
        try:
            return self._model.generate_image(**settings)
        finally:
            detach()
