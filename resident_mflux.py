"""Run the pinned mflux CLIs in-process so their weights can stay loaded between images.

Each adapter mirrors its CLI's main() for mflux 0.23: the CLI's own parser reads the
same argument list the subprocess path uses, the model is built once, and
generate_image() runs per image. Only used when Keep weights loaded is on.
"""
import importlib
from pathlib import Path
import sys

QWEN_EDIT = 'mflux.models.qwen21.cli.qwen21_edit_generate'
FLUX2_TEXT = 'mflux.models.flux2.cli.flux2_generate'
FLUX2_EDIT = 'mflux.models.flux2.cli.flux2_edit_generate'
SUPPORTED = (QWEN_EDIT, FLUX2_TEXT, FLUX2_EDIT)


def parse(module, arguments):
    """Parse CLI arguments with that CLI's parser, which also applies its global flags (for example --no-exif)."""
    saved = sys.argv
    try:
        sys.argv = [module, *arguments]
        return importlib.import_module(module).build_parser().parse_args()
    finally:
        sys.argv = saved


def key(module, args):
    """What the loaded weights depend on; per-image settings are excluded."""
    return (module, args.model, args.base_model, args.quantize, args.compute_precision,
            tuple(args.lora_paths or ()), tuple(args.lora_scales or ()), args.bake_lora)


def build(module, args):
    """Load the model as the CLI does, keeping every component for the next image."""
    from mflux.callbacks.callback_manager import CallbackManager
    from mflux.callbacks.instances.memory_saver import MemorySaver
    from mflux.cli.parser.parsers import lora_init_kwargs_from_args
    from mflux.models.common.compute_precision import ComputePrecision
    from mflux.models.common.resolution.config_resolution import ConfigResolution
    cli = importlib.import_module(module)
    common = dict(quantize=args.quantize, model_path=args.model_path,
                  compute_precision=ComputePrecision.dtype_for(args.compute_precision), **lora_init_kwargs_from_args(args))
    if module == QWEN_EDIT:
        config = ConfigResolution.resolve_restricted(args.model, 'qwen-image-2.1', model_path=args.model_path,
                                                     extra_keys=cli.FAMILY_MODELS, base_model=args.base_model)
        model, latent_creator = cli.QwenImage21Edit(model_config=config, **common), cli.QwenImage21LatentCreator
    else:
        config = ConfigResolution.resolve_restricted(args.model, cli.DEFAULT_MODEL, model_path=args.model_path,
                                                     extra_keys=cli.FAMILY_MODELS, base_model=args.base_model)
        variant = cli.Flux2KleinEdit if module == FLUX2_EDIT else cli.Flux2Klein
        model, latent_creator = variant(model_config=config, **common), cli.Flux2LatentCreator
    CallbackManager._apply_vae_tiling(args, model)
    # The CLIs discard text encoders after encoding and, in low-RAM mode, the transformer after
    # sampling; keep both so the next image skips loading. The 1 GB MLX cache limit stays.
    model.callbacks.register(MemorySaver(model=model, keep_transformer=True, cache_limit_bytes=1000**3,
                                         num_seeds=1, keep_text_encoder=True))
    return model, latent_creator


def generate(module, model, args):
    """One image with the same settings the CLI passes."""
    from mflux.utils.dimension_resolver import DimensionResolver
    from mflux.utils.prompt_util import PromptUtil
    from mflux.utils.scale_factor import ScaleFactor
    seed, prompt = args.seed[0], PromptUtil.read_prompt(args)
    if module == QWEN_EDIT:
        paths = args.image_paths or []
        width, height = args.width, args.height
        if isinstance(width, ScaleFactor) or isinstance(height, ScaleFactor):
            width, height = DimensionResolver.resolve(width=width if width is not None else ScaleFactor(1),
                                                      height=height if height is not None else ScaleFactor(1),
                                                      reference_image_path=paths[-1] if paths else None)
        return model.generate_image(
            seed=seed, prompt=prompt, negative_prompt=PromptUtil.read_negative_prompt(args), width=width, height=height,
            num_inference_steps=args.steps, guidance=args.guidance if args.guidance is not None else 1.0,
            image_paths=paths, output_resolution=args.output_resolution, use_kv_cache=args.use_kv_cache,
            mask_image=args.mask_image, auto_mask=args.auto_mask, strength=args.strength,
            use_step_cache=args.use_step_cache, step_cache_threshold=args.step_cache_threshold,
            enhance_prompt=args.enhance_prompt, verify=args.verify, verify_retries=args.verify_retries,
            scheduler=args.scheduler)
    guidance = args.guidance if args.guidance is not None else 1.0
    if module == FLUX2_EDIT:
        paths = [Path(path) for path in args.image_paths]
        width, height = DimensionResolver.resolve(width=args.width, height=args.height,
                                                  reference_image_path=paths[0] if paths else None)
        return model.generate_image(seed=seed, prompt=prompt, width=width, height=height, guidance=guidance,
                                    image_paths=paths, num_inference_steps=args.steps, scheduler='flow_match_euler_discrete')
    width, height = DimensionResolver.resolve(width=args.width, height=args.height, reference_image_path=args.image_path)
    return model.generate_image(seed=seed, prompt=prompt, width=width, height=height, guidance=guidance,
                                image_path=args.image_path, num_inference_steps=args.steps,
                                image_strength=args.image_strength, scheduler='flow_match_euler_discrete',
                                pid_decode=args.pid_decode, pid_degrade_sigma=args.pid_degrade_sigma)
