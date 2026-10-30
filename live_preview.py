"""Opt-in in-progress previews: keep only the newest denoising step as preview.png.

mflux can decode the latents after every step. Its stock handler also saves the starting
noise and rebuilds a strip of every step each time; previews only need the newest frame.
Run as a script to wrap an mflux CLI module: python live_preview.py MODULE [CLI args...].
"""
import os
from pathlib import Path
import re
import runpy
import sys

PREVIEW_NAME = 'preview.png'
IDENTIFIER = re.compile(r'[0-9a-f]{32}')


def preview_folder(project, job_id):
    """Project-local folder holding one running job's newest preview."""
    if not IDENTIFIER.fullmatch(job_id):
        raise ValueError('Unknown job ID')
    return Path(project) / 'state' / 'previews' / job_id


def preview_updated(project, job):
    """Unix time of this attempt's newest preview, or None."""
    if not job['request'].get('live_preview'):
        return None
    try:
        updated = (preview_folder(project, job['id']) / PREVIEW_NAME).stat().st_mtime
    except (OSError, ValueError):
        return None
    # A file left by an earlier attempt is not this attempt's preview.
    return updated if updated >= (job.get('started') or 0) - 1 else None


def keep_latest(folder, seed):
    """Replace preview.png with the newest step image and drop older step files."""
    folder = Path(folder)
    steps = sorted(folder.glob(f'seed_{seed}_step*.png'), key=lambda path: path.stat().st_mtime)
    if steps:
        os.replace(steps[-1], folder / PREVIEW_NAME)
        for older in steps[:-1]:
            older.unlink(missing_ok=True)


def handler_class():
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler

    class LatestPreview(StepwiseHandler):
        """mflux's stepwise images without the noise frame or composite strip."""
        def call_before_loop(self, *args, **kwargs):
            pass

        def _save_composite(self, seed):
            self.step_wise_images.clear()
            keep_latest(self.output_dir, seed)

    return LatestPreview


def attach(model, latent_creator, folder):
    """Write previews from an in-process mflux model; returns a function that stops them."""
    handler = handler_class()(model=model, output_dir=str(folder), latent_creator=latent_creator)
    model.callbacks.register(handler)

    def detach():
        for name in ('before_loop', 'in_loop', 'after_loop', 'interrupt'):
            hooks = getattr(model.callbacks, name, None)
            if isinstance(hooks, list) and handler in hooks:
                hooks.remove(handler)
    return detach


def main():
    """Run an mflux CLI module whose --stepwise-image-output-dir keeps only the newest preview."""
    from mflux.callbacks.callback_manager import CallbackManager
    latest = handler_class()

    def register(args, model, latent_creator):
        if args.stepwise_image_output_dir:
            model.callbacks.register(latest(model=model, output_dir=args.stepwise_image_output_dir, latent_creator=latent_creator))
    CallbackManager._register_stepwise_handler = staticmethod(register)
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
