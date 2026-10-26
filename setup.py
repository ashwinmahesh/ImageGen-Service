"""Choose one model, then provision it through the same Makefile as scripts."""
import os
import subprocess
import sys
from pathlib import Path
from model_profiles import select_model
from model_assets import available_models

ROOT=Path(__file__).resolve().parent


def completed_models():
    try:
        return available_models(ROOT)
    except (OSError,RuntimeError,ValueError):
        # Unreadable or unavailable storage cannot establish completion. Let the
        # normal storage setup step explain or reconfigure it after selection.
        return ()


def main():
    try:
        selected=select_model(setup_complete=completed_models)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f'Setting up {selected}',flush=True)
    env={**os.environ,'IMAGE_MODEL':selected}
    return subprocess.call(['make','setup-selected'],cwd=ROOT,env=env)


if __name__=='__main__':
    sys.exit(main())
