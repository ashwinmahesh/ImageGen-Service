"""Choose one model, then provision it through the same Makefile as scripts."""
import os
import subprocess
import sys
from pathlib import Path
from model_profiles import select_model


def main():
    try:
        selected=select_model()
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f'Setting up {selected}',flush=True)
    env={**os.environ,'IMAGE_MODEL':selected}
    return subprocess.call(['make','setup-selected'],cwd=Path(__file__).resolve().parent,env=env)


if __name__=='__main__':
    sys.exit(main())
