"""Install ImageGen's pinned packages only into the selected project environment."""
import os
from pathlib import Path
import subprocess
from python_runtime import ensure_environment

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    target = Path(os.environ.get('VENV') or PROJECT_ROOT / '.venv').expanduser()
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    python = ensure_environment(PROJECT_ROOT, target, os.environ.get('BOOTSTRAP_PYTHON'))
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', '--no-deps',
                    '-r', str(PROJECT_ROOT / 'requirements.lock.txt')], check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from None
