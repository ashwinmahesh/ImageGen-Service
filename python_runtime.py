"""Choose a compatible local CPython without loading models or changing system Python."""
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

# All 66 pinned distributions resolve as binary packages on macOS 14 arm64 for
# 3.12, 3.13 and 3.14. NetworkX 3.7 explicitly excludes the 3.14.1 patch.
# Validate the complete lock again before adding a new Python minor.
PYTHON_MINORS = (12, 13, 14)
PREFERRED_PYTHON = '3.14.9'  # Keep the Python-free shell bootstrap fallback in sync.
EXCLUDED_VERSIONS = {(3, 14, 1)}


def compatible(info):
    if not isinstance(info, dict):
        return False
    version = info.get('version', [])
    return (isinstance(version, (list, tuple)) and len(version) == 3 and version[0] == 3 and version[1] in PYTHON_MINORS
            and tuple(version) not in EXCLUDED_VERSIONS
            and info.get('implementation') == 'cpython' and info.get('system') == 'Darwin'
            and info.get('machine') == 'arm64' and not info.get('free_threaded')
            and info.get('releaselevel', 'final') == 'final')


def requirement():
    return ('ImageGen needs standard CPython 3.12, 3.13 or 3.14 on Apple Silicon macOS '
            '(excluding 3.14.1). The complete lock needs Python 3.12+; NetworkX excludes 3.14.1. '
            'Other Python minors and free-threaded builds have not been validated.')


def probe_python(candidate):
    path = shutil.which(str(candidate))
    if not path:
        return None
    probe = '''import json,platform,sys,sysconfig
print(json.dumps(dict(version=list(sys.version_info[:3]), implementation=sys.implementation.name,
 system=platform.system(), machine=platform.machine(), free_threaded=bool(sysconfig.get_config_var('Py_GIL_DISABLED')),
 releaselevel=sys.version_info.releaselevel, prefix=sys.prefix, base_prefix=sys.base_prefix,
 base=getattr(sys,'_base_executable',sys.executable))))'''
    try:
        result = subprocess.run([path, '-I', '-c', probe], capture_output=True, text=True, timeout=5)
        if result.returncode:
            return None
        info = json.loads(result.stdout)
        if not isinstance(info, dict) or not isinstance(info.get('base'), str):
            return None
        info['path'] = str(Path(path).absolute())
        return info
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def installed_pythons(root):
    yield sys.executable
    yield 'python3'
    for minor in PYTHON_MINORS[::-1]:
        yield f'python3.{minor}'
        yield f'/opt/homebrew/opt/python@3.{minor}/bin/python3.{minor}'
        yield f'/Library/Frameworks/Python.framework/Versions/3.{minor}/bin/python3'
    yield from sorted(root.glob('.python/*/bin/python3'))
    yield from sorted(root.glob('.venv*/bin/python'))


def managed_python(root):
    for name in ('.python', '.bootstrap'):
        if not (root / name).resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f'Keep {name} inside the serving project; automatic setup will not write outside it.')
    uv = subprocess.check_output(['sh', str(root / 'bootstrap_uv.sh')], text=True).strip()
    env = {**os.environ, 'UV_PYTHON_INSTALL_DIR': str(root / '.python'),
           'UV_PYTHON_BIN_DIR': str(root / '.bootstrap/bin'), 'UV_CACHE_DIR': str(root / '.bootstrap/cache'),
           'UV_PYTHON_INSTALL_BIN': '0', 'UV_NO_MODIFY_PATH': '1'}
    print(f'No compatible installed Python; installing Python {PREFERRED_PYTHON} inside {root / ".python"}.', file=sys.stderr)
    subprocess.run([uv, '--no-config', 'python', 'install', PREFERRED_PYTHON], env=env, check=True)
    python = subprocess.check_output([uv, '--no-config', 'python', 'find', '--managed-python',
                                     '--no-python-downloads', PREFERRED_PYTHON], env=env, text=True).strip()
    if not Path(python).resolve().is_relative_to((root / '.python').resolve()):
        raise RuntimeError('Managed Python must stay inside the serving project.')
    return python


def valid_environment(info, target):
    return (compatible(info) and isinstance(info.get('prefix'), str)
            and info['prefix'] != info.get('base_prefix')
            and Path(info['prefix']).resolve() == target.resolve())


def ensure_environment(root, target, explicit=None):
    root, target = Path(root).resolve(), Path(target).expanduser().absolute()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('The ImageGen dependency lock targets Apple Silicon macOS. Use a native arm64 terminal.')
    chosen = None
    if explicit:
        chosen = probe_python(explicit)
        if not compatible(chosen):
            raise RuntimeError(f'Explicit interpreter {explicit!r} is incompatible. {requirement()}')
    python = target / 'bin/python'
    if target.exists() or target.is_symlink():
        existing = probe_python(python)
        if not valid_environment(existing, target):
            raise RuntimeError(f'Existing environment {target} is incompatible, broken or not a virtual environment. '
                               f'{requirement()} It was preserved; choose a new VENV or move it aside before rebuilding.')
        if chosen and Path(chosen['base']).resolve() != Path(existing['base']).resolve():
            raise RuntimeError(f'{target} uses a different Python from the explicit interpreter. '
                               'It was preserved; choose a new VENV or omit BOOTSTRAP_PYTHON.')
        print(f'Reusing Python {".".join(map(str, existing["version"]))} in {target}.', file=sys.stderr)
        return python
    if chosen is None:
        seen = set()
        for candidate in installed_pythons(root):
            info = probe_python(candidate)
            if not info or info['base'] in seen:
                continue
            seen.add(info['base'])
            if compatible(info):
                chosen = info
                break
    if chosen is None:
        chosen = probe_python(managed_python(root))
        if not compatible(chosen):
            raise RuntimeError('Downloaded interpreter failed validation. ' + requirement())
    print(f'Using Python {".".join(map(str, chosen["version"]))} for {target}.', file=sys.stderr)
    subprocess.run([chosen['base'], '-m', 'venv', '--copies', str(target)], check=True)
    if not valid_environment(probe_python(python), target):
        raise RuntimeError(f'New environment {target} failed validation; it was preserved for inspection.')
    return python
