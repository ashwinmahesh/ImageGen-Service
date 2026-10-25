"""Project-local defaults with an optional persisted external data directory."""
import argparse
import json
import os
import sys
from pathlib import Path
import tempfile
from model_profiles import PROFILES

DATA_FOLDERS = tuple(PROFILES)
MODEL_FOLDERS = ('weights', 'cache', 'images', 'references', 'image-index', 'loras', 'logs')


def _absolute(value):
    path = Path(value).expanduser()
    if not str(value).strip() or not path.is_absolute():
        raise ValueError('DATA_DIR must be an absolute directory')
    return path.resolve()


def data_directory(project, *, use_environment=True):
    project = Path(project).resolve()
    if use_environment and 'DATA_DIR' in os.environ:
        return _absolute(os.environ['DATA_DIR'])
    settings = project / 'storage.json'
    if settings.is_file():
        try:
            return _absolute(json.loads(settings.read_text())['data_dir'])
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError('Invalid storage.json; choose storage again with make storage DATA_DIR=/absolute/path') from error
    # Reuse an existing model link on this Mac without requiring a manifest.
    weights = project / DATA_FOLDERS[0]
    return weights.resolve().parent if weights.is_symlink() else project


def require_storage(path):
    path = Path(path).expanduser().absolute()
    parts = path.parts
    if len(parts) >= 3 and parts[1] == 'Volumes':
        mount = Path('/Volumes') / parts[2]
        if not mount.is_mount():
            raise RuntimeError(f'Storage volume is not mounted: {mount}; reconnect it before continuing')
    return path


def configure_storage(project, directory=None):
    project = Path(project).resolve()
    data = _absolute(directory) if directory is not None else data_directory(project)
    require_storage(data)
    # Check every existing entry before creating or changing any links.
    if data != project:
        if data.is_relative_to(project) or project.is_relative_to(data):
            raise ValueError('External DATA_DIR must be separate from the project folder')
        for name in DATA_FOLDERS:
            link, target = project / name, data / name
            if link.is_symlink():
                if link.resolve() != target.resolve():
                    raise ValueError(f'{link} already links elsewhere; existing data was left unchanged')
            elif link.exists() and (not link.is_dir() or any(link.iterdir())):
                raise ValueError(f'{link} already contains data; move or back it up before selecting a different storage folder')
            if target.exists() and not target.is_dir():
                raise ValueError(f'{target} is not a directory')
    data.mkdir(parents=True, exist_ok=True)
    for name in DATA_FOLDERS:
        target, link = data / name, project / name
        target.mkdir(exist_ok=True)
        for folder in MODEL_FOLDERS:
            (target/folder).mkdir(exist_ok=True)
        if data != project and not link.is_symlink():
            if link.exists():
                link.rmdir()  # Preflight verified this directory is empty.
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                # Data and the original target directory are preserved on failure.
                raise RuntimeError('Cannot create storage links on this filesystem; use a local project filesystem that supports symlinks') from None
    settings = project / 'storage.json'
    with tempfile.NamedTemporaryFile(dir=project, prefix='storage-', suffix='.tmp', mode='w', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump({'data_dir': str(data)}, stream, indent=2)
        stream.write('\n')
    try:
        temporary.replace(settings)
    finally:
        temporary.unlink(missing_ok=True)
    return data


def setup_storage(project, directory=None, *, interactive=None, prompt=input):
    project = Path(project).resolve()
    configured = (project / 'storage.json').is_file() or (project / DATA_FOLDERS[0]).is_symlink()
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get('NONINTERACTIVE') != '1'
    if directory is None and 'DATA_DIR' not in os.environ and not configured and interactive:
        print('Choose where models and generated data will be stored.')
        print(f'Leave blank to use this project: {project}')
        answer = prompt('Model store folder (one subfolder per model): ').strip()
        directory = answer.strip('"').strip("'") if answer else str(project)
    return configure_storage(project, directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', help='Optional absolute external data directory')
    parser.add_argument('--non-interactive', action='store_true', help='Use saved storage or the project folder without asking')
    args = parser.parse_args()
    try:
        data = setup_storage(Path(__file__).resolve().parent, args.data_dir, interactive=False if args.non_interactive else None)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f'Storage setup failed: {error}\n')
    print(f'Data storage: {data}')
