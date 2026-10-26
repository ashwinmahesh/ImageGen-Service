#!/bin/sh
# Run dependency-free setup/menu code with an available Python. Serving stays offline.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
mode=$1
shift
case "$mode" in setup|run) ;; *) echo 'Expected setup or run mode.' >&2; exit 1 ;; esac
if [ "$mode" = setup ]; then
    trap 'printf "\nSetup cancelled. Existing files were kept; rerun make setup to continue.\n" >&2; exit 130' INT
fi
can_run() {
    "$1" -I -c 'import platform,sys; raise SystemExit(sys.implementation.name != "cpython" or sys.version_info < (3,9) or platform.system() != "Darwin" or platform.machine() != "arm64")' >/dev/null 2>&1
}
if [ -n "${BOOTSTRAP_PYTHON:-}" ]; then
    if can_run "$BOOTSTRAP_PYTHON"; then exec "$BOOTSTRAP_PYTHON" "$@"; fi
    echo 'BOOTSTRAP_PYTHON must point to an executable CPython 3.9 or newer for Apple Silicon macOS.' >&2
    exit 1
fi
for python_bin in "${VENV:-$project_root/.venv}/bin/python" \
    "$project_root"/.venv*/bin/python "$project_root"/.python/*/bin/python3 \
    python3 python3.14 python3.13 python3.12 python3.11 python3.10 python3.9; do
    if can_run "$python_bin"; then exec "$python_bin" "$@"; fi
done
if [ "$mode" != setup ]; then
    echo 'No usable project environment or Python found. Run make setup; serving never downloads Python.' >&2
    exit 1
fi
target_env=${VENV:-$project_root/.venv}
if [ -e "$target_env" ] || [ -L "$target_env" ]; then
    echo "Existing environment $target_env is broken or incompatible and was preserved; choose another VENV or move it aside before setup." >&2
    exit 1
fi
if [ -L "$project_root/.python" ]; then
    echo 'Keep .python inside the serving project; it must not be a symlink.' >&2
    exit 1
fi
uv_binary=$(sh "$project_root/bootstrap_uv.sh")
export UV_PYTHON_INSTALL_DIR="$project_root/.python"
export UV_PYTHON_BIN_DIR="$project_root/.bootstrap/bin"
export UV_CACHE_DIR="$project_root/.bootstrap/cache"
export UV_PYTHON_INSTALL_BIN=0 UV_NO_MODIFY_PATH=1
echo 'No installed Python found; setting up a project-local Python.' >&2
"$uv_binary" --no-config python install 3.14.9
python_bin=$("$uv_binary" --no-config python find --managed-python --no-python-downloads 3.14.9)
case "$python_bin" in "$project_root"/.python/*) ;; *) echo 'Managed Python must stay inside the project.' >&2; exit 1 ;; esac
if ! can_run "$python_bin"; then echo 'The downloaded Python failed validation; rerun setup.' >&2; exit 1; fi
exec "$python_bin" "$@"
