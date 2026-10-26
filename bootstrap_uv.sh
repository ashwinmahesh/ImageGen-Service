#!/bin/sh
# Pinned official uv binary; installs only inside this checkout, without shell edits.
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
if [ -L "$project_root/.bootstrap" ]; then
    echo 'Keep .bootstrap inside the serving project; it must not be a symlink.' >&2
    exit 1
fi
uv_version=0.13.5
uv_sha256=743f85035a1996ecb350659e01cd9faea9709829be09b1b4b03c73eaf3c9cd73
uv_binary="$project_root/.bootstrap/uv-$uv_version"
if [ ! -x "$uv_binary" ]; then
    if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
        echo 'Automatic Python installation supports Apple Silicon macOS.' >&2
        exit 1
    fi
    mkdir -p "$project_root/.bootstrap"
    scratch=$(mktemp -d "$project_root/.bootstrap/uv-download.XXXXXX")
    trap 'rm -rf "$scratch"' EXIT HUP INT TERM
    echo "Downloading uv $uv_version into the project to manage Python." >&2
    curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 \
        "https://github.com/astral-sh/uv/releases/download/$uv_version/uv-aarch64-apple-darwin.tar.gz" \
        -o "$scratch/uv.tar.gz"
    actual=$(shasum -a 256 "$scratch/uv.tar.gz")
    actual=${actual%% *}
    if [ "$actual" != "$uv_sha256" ]; then
        echo 'uv download checksum did not match; rerun setup to retry.' >&2
        exit 1
    fi
    tar -xzf "$scratch/uv.tar.gz" -C "$scratch" uv-aarch64-apple-darwin/uv
    mv "$scratch/uv-aarch64-apple-darwin/uv" "$uv_binary"
fi
printf '%s\n' "$uv_binary"
