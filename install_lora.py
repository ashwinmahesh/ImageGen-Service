"""Import a downloaded Z-Image adapter without modifying the source file."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
from lora_catalog import LORA_DIR, catalog


def install(source, identifier):
    source = Path(source).expanduser().resolve()
    entries = catalog()
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    entry = next((item for item in entries if item["id"] == identifier), None)
    if entry is None:
        raise ValueError("Unknown adapter ID; add a compatible entry to lora-catalog.json or loras/catalog.json first")
    # Parse the data-only format, rejecting an incorrect variant before copying.
    with source.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length <= 16 * 1024**2:
            raise ValueError("Invalid safetensors header")
        header = json.loads(stream.read(length))
    keys = [key.lower() for key in header if key != "__metadata__"]
    info = header.get("__metadata__", {})
    metadata = json.dumps({key: info.get(key) for key in [
        "ss_base_model_version", "ss_model_type", "modelspec.architecture", "ss_output_name"
    ]}).lower()
    if not keys or not any("lora" in key for key in keys):
        raise ValueError("This file does not contain LoRA tensors")
    if any(word in metadata for word in ["chroma", "illustrious", "sdxl", "flux"]):
        raise ValueError("Wrong model family: import the Z-Image Turbo version")
    if not any("layers." in key and ".attention." in key for key in keys):
        raise ValueError("Unrecognized Z-Image transformer adapter tensors")
    destination = LORA_DIR / entry["filename"]
    if source != destination.resolve():
        with tempfile.NamedTemporaryFile(dir=LORA_DIR, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    with destination.open("rb") as stream:
        entry["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    entry["bytes"] = destination.stat().st_size
    temporary = LORA_DIR / "catalog.json.partial"
    temporary.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(LORA_DIR / "catalog.json")
    print(f"Installed {identifier}: {destination}\nSHA256: {entry['sha256']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--id", default="realism")
    args = parser.parse_args()
    install(args.file, args.id)
