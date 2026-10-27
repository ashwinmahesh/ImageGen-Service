"""Import a downloaded Z-Image adapter without modifying the source file."""
import argparse
import hashlib
import json
from pathlib import Path
from storage_paths import require_storage
import shutil
import tempfile
from config import API_MODEL
from adapter_validation import inspect_adapter
from lora_catalog import LORA_DIR, catalog


def install(source, identifier):
    source = Path(source).expanduser().resolve()
    # Keep other model entries in the shared local catalog when updating one adapter.
    local=LORA_DIR/'catalog.json'
    entries=json.loads(local.read_text()) if local.is_file() else catalog(local=False)
    require_storage(LORA_DIR)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    entry = next((item for item in entries if item["id"] == identifier and API_MODEL in item["models"]), None)
    if entry is None:
        raise ValueError("Unknown preset ID; use make setup-lora to add a custom adapter")
    inspect_adapter(source,API_MODEL)
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
