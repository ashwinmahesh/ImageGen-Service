"""Download shipped public presets only; never inspect the local serving catalog."""
import argparse
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download
from lora_catalog import LORA_DIR, catalog


def main(identifier=None):
    entries = catalog(local=False)
    if identifier is not None:
        entries = [entry for entry in entries if entry["id"] == identifier]
        if not entries:
            raise ValueError("Unknown shipped adapter ID; local serving presets are not downloadable here")
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        if not entry.get("repository"):
            continue
        destination = LORA_DIR / entry["filename"]
        if destination.is_file():
            with destination.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum == entry["sha256"]:
                print(f"Already verified {entry['id']}", flush=True)
                continue
        path = Path(hf_hub_download(entry["repository"], entry["remote_filename"],
            revision=entry["revision"], local_dir=str(LORA_DIR / "downloads")))
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {entry['id']}")
        # Rename the verified download inside the SSD, retaining Hub resume metadata.
        path.replace(destination)
        print(f"Verified {entry['id']}: {destination} ({destination.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", help="Download one ID from the shipped public catalog")
    main(parser.parse_args().id)
