"""Download only catalogued adapters at pinned revisions, verifying SHA256."""
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download
from lora_catalog import LORA_DIR, catalog


def main():
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    for entry in catalog():
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
    main()
