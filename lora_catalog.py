"""Named, model-specific adapters for the local Z-Image service."""
import hashlib
import json
from pathlib import Path
import re
from config import API_MODEL, ROOT

LORA_DIR = ROOT / "loras"
_verified = {}


def catalog(*, local=True):
    path = LORA_DIR / "catalog.json" if local else ROOT / "lora-catalog.json"
    if not path.is_file():
        path = ROOT / "lora-catalog.json"
    if not path.is_file():
        return []
    entries = json.loads(path.read_text())
    seen = set()
    for entry in entries:
        identifier, filename = entry["id"], entry["filename"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", identifier) or identifier in seen:
            raise ValueError("Invalid or duplicate LoRA ID in catalog.json")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", filename):
            raise ValueError("LoRA filenames must be plain .safetensors filenames")
        seen.add(identifier)
    return [entry for entry in entries if API_MODEL in entry["models"]]


def adapter_path(entry):
    path = (LORA_DIR / entry["filename"]).resolve()
    if path.parent != LORA_DIR.resolve():
        raise ValueError("LoRA must be inside the configured adapter directory")
    return path


def list_loras():
    return [{**entry, "available": adapter_path(entry).is_file() and bool(entry.get("sha256"))} for entry in catalog()]


def resolve_lora(identifier, strength=None):
    if identifier is None:
        return None
    entry = next((entry for entry in catalog() if entry["id"] == identifier), None)
    if entry is None:
        raise ValueError("Unknown LoRA for this model; see GET /v1/loras")
    path = adapter_path(entry)
    if not path.is_file() or not entry.get("sha256"):
        raise ValueError("LoRA is not installed. Import its Z-Image .safetensors file with make install-lora FILE=/absolute/path")
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, entry["sha256"])
    if key not in _verified:
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != entry["sha256"]:
            raise ValueError("LoRA checksum changed; re-import the adapter before generating")
        _verified.clear()
        _verified[key] = True
    return {"id": entry["id"], "name": entry["name"], "file_path": str(path),
            "strength": entry["default_strength"] if strength is None else strength,
            "sha256": entry["sha256"], "source": entry["source"],
            "trigger_prompt": entry.get("trigger_prompt", "")}


def effective_prompt(prompt, selection):
    trigger = selection.get("trigger_prompt", "") if selection else ""
    present = bool(trigger and re.search(r"(?<!\w)" + re.escape(trigger) + r"(?!\w)", prompt, flags=re.IGNORECASE))
    return f"{trigger}, {prompt}" if trigger and not present else prompt
