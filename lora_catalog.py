"""Named, model-specific adapters for the local Z-Image service."""
import hashlib
import re
from adapter_catalog import read_catalog, adapter_path as catalog_adapter_path, list_entries
from config import API_MODEL, PROJECT_ROOT as ROOT, ROOT as DATA_ROOT, SHARED_ROOT, PROFILE

LORA_DIR = (SHARED_ROOT if PROFILE["backend"] == "sd-cpp" else DATA_ROOT) / "loras"
_verified = {}


def catalog(*, local=True):
    return read_catalog(ROOT,LORA_DIR,API_MODEL,local=local)


def adapter_path(entry):
    return catalog_adapter_path(LORA_DIR,entry)


def list_loras():
    return list_entries(LORA_DIR,catalog())


def resolve_lora(identifier, strength=None, *, model=None):
    if identifier is None:
        return None
    if model is not None and model!=API_MODEL:
        from adapter_catalog import model_loras
        directory,entries=model_loras(ROOT,model)
    else:directory,entries=LORA_DIR,catalog()
    entry = next((entry for entry in entries if entry["id"] == identifier), None)
    if entry is None:
        raise ValueError("Unknown LoRA for this model; see GET /v1/loras")
    path = catalog_adapter_path(directory,entry)
    if not path.is_file() or not entry.get("sha256"):
        raise ValueError("LoRA is not installed. Import a compatible .safetensors file with make install-lora FILE=/absolute/path")
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
