"""Download a pinned MLX 4-bit checkpoint directly to the SSD, without symlinks."""
import json
from config import MODEL_DIR, MODEL_ID, MODEL_REVISION
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(MODEL_DIR),
        allow_patterns=["text_encoder/*", "tokenizer/*", "transformer/*", "vae/*", "README.md"],
        max_workers=3,
    )
    (MODEL_DIR / "download.json").write_text(json.dumps({
        "repository": MODEL_ID, "revision": MODEL_REVISION,
    }, indent=2) + "\n")
    print(f"Model ready: {MODEL_DIR}")
