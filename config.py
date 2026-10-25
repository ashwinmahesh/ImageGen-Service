"""All large assets live beside this file on the external SSD."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "weights"
IMAGE_DIR = ROOT / "images"
MODEL_ID = "filipstrand/Z-Image-Turbo-mflux-4bit"
MODEL_REVISION = "b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b"
API_MODEL = "z-image-turbo"

os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "matplotlib"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
