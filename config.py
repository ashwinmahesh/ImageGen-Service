"""Project code lives internally; model assets and media use DATA_DIR."""
import os
from pathlib import Path
from storage_paths import data_directory, require_storage

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = data_directory(PROJECT_ROOT)
MODEL_DIR = ROOT / "weights"
IMAGE_DIR = ROOT / "images"
MODEL_ID = "filipstrand/Z-Image-Turbo-mflux-4bit"
MODEL_REVISION = "b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b"
API_MODEL = "z-image-turbo"

os.environ["HF_HOME"] = str(ROOT / "cache" / "huggingface")
os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache")
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache" / "matplotlib")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
