"""One serving project, a selected backend, and separate per-model data."""
import os
from pathlib import Path
from model_profiles import PROFILES, select_model, RUNTIME_RELEASE, RUNTIME_URL, RUNTIME_SHA256, SHARED_ASSETS
from storage_paths import data_directory, require_storage

PROJECT_ROOT=Path(__file__).resolve().parent
STORE_ROOT=data_directory(PROJECT_ROOT)
API_MODEL=select_model(interactive=False)
PROFILE=PROFILES[API_MODEL]
MODEL_ROOTS={identifier: STORE_ROOT/identifier for identifier in PROFILES}
ROOT=MODEL_ROOTS[API_MODEL]
MODEL_DIR=ROOT/'weights'
IMAGE_DIR=ROOT/'images'
MODEL_ID=PROFILE['repository']
MODEL_REVISION=PROFILE['revision']
MODEL_FILE=MODEL_DIR/PROFILE.get('filename','unused')
SHARED_ROOT=MODEL_ROOTS['chroma1-hd']
T5_FILE=SHARED_ROOT/'weights'/SHARED_ASSETS[0][2]
VAE_FILE=SHARED_ROOT/'weights'/SHARED_ASSETS[1][2]
RUNTIME_DIR=PROJECT_ROOT/'runtime'
ASSETS=[] if PROFILE['backend']=='mflux' else [(MODEL_ID,MODEL_REVISION,PROFILE['filename'],PROFILE['sha256'])]
for name,relative in [('HF_HOME','cache/huggingface'),('XDG_CACHE_HOME','cache'),('MPLCONFIGDIR','cache/matplotlib')]:
    os.environ[name]=str(ROOT/relative)
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
