# Setup and runtime reference

Start with the [README quick start](../README.md#quick-start). This page explains advanced configuration, model sources and operational details. For image requests and queue semantics, see the [API reference](api.md).

## First-time setup

Requires an Apple Silicon Mac with Metal, Git, Make and Python 3.14 (the tested version). With Homebrew: `brew install python@3.14`. The pinned Chroma native binary targets macOS 26.6.2 arm64; other systems need a compatible native build. The Python dependency lock includes MLX/MFLUX for Z-Image. This is not a Windows/CUDA distribution.

```sh
cd /absolute/path/to/downloaded-project
make setup BOOTSTRAP_PYTHON=/opt/homebrew/bin/python3.14
make test
make serve
```

**`make setup` lists all supported models** in a terminal:

```text
Choose an image model:
  1. Z-Image Turbo (z-image-turbo)
  2. Chroma1-HD (chroma1-hd)
  3. Chroma1-Flash (chroma1-flash)
Model [1 / Z-Image Turbo]:
```

Press Enter for **Z-Image Turbo**, or enter a number or full model ID. Setup installs dependencies and downloads the selected image model's required assets plus one shared prompt-expansion LLM; it does not start serving. Repeat setup to install another model. All models use the same environment and source code.

**`make serve` numbers only models already set up.** Other supported models are shown separately without numbers, with the command to set each up. For example, with only Z-Image installed:

```text
Choose a set up image model:
  1. Z-Image Turbo (z-image-turbo)

Not set up (not selectable):
  - Chroma1-HD (chroma1-hd) — make setup MODEL=chroma1-hd
  - Chroma1-Flash (chroma1-flash) — make setup MODEL=chroma1-flash
Run a model's setup command to add it to the numbered options, then run make serve again.
Model [1 / Z-Image Turbo]:
```

Enter prefers Z-Image when ready, otherwise the first ready model in registry order. Menu numbers refer only to the ready list. A listed but unready model cannot be selected by ID either; set it up first. If none are ready, the launcher shows setup commands and exits without starting or stopping a server. Discovery checks the completion marker and required local paths, including Chroma's shared encoder/VAE and native runtime/library. It does not load GPU weights or perform a full checksum scan; startup remains the final loadability check. A missing external volume reports a mount error, not an empty fallback store.

Explicit selection skips the menu and still validates the selected model's assets before touching a running server:

```sh
make setup MODEL=z-image-turbo
make setup MODEL=chroma1-flash
make serve MODEL=chroma1-flash
make serve MODEL=chroma1-hd NO_LORAS=1
make serve-no-loras MODEL=z-image-turbo
```

Without a terminal, or with `NONINTERACTIVE=1`, setup/download default to Z-Image Turbo. Serving defaults to Z-Image when ready, otherwise the first ready model, or exits with setup guidance if none are ready. `MODEL=...` works for setup, download, serving and adapter commands. `IMAGE_MODEL` is the equivalent Python environment setting. `PORT=5733` changes the endpoint. Use `BOOTSTRAP_PYTHON=/absolute/path/to/python3.14` if `python3.14` is installed elsewhere. `VENV=/absolute/internal/path/.venv` overrides the environment; use the same value consistently.

## Model storage

First setup also asks for a model store folder. **Blank/Enter uses the project directory**; another absolute directory (or `~/...`) stores data there and creates per-model symlinks in the project. The chosen store contains one folder per model:

```text
<store>/z-image-turbo/{weights,images,references,image-index,cache,loras,logs}
<store>/chroma1-hd/{weights,images,references,image-index,cache,loras,logs}
<store>/chroma1-flash/{weights,images,references,image-index,cache,loras,logs}
<store>/prompt-enhancer/weights  # shared Qwen3.5-2B 4-bit, about 1.7 GB
```

Source, Git, UI, the native runtime, `.venv`, and **SQLite state** stay in the project. Per-model queue databases and their journals/worker leases live in `<project>/state/queues/<model-id>/`, independently of `DATA_DIR`; `state/` is ignored by Git. The store choice is remembered in ignored `storage.json`. Repeated setup reuses it and existing assets without another storage prompt. Existing links are recognized. Conflicting/nonempty folders are never overwritten or silently moved. Missing external volumes fail clearly without an internal fallback download.

```sh
# Automation: choose model and store explicitly, with no prompts.
make setup MODEL=z-image-turbo DATA_DIR="/Volumes/MySSD/models" NONINTERACTIVE=1
# Only configure storage links:
make storage DATA_DIR="/Volumes/MySSD/models"
# Reuse saved storage, or the project-local default on a fresh checkout:
make setup MODEL=z-image-turbo NONINTERACTIVE=1
```

`DATA_DIR` means the **store containing model folders**, not one model's weights folder. A fresh clone needs no external SSD or original user's home path. Keep source and environment in the shared project and use the configured store for weights and media.

Chroma HD and Flash share their T5 encoder and VAE in `chroma1-hd/weights`, and one native runtime in `<project>/runtime`. **Flash setup does not download the HD diffusion checkpoint.** Back up the actual SSD data, including indexes and tags, **and the project-local `state/` folder**; symlinks are not backups.

## Start, switch and stop

`make queue` starts or reuses the lightweight queue API in the background at **http://127.0.0.1:55302**; it returns once ready. It needs the installed Python environment, but no model weights, LLM, SSD or GPU. Logs are in ignored `state/queue-api/server.log`. No login/autostart service is installed.

`make serve` ensures that same queue API is running, then runs the selected **generator and UI** in the foreground; keep its terminal open. Ctrl-C finishes active generation and stops the generator. **The queue API stays running**, and other agents can keep submitting jobs. `make stop-queue` stops only the verified queue process and retains every job. The generator reads/checkpoints SQLite directly, so previously accepted jobs can finish even if the queue HTTP process stops; UI requests still read/write SQLite through the generator API while the standalone queue API is down; only clients using its separate origin need it restarted.

- Webpage: <http://127.0.0.1:5732/>
- All-model image gallery: <http://127.0.0.1:5732/gallery>
- Generator API docs and live schema: <http://127.0.0.1:5732/docs> and `/openapi.json`
- Queue-only API docs: <http://127.0.0.1:55302/docs> (available without generation)
- Readiness: `make health` or `GET /health`

The default is **`http://127.0.0.1:5732/`**, also reachable as `http://localhost:5732/`. `PORT=...` changes the port for serving, health and smoke commands; use the same override for each command. `QUEUE_PORT=...` changes the independent queue API port; use the same value for `make queue`, `make serve` and `make stop-queue`. The Python generator uses `IMAGEGEN_QUEUE_PORT` when invoked directly. Both services bind to loopback. No hosts-file mapping or `http://imagegen` shortcut is installed.

Health reports `status`, active `model`, `busy`, `project_dir`, active `data_dir`, `loras_enabled`, `prompt_expansion`, model `capabilities`, and durable `queue` counts/state/database path. `GET /v1/models` lists all supported models, download markers and the active one. A download marker indicates completed setup, not a full startup integrity check; startup verifies required local assets too. The UI reads model identity, step defaults and limits from health.

To switch, stop the server and run `make serve` again, or run `make serve MODEL=...` in another terminal. The launcher reuses an identical instance or gracefully replaces a **verified managed idle** service. It refuses to interrupt generation or replace unrelated listeners. It never hot-swaps models inside an inference request. A request's `model`, if specified, must equal the active model; mismatches return 422. Serving never downloads models. If assets are missing, run `make setup MODEL=<id>` first.

If the launcher finds the same model/mode already running from this checkout, it prints the URL and exits successfully; the original terminal still owns that server. A model switch leaves its waiting queue persisted until that model starts again. During initial model loading the HTTP listener may not yet be reachable; connection refusal can be normal until startup completes. Read logs if readiness never arrives.

## Models and effective settings

| Startup ID | Runtime/checkpoint | Default steps | Allowed steps | Sampling |
| --- | --- | ---: | ---: | --- |
| `z-image-turbo` | MFLUX/MLX 4-bit | 9 | 1–12 | MFLUX Turbo defaults |
| `chroma1-hd` | stable-diffusion.cpp Metal, Q4_K_M | 40 | 1–80 | Euler, CFG 3 |
| `chroma1-flash` | stable-diffusion.cpp Metal, Q4_K_M | 8 | 1–40 | Heun, CFG 1, beta schedule, flow shift 1 |

Pinned Hugging Face sources and revisions are in `model_profiles.py`: [Z-Image Turbo](https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit), [Chroma1-HD](https://huggingface.co/silveroxides/Chroma1-HD-GGUF), [Chroma1-Flash](https://huggingface.co/silveroxides/Chroma1-Flash-GGUF), [shared T5 encoder](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf), and [VAE](https://huggingface.co/lodestones/Chroma1-HD). Native runtime release, download URL and checksums are pinned there too. Download scripts resume Hugging Face transfers; Chroma files and the runtime archive are SHA256-verified before recording completion. See upstream cards/licenses for usage terms.

Only fields in the [generation API reference](api.md#generation-api) are accepted. Negative prompts, guidance/CFG, custom samplers/schedulers, masks and instruction-editing controls are not exposed as request options; Chroma sampling values above are fixed by its backend profile. No model-specific API option namespace is implemented yet.

Z-Image keeps a loaded instance; Chroma loads GPU weights per CLI generation and releases them afterward. Chroma can be substantially slower on this Mac. Native sampling progress, when available, appears in `/health` while busy. Resolution, reference preservation, and steps affect latency. Do not infer Mac performance from CUDA benchmarks.

## Optional adapters

Default setup downloads no adapters. **Realism for Z-Image Turbo is the only bundled optional preset**:

```sh
make setup MODEL=z-image-turbo WITH_REALISM=1
make download-realism MODEL=z-image-turbo
make install-lora MODEL=z-image-turbo FILE=/absolute/path/adapter.safetensors LORA_ID=realism
```

Public downloads use only tracked `lora-catalog.json`, never personal catalogs. Compatible personal catalogs/checkpoints may live under ignored per-model `loras` folders; HD/Flash share `chroma1-hd/loras`. `GET /v1/loras` returns only the active model's entries and availability. The loader applies an adapter directly; a configured trigger phrase is prepended once automatically. Do not add `<lora:...>` tags to prompts. `--no-loras` / `NO_LORAS=1` hides the selector, returns an empty disabled catalog and rejects adapter requests. Adapter files and personal presets are never included in Git/default setup.

## Verification and troubleshooting

`make test` discovers the suite in `tests/` and runs fake-backend/storage/download tests without weights/GPU work. Run commands from the project root. For a focused module, use `.venv/bin/python -m unittest tests.test_queue -v`. `make smoke` calls the running model, creates a real 512px image with its default steps, decodes it and rejects a blank result. This can be slow for Chroma. Health and gallery are responsive during inference. Queue checks cover ordering, restart recovery, artifact reconciliation, idempotency, cross-process ownership, shared-GPU contention and waiting-only delete/clear. Native generation logs are under the active model's `logs/`.

If a download is interrupted, rerun `make setup MODEL=<id>` or `make download MODEL=<id>`; files are reused/resumed. An incomplete setup has no new completion marker. For a missing SSD, reconnect it; do not remove broken links or choose a fallback to hide the problem. A port conflict should be resolved with `PORT=...` or by stopping the identified service. Avoid multiple GPU services on a memory-constrained Mac. If setup cannot create symlinks, use a symlink-capable project filesystem or project-local storage. Local model paths and the native runtime are verified before inference; startup does not fetch network resources.

Both services accept server-local paths and bind to `127.0.0.1`. They are intended for trusted local clients; exposing them to a network requires access control. Only the four approved documentation screenshots are tracked images. Weights, media, references, indexes, settings, SQLite state, caches, logs and credentials stay outside Git.

For code changes, follow [AGENTS.md](../AGENTS.md). For an AI operating the service, use [the service skill](../skill/image-model-server/SKILL.md).
