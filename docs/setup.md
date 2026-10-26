# Setup and runtime reference

Start with the [README quick start](../README.md#quick-start). This page explains advanced configuration, model sources and operational details. For image requests and queue semantics, see the [API reference](api.md).

## First-time setup

Requires an Apple Silicon Mac with Metal, Git and Make. You do not need to install Python first: setup reuses a compatible existing environment or interpreter and obtains a missing compatible interpreter inside the project. The Python lock resolves for macOS 14+ arm64; native-runtime requirements below remain separate. The pinned Chroma native binary targets macOS 26.6.2 arm64; other systems need a compatible native build. The Python dependency lock includes MLX/MFLUX for Z-Image, FLUX.2 Klein and Qwen-Image 2.1 Turbo. This is not a Windows/CUDA distribution.

```sh
cd /absolute/path/to/downloaded-project
make setup
make test
make serve
```

**`make setup` lists all supported models** in a terminal. Ready models show `[SETUP COMPLETE]` beside their name; for example, with Z-Image Turbo already installed:

```text
Choose an image model:
  1. Z-Image Turbo [SETUP COMPLETE] (z-image-turbo)
  2. Chroma1-HD (chroma1-hd)
  3. Chroma1-Flash (chroma1-flash)
  4. FLUX.2 Klein 4B (flux2-klein-4b)
  5. FLUX.2 Klein 9B (flux2-klein-9b)
  6. Qwen-Image 2.1 Turbo (qwen-image-2.1-turbo)
  q. Cancel setup
Model [1 / Z-Image Turbo]:
```

Press Enter for **Z-Image Turbo**, or enter a number or full model ID. Setup installs dependencies and downloads the selected image model's required assets plus one shared prompt-expansion LLM; it does not start serving. Repeat setup to install another model. All models use the same environment and source code.

Cancel with `q` (also `quit`/`cancel`) at the model menu, or **Ctrl+C at any point during setup**, including storage selection, dependency installation and downloads. An input EOF at the model menu also cancels. Setup prints a short cancellation message without a Python traceback and stops only its own setup subprocess group. Existing files and cached partial downloads are retained; rerun `make setup` to finish or repair interrupted setup. No generation/queue service is started or stopped.

The startup log explains that **rerunning setup for an already downloaded model is safe**: unchanged model weights are a no-op with no redownload, while missing or changed pinned files are downloaded/updated in place. It does not create a duplicate model folder or automatically switch to an unpinned upstream revision. Dependency installation/checks, shared prompt-model verification and completion metadata can still run, so the entire setup command is not a strict no-op.

Setup labels use the same read-only local asset preflight as the serving menu, including shared Chroma files/runtime and indexed MLX checkpoint shards. An old completion marker alone is insufficient. The label reports model assets, not a running server or optional LLM/LoRA readiness. It does not load weights, hash large files or download anything. Missing/unreadable storage gets no completion labels; the storage setup step still reports configuration or mount errors. All models remain selectable with unchanged numbers/default, including completed models. Explicit model selection and noninteractive setup skip the menu and its readiness scan.

**`make serve` numbers only setup models to choose a default; every setup model remains eligible for queued work.** Klein names show `(edits enabled)` and Qwen shows `(edit selected area enabled)`. Other supported models are shown separately without numbers, with the command to set each up. For example, with only Z-Image installed:

```text
Choose the default image model (all setup models can run):
  1. Z-Image Turbo (z-image-turbo)

Not set up (not selectable):
  - Chroma1-HD (chroma1-hd) — make setup MODEL=chroma1-hd
  - Chroma1-Flash (chroma1-flash) — make setup MODEL=chroma1-flash
  - FLUX.2 Klein 4B (edits enabled) (flux2-klein-4b) — make setup MODEL=flux2-klein-4b
  - FLUX.2 Klein 9B (edits enabled) (flux2-klein-9b) — make setup MODEL=flux2-klein-9b
  - Qwen-Image 2.1 Turbo (edit selected area enabled) (qwen-image-2.1-turbo) — make setup MODEL=qwen-image-2.1-turbo
Run a model's setup command to add it to the numbered options, then run make serve again.
Model [1 / Z-Image Turbo]:
```

Enter prefers a ready saved default, otherwise ready Z-Image or the first ready model in registry order. Menu numbers refer only to the ready list. A listed but unready model cannot be selected by ID either; set it up first. At least one model must have completed setup and pass readiness checks before serving, including explicit `MODEL` and `NO_LLM`/`NO_LORAS` startup. If the environment is missing or no models are ready, serving exits nonzero with `make setup` guidance before any LLM question, queue startup, SQLite creation or changes to running services. Discovery checks the completion marker and required local paths, including Chroma's shared encoder/VAE and native runtime/library. It does not load GPU weights or perform a full checksum scan; startup remains the final loadability check. A missing external volume reports a mount error, not an empty fallback store.

Explicit selection skips the menu and still validates the selected model's assets before touching a running server:

```sh
make setup MODEL=z-image-turbo
make setup MODEL=chroma1-flash
make serve MODEL=chroma1-flash
make serve MODEL=chroma1-hd NO_LORAS=1
make serve-no-loras MODEL=z-image-turbo
```

Without a terminal, or with `NONINTERACTIVE=1`, setup/download default to Z-Image Turbo. Serving reuses a ready saved default, otherwise ready Z-Image or the first ready model, or exits with setup guidance if none are ready. `MODEL=...` works for setup, download, serving and adapter commands. `IMAGE_MODEL` is the equivalent Python environment setting. `PORT=5733` changes the endpoint. An optional `BOOTSTRAP_PYTHON=/absolute/path/to/python3.12` selects a compatible interpreter during setup; it is not needed for serving or model-only downloads. `VENV=/absolute/internal/path/.venv` overrides the environment; use the same value consistently.

## Python selection

All image backends and prompt expansion continue to share one environment. `make setup` and `make setup-env` follow this order:

1. Validate and reuse the target `VENV` (default `.venv`). A compatible environment keeps its existing Python.
2. If that environment does not exist, try compatible installed Python interpreters, including existing project-local Python downloads.
3. Only if none is suitable, obtain the tested Python 3.14.9 under `<project>/.python` and create the environment from it.
4. Install the exact `requirements.lock.txt` pins without resolving extra unpinned dependencies, then run `pip check`.

| Compatibility | Current policy |
| --- | --- |
| Interpreter | Standard CPython 3.12, 3.13 or 3.14, final releases, native Apple Silicon macOS |
| Excluded patch | Python 3.14.1, explicitly excluded by NetworkX 3.7 |
| Lower bound | Python 3.12, required by the pinned NumPy 2.5.4, ContourPy 1.4.0 and NetworkX 3.7 |
| Unvalidated builds | Python 3.15+, older minors, free-threaded Python, PyPy, x86/Rosetta and non-macOS runtimes |
| Managed download | Python 3.14.9, only during setup and only inside this project |

The full 67-package lock was checked using binary-only resolution for each supported minor on macOS 14 arm64, rather than inferring compatibility from MFLUX alone. The installed 3.14.9 stack also passes `pip check` and the regression suite. Resolution verifies package compatibility; it does not establish identical inference or performance across interpreters. Revalidate the full lock and binary availability before widening these bounds. The supplied Chroma binary still targets macOS 26.6.2 arm64 and is independent of Python selection.

The dependency-free `bootstrap_python.sh` can run setup menus using an available native CPython 3.9+ without treating that menu interpreter as compatible with model dependencies. On a Python-free machine, it obtains the compatible project-local interpreter first. Its official [uv](https://docs.astral.sh/uv/concepts/python-versions/) helper is pinned to 0.13.5; the downloaded Apple Silicon archive is checked against the recorded SHA256 before execution. The helper and cache live under `.bootstrap`; managed interpreters live under `.python`. No system interpreter, global packages, PATH or shell profiles are changed. These folders stay separate from the model/media store, and neither is published to Git.

```sh
# Optional explicit interpreter and a fresh, ignored environment.
make setup BOOTSTRAP_PYTHON=/absolute/path/to/python3.12 VENV="$(pwd)/.venv-new"
make serve VENV="$(pwd)/.venv-new"
# Install/repair pinned packages without downloading model weights.
make setup-env
```

Explicit incompatible interpreters fail with an explanation; setup does not silently choose another one. An existing environment that is broken, incompatible, not a real virtual environment, or based on a different explicitly requested Python is preserved. Choose another `VENV` or move it aside yourself before rebuilding. A partial newly created environment is also left available for inspection after a failure. Repeated setup resumes safely without replacing working environments or changing model/media/queue locations.

`make serve`, `make queue`, `make stop-queue`, `make storage` and model/LoRA/LLM download commands run directly with the selected existing environment. They never provision Python or install packages. Model-download commands can still use the network for model assets. If no environment has been set up, run `make setup` first. The `BOOTSTRAP_PYTHON` override affects setup only. An existing configured external volume must still be mounted; Python setup does not create fallback model storage.

## Model storage

First setup also asks for a model store folder. **Blank/Enter uses the project directory**; another absolute directory (or `~/...`) stores data there and creates per-model symlinks in the project. The chosen store contains one folder per model:

```text
<store>/z-image-turbo/{weights,images,references,image-index,cache,loras,logs}
<store>/chroma1-hd/{weights,images,references,image-index,cache,loras,logs}
<store>/chroma1-flash/{weights,images,references,image-index,cache,loras,logs}
<store>/flux2-klein-4b/{weights,images,references,image-index,cache,loras,logs}
<store>/flux2-klein-9b/{weights,images,references,image-index,cache,loras,logs}
<store>/qwen-image-2.1-turbo/{weights,images,references,image-index,cache,loras,logs}
<store>/prompt-enhancer/weights  # shared Qwen3.5-2B 4-bit, about 1.7 GB
```

Source, Git, UI, the native runtime, `.venv`, managed `.python`/`.bootstrap` downloads, and **SQLite state** stay in the project. The shared queue database and their journals/worker leases live in `<project>/state/queue/`, independently of `DATA_DIR`; `state/` is ignored by Git. The store choice is remembered in ignored `storage.json`. Repeated setup reuses it and existing assets without another storage prompt. Existing links are recognized. Conflicting/nonempty folders are never overwritten or silently moved. Missing external volumes fail clearly without an internal fallback download.

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

`make serve` ensures that same queue API is running, then runs one **generator and UI** for every setup model in the foreground; keep its terminal open. Startup briefly explains that the optional local text model can add visual detail to descriptions and can be disabled per image. Ctrl-C finishes active generation and stops the generator. **The queue API stays running**, and other agents can keep submitting jobs. `make stop-queue` stops only the verified queue process and retains every job. The generator reads/checkpoints SQLite directly, so previously accepted jobs can finish even if the queue HTTP process stops; UI requests still read/write SQLite through the generator API while the standalone queue API is down; only clients using its separate origin need it restarted.

- Webpage: <http://127.0.0.1:5742/>
- All-model image gallery: <http://127.0.0.1:5742/gallery>
- Generator API docs and live schema: <http://127.0.0.1:5742/docs> and `/openapi.json`
- Queue-only API docs: <http://127.0.0.1:55302/docs> (available without generation)
- Readiness: `make health` or `GET /health`

The default is **`http://127.0.0.1:5742/`**, also reachable as `http://localhost:5742/`. `PORT=...` changes the port for serving, health and smoke commands; use the same override for each command. `QUEUE_PORT=...` changes the independent queue API port; use the same value for `make queue`, `make serve` and `make stop-queue`. The Python generator uses `IMAGEGEN_QUEUE_PORT` when invoked directly. Both services bind to loopback. No hosts-file mapping or `http://imagegen` shortcut is installed.

Health reports service/version, `runtime_model_switching:true`, `queue_scope:shared`, saved `default_model`, nullable `active_model`, `model_loaded`, busy/stage and global queue state. GET `/v1/models` lists only setup models with per-model capabilities/defaults. A marker alone is insufficient; the shared preflight checks all required local assets and runtime files. Models load on demand after the HTTP listener starts.

Choose a setup model in each webpage/API request. The generator reads one shared queue and switches its managed backend subprocess between batches, reusing a resident Z-Image instance for consecutive jobs. Only one image backend is active; the previous process is stopped and reaped before another starts. `make serve MODEL=...` selects the saved default for omitted models on both APIs, not an exclusive backend. Repeating serve with the same modes updates that default without restarting even a busy matching generator; changing LLM/adapter modes still requires an idle verified process. No downloads happen during serving. Missing models need `make setup MODEL=<id>`.

If a matching generator is already running from this checkout, serve updates its saved default and reuses it; the original terminal still owns the process. Models switch inside the generator between jobs, while the HTTP listener stays responsive. Changing startup modes requires an idle verified generator. If an older queue process is running, stop it with `make stop-queue`, then run `make serve` to start the new shared queue.

## Models and effective settings

| Startup ID | Runtime/checkpoint | Default steps | Allowed steps | Sampling |
| --- | --- | ---: | ---: | --- |
| `z-image-turbo` | MFLUX/MLX 4-bit | 9 | 1–12 | MFLUX Turbo defaults |
| `chroma1-hd` | stable-diffusion.cpp Metal, Q4_K_M | 40 | 1–80 | Euler, CFG 3 |
| `chroma1-flash` | stable-diffusion.cpp Metal, Q4_K_M | 8 | 1–40 | Heun, CFG 1, beta schedule, flow shift 1 |
| `flux2-klein-4b` | MFLUX/MLX 8-bit | 4 | 1–4 | Distilled, guidance 1 |
| `flux2-klein-9b` | MFLUX/MLX 4-bit | 4 | 1–4 | Distilled, guidance 1 |
| `qwen-image-2.1-turbo` | MFLUX/MLX 4-bit, vision-capable | 8 | 1–8 | Checkpoint Turbo sigma schedule, guidance 1 |

Pinned Hugging Face sources and revisions are in `model_profiles.py`: [Z-Image Turbo](https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit), [Chroma1-HD](https://huggingface.co/silveroxides/Chroma1-HD-GGUF), [Chroma1-Flash](https://huggingface.co/silveroxides/Chroma1-Flash-GGUF), [shared T5 encoder](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf), and [VAE](https://huggingface.co/lodestones/Chroma1-HD). Native runtime release, download URL and checksums are pinned there too. Download scripts resume Hugging Face transfers; Chroma files and the runtime archive are SHA256-verified before recording completion. See upstream cards/licenses for usage terms.

Klein uses pinned community conversions: [4B Q8](https://huggingface.co/mflux-community/flux2-klein-4b-mflux-q8) and [9B Q4](https://huggingface.co/mflux-community/flux2-klein-9b-mflux-q4). [Qwen Turbo Q4](https://huggingface.co/mflux-community/qwen-image-2-1-turbo-mflux-q4) includes the text/vision encoder, processor and Turbo schedule. Setup validates required files, quantization, every indexed shard and Qwen's vision keys before writing its completion marker. Serving checks that marker against the pinned source/revision. This is a file/index preflight, not a real inference or full checksum scan.

Klein 4B is Apache 2.0. Klein 9B's FLUX Non-Commercial License requires a separate BFL license for commercial model use. Qwen's [Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE) permits research/evaluation use only and requires separate permission for commercial model use. Their restrictions are surfaced conditionally when that backend is loaded, on the webpage, and under each model's exact ID in discovery/capabilities; no acknowledgment step is added.

Only fields in the [generation API reference](api.md#generation-api) are accepted. Negative prompts, guidance/CFG, custom samplers/schedulers, multiple-reference controls are not exposed as request options; Qwen photo edits also accept a mask; Chroma sampling values above are fixed by its backend profile. No model-specific API option namespace is implemented yet.

Selected-area edits are available through both servers: generator `/v1/jobs` accepts flat `task:"edit"`, `reference_image` and `mask_image`; standalone queue `/v1/jobs` accepts those fields inside `request` with `model:"qwen-image-2.1-turbo"`. The queue can accept edits while the generator is off and validates media only when consumed. Keep source/mask paths accessible on this Mac through recovery. See the [complete selected-area examples](api.md#selected-area-qwen-edits).

Z-Image keeps a loaded instance. Chroma, Klein and Qwen load GPU weights per CLI generation and release them afterward. Klein and Qwen run the pinned MFLUX CLI with local checkpoint paths, offline flags and low-RAM callbacks; Qwen also uses VAE tiling. Their ready status means local assets and the worker passed preflight; GPU weights load on the first request. Text-only generation and reference editing choose the appropriate runtime pipeline. For Klein/Qwen references, omit `image_strength`; the prompt specifies the edit. Dedicated photo edits use `task:"edit"` or `/v1/images/edits`, require a source, support optional preservation requirements, and disable composition variation. Qwen accepts a mask; Klein supports whole-photo edits. Generate remains available for every model. The webpage selector lists only setup models and submits the chosen model with each job. Automatic loading stays off the HTTP process; the API remains responsive during switches. Qwen Turbo is intended for 1024px or larger, and memory varies with resolution and reference conditioning. The new models have not yet had real inference measured on this Mac. Chroma can be substantially slower on this Mac. Native sampling progress, when available, appears in `/health` while busy. Resolution, reference preservation, and steps affect latency. Do not infer Mac performance from CUDA benchmarks.

## Optional adapters

Default setup downloads no adapters. **Realism for Z-Image Turbo is the only bundled optional preset**:

```sh
make setup MODEL=z-image-turbo WITH_REALISM=1
make download-realism MODEL=z-image-turbo
make install-lora MODEL=z-image-turbo FILE=/absolute/path/adapter.safetensors LORA_ID=realism
```

Public downloads use only tracked `lora-catalog.json`, never personal catalogs. Compatible personal catalogs/checkpoints may live under ignored per-model `loras` folders; HD/Flash share `chroma1-hd/loras`. `GET /v1/loras?model=<id>` returns the requested setup model’s entries; omission uses the shared default. The loader applies an adapter directly; a configured trigger phrase is prepended once automatically. Do not add `<lora:...>` tags to prompts. `--no-loras` / `NO_LORAS=1` hides the selector, returns an empty disabled catalog and rejects adapter requests. Adapter files and personal presets are never included in Git/default setup.

## Observability

`make serve` starts an independent background sampling thread before loading the selected model, and serves `/observability`. It records the executing job model, session, stage and executing job/attempt alongside CPU/RAM/GPU samples. The standalone queue API does not collect metrics, own this database or serve this page. No recording daemon, OS service or administrator permission is installed. Stopping the generator ends sampling; its normal shutdown still finishes active work first.

History uses a separate WAL SQLite database at `<project>/state/observability/metrics.sqlite3`, with journals alongside it. This path must resolve inside the project; DATA_DIR never moves it. It is ignored by Git. Back up the state directory to preserve history. Raw samples are pruned after eight days, every five minutes during sampling; disk pages are reused rather than vacuumed during generation. Finished/interrupted job summaries remain. Queue schema and queue transactions are unchanged.

Sampling is every second during execution and every five seconds while idle, with boundary samples for short jobs. An idle sampler wakes when work begins. CPU/RAM use pinned `psutil==7.2.2`; GPU uses a bounded, non-sudo IORegistry query and backs off for 30 seconds when counters are unavailable. An existing clone can install the new pinned dependency with `make setup-env`; serving never installs packages. Unsupported counters are null. A sampler/storage failure is reported by health/the page and observability APIs without disabling generation or queue operations.

The graphs aggregate bounded sample means and sampled peaks; gaps/restarts remain visible. GPU counters are whole-device, and service RSS sums shared pages across child processes. Do not sum GPU driver memory with process/system RAM. Short-lived children and spikes can fall between samples; job correlation is not exclusive GPU attribution. See [Observability API](api.md#observability) for metrics, filters, pagination and examples.

## Verification and troubleshooting

`make test` discovers the suite in `tests/` and runs fake-backend/storage/download tests without weights/GPU work. Run commands from the project root. For a focused module, use `.venv/bin/python -m unittest tests.test_queue -v`. `make smoke` calls the running model, creates a real 512px image with its default steps, decodes it and rejects a blank result. This can be slow for Chroma. Health and gallery are responsive during inference. Queue checks cover ordering, restart recovery, artifact reconciliation, idempotency, cross-process ownership, shared-GPU contention and waiting-only delete/clear. Native generation logs are under the active model's `logs/`. New runtime attempts also save best-effort scheduler/Python output in `<project>/state/job-logs/<job-id>/attempt-<number>.log`. They are ignored by Git and retained until you remove local logs; metric retention does not prune them. The job monitor reads bounded tails, so old shared logs are never shown as if they belonged to one job.

If a download is interrupted, rerun `make setup MODEL=<id>` or `make download MODEL=<id>`; files are reused/resumed. An incomplete setup has no new completion marker. For a missing SSD, reconnect it; do not remove broken links or choose a fallback to hide the problem. A port conflict should be resolved with `PORT=...` or by stopping the identified service. Avoid multiple GPU services on a memory-constrained Mac. If setup cannot create symlinks, use a symlink-capable project filesystem or project-local storage. Local model paths and the native runtime are verified before inference; startup does not fetch network resources.

Both services accept server-local paths and bind to `127.0.0.1`. They are intended for trusted local clients; exposing them to a network requires access control. Only the four approved documentation screenshots are tracked images. Weights, media, references, indexes, settings, SQLite state, caches, logs and credentials stay outside Git.

For code changes, follow [AGENTS.md](../AGENTS.md). For an AI operating the service, use [the service skill](../skill/image-model-server/SKILL.md).

If switching models reports **Cannot safely release the previous model runtime**, the consumer stops to prevent overlapping weights. Current versions save the underlying scheduler cleanup traceback to `state/model-process/server.log` and the affected job’s attempt log. Older versions only captured child output there, so that file can lack the cause. Process cleanup inspects statuses only for the owned process group, reaps its leader, and ignores exited/zombie members. Restart with `make serve` after updating; use **Retry** for the failed request when ready. Waiting jobs remain persisted. Do not delete queue state to clear a runtime error.
