# Local Image Model Server

One Apple Silicon project for **Z-Image Turbo**, **Chroma1-HD**, and **Chroma1-Flash**, with a shared webpage, API, and searchable image gallery. Select a model when starting the server; one backend runs at a time. The repository URL remains `ashwinmahesh/Z-Image-Turbo-Service` for existing users.

## First-time setup

Requires an Apple Silicon Mac with Metal, Git, Make and Python 3.14 (the tested version). With Homebrew: `brew install python@3.14`. The pinned Chroma native binary targets macOS 26.6.2 arm64; other systems need a compatible native build. The Python dependency lock includes MLX/MFLUX for Z-Image. This is not a Windows/CUDA distribution.

```sh
cd /absolute/path/to/downloaded-project
make setup BOOTSTRAP_PYTHON=/opt/homebrew/bin/python3.14
make test
make serve
```

Both **`make setup` and `make serve` show a model menu** in a terminal:

```text
Choose an image model:
  1. Z-Image Turbo (z-image-turbo)
  2. Chroma1-HD (chroma1-hd)
  3. Chroma1-Flash (chroma1-flash)
Model [1 / Z-Image Turbo]:
```

Press Enter for **Z-Image Turbo**, or enter a number or full model ID. Setup installs dependencies and downloads only the selected model's required assets; it does not start serving. Repeat setup to install another model. All models use the same environment and source code. Explicit selection skips the menu:

```sh
make setup MODEL=z-image-turbo
make setup MODEL=chroma1-flash
make serve MODEL=chroma1-flash
make serve MODEL=chroma1-hd NO_LORAS=1
make serve-no-loras MODEL=z-image-turbo
```

Without a terminal, or with `NONINTERACTIVE=1`, an omitted model defaults to Z-Image Turbo. `MODEL=...` works for setup, download, serving and adapter commands. `IMAGE_MODEL` is the equivalent Python environment setting. `PORT=8001` changes the endpoint. Use `BOOTSTRAP_PYTHON=/absolute/path/to/python3.14` if `python3` resolves to another version. `VENV=/absolute/internal/path/.venv` overrides the environment; use the same value consistently.

## Model storage

First setup also asks for a model store folder. **Blank/Enter uses the project directory**; another absolute directory (or `~/...`) stores data there and creates per-model symlinks in the project. The chosen store contains one folder per model:

```text
<store>/z-image-turbo/{weights,images,references,image-index,cache,loras,logs}
<store>/chroma1-hd/{weights,images,references,image-index,cache,loras,logs}
<store>/chroma1-flash/{weights,images,references,image-index,cache,loras,logs}
```

Source, Git, UI, the native runtime, and `.venv` stay in the project. The store choice is remembered in ignored `storage.json`. Repeated setup reuses it and existing assets without another storage prompt. Existing links are recognized. Conflicting/nonempty folders are never overwritten or silently moved. Missing external volumes fail clearly without an internal fallback download.

```sh
# Automation: choose model and store explicitly, with no prompts.
make setup MODEL=z-image-turbo DATA_DIR="/Volumes/MySSD/models" NONINTERACTIVE=1
# Only configure storage links:
make storage DATA_DIR="/Volumes/MySSD/models"
# Reuse saved storage, or the project-local default on a fresh checkout:
make setup MODEL=z-image-turbo NONINTERACTIVE=1
```

`DATA_DIR` means the **store containing model folders**, not one model's weights folder. On the original Mac, the single serving project is `~/Documents/models/image-model-server`, and the store is `/Volumes/ASH-SSD/models`. Existing model, output, tag and custom-location index paths are reused in place. A fresh clone needs no external SSD or original user's home path. Prior separate source folders can be retained as backups; use the consolidated project for new changes.

Chroma HD and Flash share their T5 encoder and VAE in `chroma1-hd/weights`, and one native runtime in `<project>/runtime`. **Flash setup does not download the HD diffusion checkpoint.** Back up the actual SSD data, including indexes and tags; symlinks are not backups.

## Start, switch and stop

`make serve` runs in the foreground; keep the terminal open. Ctrl-C stops it gracefully. AI tools should retain the terminal session and use another session for checks.

- Webpage: <http://127.0.0.1:8000/>
- All-model image gallery: <http://127.0.0.1:8000/gallery>
- API docs and live schema: <http://127.0.0.1:8000/docs> and `/openapi.json`
- Readiness: `make health` or `GET /health`

Health reports `status`, active `model`, `busy`, `project_dir`, active `data_dir`, `loras_enabled`, and model `capabilities`. `GET /v1/models` lists all supported models, download markers and the active one. A download marker indicates completed setup, not a full startup integrity check; startup verifies required local assets too. The UI reads model identity, step defaults and limits from health.

To switch, stop the server and run `make serve` again, or run `make serve MODEL=...` in another terminal. The launcher reuses an identical instance or gracefully replaces a **verified managed idle** service. It refuses to interrupt generation or replace unrelated listeners. It never hot-swaps models inside an inference request. A request's `model`, if specified, must equal the active model; mismatches return 422. Serving never downloads models. If assets are missing, run `make setup MODEL=<id>` first.

The original Mac's shortcuts `serve-images`, `serve-z-image-turbo`, `serve-chroma1-hd`, `serve-chroma1-flash` and the three `-no-loras` variants all use this one project. Other users can use Make directly without aliases.

## Models and effective settings

| Startup ID | Runtime/checkpoint | Default steps | Allowed steps | Sampling |
| --- | --- | ---: | ---: | --- |
| `z-image-turbo` | MFLUX/MLX 4-bit | 9 | 1–12 | MFLUX Turbo defaults |
| `chroma1-hd` | stable-diffusion.cpp Metal, Q4_K_M | 40 | 1–80 | Euler, CFG 3 |
| `chroma1-flash` | stable-diffusion.cpp Metal, Q4_K_M | 8 | 1–40 | Heun, CFG 1, beta schedule, flow shift 1 |

Pinned Hugging Face sources and revisions are in `model_profiles.py`: [Z-Image Turbo](https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit), [Chroma1-HD](https://huggingface.co/silveroxides/Chroma1-HD-GGUF), [Chroma1-Flash](https://huggingface.co/silveroxides/Chroma1-Flash-GGUF), [shared T5 encoder](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf), and [VAE](https://huggingface.co/lodestones/Chroma1-HD). Native runtime release, download URL and checksums are pinned there too. Download scripts resume Hugging Face transfers; Chroma files and the runtime archive are SHA256-verified before recording completion. See upstream cards/licenses for usage terms.

Z-Image keeps a loaded instance; Chroma loads GPU weights per CLI generation and releases them afterward. Chroma can be substantially slower on this Mac. Native sampling progress, when available, appears in `/health` while busy. Resolution, reference preservation, and steps affect latency. Do not infer Mac performance from CUDA benchmarks.

## Generation API

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"z-image-turbo","prompt":"A fox in a snowy forest, soft morning light","size":"1024x1024","seed":42,"tags":["Wildlife","Winter"],"response_format":"url"}'
```

`model` and `steps` may be omitted to use the active model/default. Response fields include `created`, `generation_seconds`, and `data[0]` containing `seed`, `tags`, `size`, `width`, `height`, `generation_size`, **absolute `file_path` and `metadata_path`**, plus `url` or `b64_json`. These paths refer to the server's filesystem. PNG and JSON metadata are always saved, including for base64 responses. URLs remain resolvable after restarts/model switches while files and indexes remain available.

| Request field | Default / constraints |
| --- | --- |
| `prompt` | Required nonblank text; max 4000 characters. Z-Image processes the first 512 tokens including formatting and truncates excess. Put essential details first. |
| `size` | `1024x1024`; each dimension a multiple of 64, 256–2048. Large images need more time/memory. VAE tiling is used above 1024 for Z-Image and always for Chroma. |
| `steps` | Active model's default and range above; references require at least 2. |
| `seed` | Omit/null for random; integer 0–4294967295. Returned and saved. |
| `n` | Only 1. |
| `response_format` | `b64_json` or `url`; default `b64_json`. |
| `output_dir` | Default `<store>/<active-model>/images`; optional absolute folder or `~/...`. Created if writable; unique filenames never overwrite originals. |
| `reference_image` | Optional absolute local path to one Pillow-readable photo; HTTP URLs are not accepted. |
| `image_strength` | Default 0.6; 0 inclusive to 1 exclusive, only with a reference. Higher preserves more of the original and runs fewer denoising steps. 0 starts from noise. Chroma's native inverse denoising parameter is translated internally. |
| `lora` | Optional available ID from `/v1/loras`, otherwise null/base model. One adapter, compatible with the active family. |
| `lora_strength` | 0–2; requires `lora`; omitted uses catalog default. |
| `tags` | Up to 32 text labels, each 1–64 normalized characters; duplicates removed case-insensitively. No commas/control characters. |

Unknown fields and invalid combinations return 422. A busy server returns 429 with `Retry-After`; check `/health` before retrying. Invalid local paths/adapter selections return 400. Backend failures return 500 with details in local logs. This synchronous endpoint can take minutes; use a long client timeout (e.g. 1800 seconds). **Disconnecting or timing out does not cancel inference**; capacity remains held until it finishes. Check health and the gallery before retrying to avoid duplicate images. There is no queue, cancellation API, or idempotency guarantee.

Generation metadata records requested/effective dimensions, prompt, effective trigger prompt, active model/revision/backend, sampler settings, seed, steps, timing, reference path, adapter, initial tags, and output paths. The original file and generation record remain unchanged when tags are edited later.

## Reference photos and variations

The webpage accepts a photo upload or an absolute path. The API also accepts raw bytes:

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/reference-images \
  -H 'Content-Type: image/jpeg' --data-binary @/absolute/path/reference.jpg
```

The response contains an oriented PNG copy's `file_path`, preview `url`, `width`, and `height`. Uploads are limited to 20 MiB and 40 megapixels, stored separately in `<store>/<active-model>/references`, and retained for reuse. Pass the path in a generation request:

```json
{"prompt":"The same village at dusk, warm lanterns and mist","reference_image":"/absolute/path/reference.jpg","image_strength":0.6,"size":"1536x1536","response_format":"url"}
```

The oriented reference **aspect ratio is preserved automatically**. Only the larger dimension in `size` sets the output long edge. Padding to the backend grid is trimmed; returned dimensions can differ from the requested canvas and the short edge can be below 256. Originals are never modified. “Make a variation” uses a generated image as the next reference. This is image-to-image variation, not instruction-based localized editing or inpainting; exact edits/identity retention are not guaranteed.

## Searchable gallery, tags and pagination

The gallery combines generations across all three model folders, including indexed custom output destinations. References are excluded. It uses cached thumbnails, supports search by prompt/tags/model/adapter/seed/dimensions, sorting, editable tags, Previous/Next, page-size selection, full previews, settings and downloads. Reuse actions prefill the generator for review; they do not immediately generate. Images from another model may be used as references, but model-specific step/adapter settings use current defaults.

```sh
curl 'http://127.0.0.1:8000/v1/images?q=forest&tag=Winter&limit=24&offset=0&sort=newest'
curl 'http://127.0.0.1:8000/v1/images/0123456789abcdef0123456789abcdef'
curl --request PATCH 'http://127.0.0.1:8000/v1/images/0123456789abcdef0123456789abcdef/tags' \
  -H 'Content-Type: application/json' -d '{"tags":["Campaign","Village"]}'
```

`q` is a case-insensitive substring. Repeated `tag` parameters require all exact labels, case-insensitively; they combine with search. `limit` is 1–100 (default 24), `offset` is nonnegative, `sort` is `newest` or `oldest`. The response returns `total`, `limit`, `offset`, `has_previous`, `previous_offset`, `has_more`, `next_offset`, and `data`. Missing files are skipped; old/corrupt/missing sidecars do not prevent browsing valid PNGs. Tag edits are stored atomically in the owning model's `image-index`; `[]` clears tags. The index also resolves custom-folder images after restart. Back it up.

## Optional adapters

Default setup downloads no adapters. **Realism for Z-Image Turbo is the only bundled optional preset**:

```sh
make setup MODEL=z-image-turbo WITH_REALISM=1
make download-realism MODEL=z-image-turbo
make install-lora MODEL=z-image-turbo FILE=/absolute/path/adapter.safetensors LORA_ID=realism
```

Public downloads use only tracked `lora-catalog.json`, never personal catalogs. Compatible personal catalogs/checkpoints may live under ignored per-model `loras` folders; HD/Flash share `chroma1-hd/loras`. `GET /v1/loras` returns only the active model's entries and availability. The loader applies an adapter directly; a configured trigger phrase is prepended once automatically. Do not add `<lora:...>` tags to prompts. `--no-loras` / `NO_LORAS=1` hides the selector, returns an empty disabled catalog and rejects adapter requests. Adapter files and personal presets are never included in Git/default setup.

## Verification, recovery and AI handoff

`make test` runs fake-backend/storage/download tests without weights/GPU work. `make smoke` calls the running model, creates a real 512px image with its default steps, decodes it and rejects a blank result. This can be slow for Chroma. Health and gallery are responsive during inference. Native generation logs are under the active model's `logs/`.

If a download is interrupted, rerun `make setup MODEL=<id>` or `make download MODEL=<id>`; files are reused/resumed. An incomplete setup has no new completion marker. For a missing SSD, reconnect it; do not remove broken links or choose a fallback to hide the problem. A port conflict should be resolved with `PORT=...` or by stopping the identified service. Avoid multiple GPU services on a memory-constrained Mac. If setup cannot create symlinks, use a symlink-capable project filesystem or project-local storage. Local model paths and the native runtime are verified before inference; startup does not fetch network resources.

For another AI tool:

> Read README.md and skill/image-model-server/SKILL.md in this checkout. Run make setup for the desired model (default Z-Image Turbo), choosing the requested model store or Enter for local storage. Run make test. Start make serve MODEL=<id> in a retained session, wait for /health ready and verify the active model and serving mode. Report the UI URL. Preserve existing jobs and data; keep all model weights, media, tags, personal adapters, settings and symlinks outside Git.

No Codex plugin, original paths, shell aliases, or chat history is required. The service accepts local file paths and is intended for trusted local clients; it binds to `127.0.0.1` with host checks and no permissive CORS. Do not expose it to a network without an access-control design. The source-only repository ignores model folders, environments, runtime binaries, output/reference images, indexes, logs, credentials and local settings. Cleanup should preserve actual data unless explicitly requested.
