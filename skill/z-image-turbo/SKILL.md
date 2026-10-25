---
name: z-image-turbo
description: Set up and use the local Z-Image Turbo server from a downloaded or cloned source folder on an Apple Silicon Mac, including image generation, reference-photo variations, and custom output folders. Use when the user chooses this local server.
---

# Z-Image Turbo: setup and local generation

This skill works from a downloaded source folder as well as the original Mac installation. Any AI tool can read it directly; a Codex installation or shell alias is not required.

## Locate the project

Use the checkout supplied by the user or the current working directory. Confirm it contains `Makefile`, `README.md`, `config.py`, `server.py`, and `requirements.lock.txt`. Inside the repository this skill is at `skill/z-image-turbo/SKILL.md`, two directories below the project root. If the skill was installed separately, locate the actual checkout first; the installed skill folder is not the server project.

Read the project's `README.md` for detailed setup and the API contract. Paths such as `<data>/images` mean this checkout's absolute path, derived by `config.py`; they are not literal request values. The original serving project is `~/Documents/models/z-image-turbo`, with data under `/Volumes/ASH-SSD/models/z-image-turbo`, but another machine or checkout need not use that path. Use `make` from the actual project root rather than relying on the original Mac's aliases.

On this Mac, source/Git/UI/environment belong under `~/Documents/models/z-image-turbo` and the model/data symlinks point to `/Volumes/ASH-SSD/models/z-image-turbo`. For other users, a fresh `make setup` prompts for a model/data folder in a terminal: blank means the project itself; choosing another absolute path creates symlinks. The choice is saved in ignored `storage.json` and reused on retries/serving. For an AI/script, use `make setup NONINTERACTIVE=1` for saved/project-local storage or `make setup DATA_DIR=/absolute/path` for an explicit location. `make storage DATA_DIR=...` configures links only. Existing nonempty folders/different links are preserved and reported as conflicts. `<data>` means the selected folder, `PROJECT_ROOT` is source, and runtime `ROOT` is data. External mounts must be present; never create a replacement mount folder or download an internal fallback. Keep `VENV` overrides consistent. Existing artifact paths are preserved.

## First-time setup

The runtime requires macOS on Apple Silicon (`Darwin`, `arm64`) with an available Metal GPU. Git, Make, and Python 3.14 are prerequisites. The default bootstrap Python is `/opt/homebrew/bin/python3.14`; Homebrew users can install it with `brew install python@3.14`. If the platform or prerequisite is missing, identify it before attempting the MLX installation.

From the project root:

```sh
make setup
make test
```

`make setup` creates the environment if missing, installs pinned dependencies, and downloads the pinned Hugging Face base checkpoint into `<data>/weights/`. It needs network access and space for roughly 6 GB of weights plus the Python runtime and dependencies. Downloads resume and reuse completed files on retries. `make test` uses fake generators and downloads; it does not require model weights or a running GPU server.

The default environment is `<project>/.venv` on internal storage (`~/Documents/models/z-image-turbo/.venv` on this Mac). For another location, pass the same `VENV=/absolute/internal/path/.venv` to every Make command. Keep the serving checkout, Git and environment internal; weights and runtime data stay on the external SSD. If needed, set `BOOTSTRAP_PYTHON=/path/to/python3.14` during setup. `make setup-env` installs only dependencies; `make download` downloads only the base model once the environment exists. Never fabricate `weights/download.json` to bypass an incomplete download.

Setup installs only the base model by default. Realism is the only shipped optional adapter: if requested, use `make setup WITH_REALISM=1` or `make download-realism` after setup. The downloader uses only the tracked `lora-catalog.json`; it never reads personal `loras/catalog.json` entries. Local catalog overrides affect serving only and are not included in Git or setup. Images, reference photos, weights, adapters, caches, logs, and runtime records must stay ignored by Git; do not force-add them.

## Start and verify

Start the requested mode in a retained terminal/session:

```sh
make serve
# Base model only, with adapters disabled for the entire server:
make serve-no-loras
```

The default port is 8000. Use `PORT=8001` consistently with serving, health checks, smoke tests, and API URLs if another port is chosen. If using a custom environment, include `VENV=...` as well. Keep only one model instance on the original 24 GB Mac.

Wait for `Application startup complete`, then check:

```sh
curl --fail --silent --show-error http://127.0.0.1:8000/health
```

Require `status:ready` and `model:z-image-turbo`; inspect `busy` before submitting work. Connection refusal during model loading is expected. `loras_enabled:false` identifies base-only mode. Existing matching services can be reused. The launcher refuses to interrupt a busy generation or replace an unrelated service; do not bypass that refusal by killing a process. The generator webpage is `/`, saved-image gallery `/gallery`, interactive docs `/docs`, and `/openapi.json` is authoritative for the running instance.

For an optional real GPU generation check, run `make smoke` after readiness. It generates and validates a 512×512 PNG, saved locally outside Git. Report setup/test results, the serving URL and mode, and any actual generated file paths. Do not claim a GPU smoke test ran when only `make test` was used.

## Browse saved generations

Use `/gallery` to search saved outputs, inspect prompts/settings and file paths, edit searchable image tags, download PNG/JSON, or pre-fill a variation/reuse form. The library includes default-folder UUID PNGs and indexed custom destinations, excluding reference uploads. Legacy files without readable metadata still appear. The page has Previous/Next controls and a page-size selector; its URL preserves search/filter/page state. Refresh after a generation; browsing uses CPU thumbnails and can run while inference is busy.

Programmatic access: `GET /v1/images?limit=24&offset=0&sort=newest&q=castle`. `limit` is 1–100, `offset` nonnegative, `sort` newest/oldest, and optional `q` searches prompts, seeds, dimensions, models, tags and adapter labels. Repeated `tag` parameters filter by all exact case-insensitive labels and combine with `q`. Read `data`, `total`, echoed `limit`/`offset`, `has_previous`/`previous_offset`, and `has_more`/`next_offset`; reuse the same filters and limit when navigating; use each item's returned `id`, `file_path`, `url`, `thumbnail_url`, and optional `metadata_url`. `GET /v1/images/{id}` adds saved `metadata` when available. Fields absent from old sidecars are omitted; timestamps fall back to PNG modification time. An individual missing file returns 404; unavailable project storage returns 503. Read the live OpenAPI schema for full fields.

Set initial `tags:["castle","campaign"]` during generation. Edit later with `PATCH /v1/images/{id}/tags` and JSON `{"tags":["castle","campaign"]}`; send `[]` to clear. At most 32 labels of 1–64 characters; whitespace/Unicode are normalized and case-insensitive duplicates removed. No commas/control characters. Current tags live in ignored `image-index/<id>.json`; preserve this index in backups. PNGs and original sidecars are unchanged by edits; their initial tags can differ from the current top-level `tags` in list/detail responses. Invalid edits return 422; missing images 404; storage failures 503. The last successful replacement wins.

Reusing gallery settings does not start generation, guarantee an identical output, or install a missing adapter. Respect the current serving mode; an original reference must still exist if its path is reused. Keep outputs, sidecars, index records, and `cache/thumbnails/` out of Git. Deleting the thumbnail cache only removes previews; it never deletes originals.

## Generate and use references

POST JSON to `/v1/images/generations`, for example:

```json
{"model":"z-image-turbo","prompt":"A small red fox resting beneath snowy pine trees, soft winter light","size":"512x512","steps":9,"response_format":"url"}
```

Only `n:1` is supported. Requested dimensions are multiples of 64, 256–2048 per side; default size is 1024×1024. Above 1024, VAE tiling is automatic. Seed is an optional unsigned 32-bit integer. The API accepts 4,000 characters, but the tokenizer truncates at 512 tokens including formatting; essential details go first.

For a reference, pass one existing absolute local `reference_image` path and `image_strength:0.6`. Higher strength preserves MORE of the source; valid range is 0 inclusive to 1 exclusive, with at least two steps. Output follows the EXIF-oriented source aspect ratio automatically: the longer requested size dimension sets its long edge. Model padding is trimmed, so saved dimensions can differ from the request. Read returned `size`, `width`, and `height`. This is whole-image variation, not masked inpainting, exact instruction editing, or guaranteed identity preservation.

The webpage accepts a photo picker or local path. Programmatic upload uses raw PNG/JPEG/WebP bytes, not multipart, at `/v1/reference-images` (20 MiB and 40 MP maximum); use the returned `file_path` in the generation request. Reuse a generated image's actual `file_path` as the next reference for refinement.

Default outputs go to `<data>/images`. Optional `output_dir` must be an absolute directory on the server's Mac (`~/` is supported). Filenames are unique PNG/JSON pairs. Read `data[0].file_path`, `metadata_path`, and `seed`; show the verified saved PNG with its absolute path when supported. Do not invent filenames or assume another checkout's output path. URLs require the service and saved file to remain available.

For adapters, inspect `GET /v1/loras`: choose an `available:true` ID only when `enabled:true`. Send `lora:<id>` and optional `lora_strength` (0–2, catalog defaults currently 0.8); omit/null disables adapters. One per request. Required trigger text is added automatically; `<lora:...>` prompt tags are rejected. Base-only mode rejects non-null adapters with 422. Adapter or strength changes reload clean model weights.

Allow a long client timeout (up to 30 minutes for large jobs). On 429, wait for `busy:false`. On 400/422, correct the reported input. A timeout/disconnect does not cancel generation: inspect health and the destination for a new PNG/sidecar before retrying. On 500, inspect server logs and memory pressure rather than repeatedly submitting the same job. See README for setup recovery and the complete schema.
