# Z-Image Turbo on Apple Silicon

Local MFLUX/MLX image generation with 4-bit Z-Image Turbo. This README is the setup and API entry point for humans and AI tools using a downloaded ZIP or Git checkout.

## Hand off to another AI tool

No chat history, Codex plugin, or preinstalled shell alias is needed. Give the AI the downloaded project folder and this instruction:

> Read README.md and skill/z-image-turbo/SKILL.md in this source folder. Set up this checkout with make setup, verify it with make test, and start the requested serving mode in a persistent terminal. Wait for /health to report ready and model z-image-turbo. Report the URL, mode, and verification results. Keep generated images, reference photos, weights, and runtime data outside Git.

The project root is the folder containing `Makefile`, `config.py`, and `server.py`, regardless of its name or location. In this document `<project>` means that folder's absolute path. Replace example paths with real paths on the machine running the server. The skill can be read directly by any AI tool; installation into a tool's skill directory is optional.

## One-time setup from source

Requires an Apple Silicon Mac with a working Metal GPU, Git, Make, and Python 3.14. On a Homebrew installation, install the prerequisite with `brew install python@3.14`. This project uses pinned packages for the tested Python 3.14 runtime. The existing 24 GB Mac setup uses a 4-bit checkpoint.

From the downloaded project folder (replace the path below):

```sh
cd "/absolute/path/to/downloaded-checkout"
make setup
make test
make serve
```

`make setup` does not start the server. `make test` verifies API/setup behavior with fake generators and fake downloads; it does not download weights or run GPU inference. `make serve` stays in the foreground, so an AI tool should retain that terminal/session and use another session for readiness checks. Startup loads the model before accepting requests.

After `Application startup complete`:

```sh
curl --fail --silent --show-error http://127.0.0.1:8000/health
```

Expect `status:ready`, `model:z-image-turbo`, and inspect `busy` and `loras_enabled`. Open `http://127.0.0.1:8000/` for the UI. For an optional actual GPU check, run `make smoke` from another terminal; it generates and validates a 512×512 PNG stored locally in the ignored images folder. Distinguish this from the fake-generator unit tests when reporting verification.

`make setup` creates the Python environment if missing, installs `requirements.lock.txt`, then downloads the base model from [filipstrand/Z-Image-Turbo-mflux-4bit](https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit) at revision `b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b`. Model files are roughly 6 GB and go directly into this checkout’s ignored `weights/` folder. Hugging Face resumes partial downloads and reuses completed files when setup is repeated. After setup, serving is offline; there is no model download during startup.

The default virtual environment is `~/Documents/models/z-image-turbo/.venv` on internal storage; this avoids symlink limitations on the exFAT SSD. Override it with `VENV=/absolute/internal/path/.venv` on **both** setup and serving commands. If Python is installed elsewhere, pass `BOOTSTRAP_PYTHON=/path/to/python3.14` to setup. `make setup-env` installs only Python dependencies, and `make download` runs only the base-model download after the environment exists.

LoRA adapters are optional and are not needed for base-model generation:

```sh
make setup WITH_LORAS=1  # Also fetch the pinned adapters
# Or, after normal setup:
make download-loras
# Serve with adapters disabled:
make serve-no-loras
```

The tracked `lora-catalog.json` contains only adapter names, source revisions and checksums. A fresh clone uses that catalog automatically; an existing local `loras/catalog.json` takes precedence. Downloaded adapter files and local catalog overrides are ignored by Git. The shell aliases below are conveniences on this Mac; `make serve` works without installing any alias.

## Setup recovery and completion checks

| Situation | Action |
| --- | --- |
| Wrong platform | This runtime targets Apple Silicon macOS with Metal; do not treat Intel, Linux, or Windows as a working MLX setup. |
| Bootstrap Python not found | Install Python 3.14 or set `BOOTSTRAP_PYTHON` to its executable. For a custom environment path, use the same `VENV` for setup, tests, serving, and downloads. |
| Dependency installation fails | Read the pip error. Keep the pinned versions; use a fresh internal-storage environment if an unrelated or older environment was selected. |
| Download interrupted or weights missing | Rerun `make setup`, or `make download` if dependencies are already installed. Hugging Face resumes existing downloads. Fix network/disk/access errors rather than creating `weights/download.json` manually. |
| Port already occupied | Inspect `/health`. The launcher may reuse a matching instance and can switch an idle managed server; it refuses to interrupt a busy generation or replace an unrelated service. Keep one GPU model instance on the original 24 GB Mac. |
| Server starting, health connection refused | Keep its terminal alive and wait for startup to finish; inspect its logs instead of starting duplicates. |
| Metal unavailable or memory error | Confirm Apple Silicon/Metal availability, inspect server logs, and reduce image size if memory is the cause. A passing unit suite alone does not establish working GPU inference. |

For setup-only work, report the project path, actual environment path, and setup/unit-test results. If serving was requested, also report a successful health check, the actual URL/port, and whether LoRAs are enabled. Report a GPU smoke result only if `make smoke` or a real generation ran. Do not restart an existing active generation merely to verify a documentation/setup change.

## What belongs in Git

This is a source-only repository. `.gitignore` excludes generated images, uploaded reference photos, image sidecars and URL records, model weights, downloaded LoRAs, caches, logs, virtual environments, credentials and macOS metadata. Image extensions are ignored throughout the repository, including inside source folders and with uppercase filenames. The UI’s small inline SVG icon is source markup; no separate image assets are tracked. Existing photos and model files remain local on disk.

Before committing, use `git status --short` and `git ls-files` to inspect the source list. Do not force-add ignored image or model files. A fresh checkout obtains its weights through `make setup`, rather than Git or Git LFS.

## Start and use

From the actual checkout root after setup:

```sh
make serve
```

The original Mac has `serve-z-image-turbo` and `serve-z-image-turbo-no-loras` aliases pointing to `/Volumes/ASH-SSD/models/z-image-turbo`. Those are optional conveniences specific to that installation. A clone/ZIP needs no alias or shell-profile changes; use its Makefile directly.

Wait for `Application startup complete`. Open http://127.0.0.1:8000/ for the webpage, http://127.0.0.1:8000/docs for interactive API docs, or http://127.0.0.1:8000/openapi.json for the machine-readable schema. The page supports prompts, sizes, seeds, steps, a visible reference-photo picker (with preview and remove controls), a reference file-path alternative, and a save folder. It previews the image and shows its saved path, with download, reuse-seed, and Make a variation buttons.

The original Mac's three model aliases switch between Z-Image Turbo, Chroma1-HD and Chroma1-Flash on the shared port 8000 when the current service is idle. If it is generating, the launcher refuses to interrupt it; retry after it finishes.

The server runs in the foreground; Ctrl-C stops it. `make start` is equivalent to `make serve`. `make serve PORT=8001` changes the port. The storage containing the checkout and weights must remain available. Do not start a second instance on this 24 GB Mac. The zsh alias is available in newly opened shells (or after `source ~/.zshrc`).

## Serve without LoRAs

From this project folder:

```sh
make serve-no-loras
# Or:
make serve NO_LORAS=1
```

Both pass `--no-loras` through `launch.py` to `server.py`. `PORT=8001` works with either Make target; also use it for `make health`, `make smoke`, and the client URL. A custom `VENV=...` must be supplied consistently too. This mode serves only the base model: the webpage hides adapter controls and displays a base-model notice, `GET /health` returns `loras_enabled:false`, and `GET /v1/loras` returns `enabled:false` with an empty `data` list. Requests with a non-null `lora` fail with HTTP 422 before inference. Omit `lora` or send null; reference photos and all other settings remain available.

Use `make serve` to restore the LoRA selector. The launcher restarts an idle managed server when the requested mode changes, including for the same model. It refuses to switch modes while an image is generating. Ordinary serving already uses the unmodified base model whenever **None** is selected; the new startup mode additionally prevents adapter use for the entire process.

## API contract

Base URL: `http://127.0.0.1:8000`. No API key is required. This is a small OpenAI-style images endpoint, not the full OpenAI API. It binds to localhost and serves offline. It has no prompt moderation layer.

`POST /v1/images/generations`, `Content-Type: application/json`:

| Field | Default | Contract |
| --- | --- | --- |
| `prompt` | required | Nonblank string, at most 4,000 characters. Runtime processes 512 tokens including chat formatting; excess text is truncated. |
| `model` | `z-image-turbo` | Only this model ID is accepted. |
| `size` | `1024x1024` | WIDTHxHEIGHT; each dimension 256–2048 inclusive, multiple of 64. With a reference, only the long edge is used; the photo determines the aspect ratio. |
| `steps` | `9` | Integer 1–12; at least 2 with a reference image. |
| `seed` | `null` | Integer 0–4294967295, or null/omitted for random. Actual seed is returned. |
| `n` | `1` | Only one image per request. |
| `lora` | null | Named adapter from `GET /v1/loras`; null/omitted disables adapters. |
| `lora_strength` | catalog default | Optional influence 0–2, only with `lora`; all installed adapters default to 0.8. |
| `response_format` | `b64_json` | `url` or `b64_json`; both save a PNG and metadata on disk. |
| `output_dir` | `<project>/images` | Absolute directory on this Mac; `~/` is expanded. Missing directories are created if writable. This changes only the current request. |
| `reference_image` | `null` | One absolute local image file path; `~/` is expanded. PNG, JPEG, WebP and other Pillow-readable formats. The webpage can upload a photo first and pass its saved local path here. No URLs, lists, masks, or multiple reference photos in this field. |
| `image_strength` | `0.6` | Number >=0 and <1, only valid with `reference_image`. Higher values preserve MORE of the source and perform fewer denoising steps (MFLUX semantics). 0 starts from full noise. |

Unknown fields are rejected. The output filename is a unique UUID, not a user-selected name. Existing images and the reference file are not overwritten. Paths refer to the computer running the server, not a remote caller's filesystem.


### Style LoRAs

The **Style LoRA** selector and API use the same catalog. The shipped choices are **None — original model** and the optional **Realism — suayptalha** (`realism`). One adapter per request. A separate **LoRA strength** field controls its influence; it is independent of reference-photo `image_strength`. The local starting strength is 0.8, configurable from 0 to 2. These starting defaults are local choices, not a claim that every adapter has the same optimal strength.

`GET /v1/loras` lists model-compatible entries with `available`, ID, default strength, source, repository revision, file size and SHA256. Only available entries can generate. Null or omitted `lora` means no adapter. Omitted/null `lora_strength` uses the catalog default. Unknown/unavailable/changed weights return 400 before inference; malformed fields or strength without an adapter return 422.

Example JSON for `POST /v1/images/generations`:

```json
{"model":"z-image-turbo","prompt":"A candid photograph of a fully clothed mechanic in a sunlit workshop, natural colours and realistic skin texture","size":"512x512","steps":9,"seed":42,"lora":"realism","lora_strength":0.8,"response_format":"url"}
```

The Realism trigger word is added automatically. Write scene descriptions in `prompt`; `<lora:...>` prompt tags are rejected. Sidecars retain the selected ID, applied strength, actual adapter path, checksum and source, plus `effective_prompt` with the trigger text. The Make a variation button retains the adapter and strength. The normal 512-token prompt limit includes any added trigger text.

MFLUX applies the adapter without baking it into the quantized base weights. Changing the selected adapter, changing its strength, or selecting None releases the previous model and reloads clean weights, with at most one GPU model resident. Repeated requests with the same selection reuse the loaded model. This adds some time when switching and avoids residual effects from the previous adapter. Model reload time is included in `generation_seconds`.

Files live in `<project>/loras/`; `lora-catalog.json` supplies the source defaults; a local `loras/catalog.json` overrides names and compatibility. Run `make download-loras` to download/resume the pinned catalogued adapters and verify their checksums. Already verified files are reused. Downloads occur separately from the offline server. To import another compatible adapter, add an entry to `lora-catalog.json` (or the local override) then run `make install-lora FILE=/absolute/path/to/adapter.safetensors LORA_ID=its-id`. Refresh the UI after catalog changes; a server restart is not needed.

Installed adapter sources and pins:

- Realism: https://huggingface.co/suayptalha/Z-Image-Turbo-Realism-LoRA , revision `8dc179cb56844d4ddc6e4527c30e98347eeae971`, `pytorch_lora_weights.safetensors` saved as `z-image-turbo-realism.safetensors`, 85,094,800 bytes. SHA256 `b38b074964f1f6564c22fee42125accd7a218a916966b5d0f0b2f63d4939c2f7`.


Realism was verified on this Mac at 512×512, nine steps, seed 42, strength 0.8 in 16.71 seconds including reload. This is an individual sample, not a controlled speed/quality benchmark against the original model.

### Larger resolutions

The UI offers 1536×1024, 1024×1536, 1536×1536 and 2048×2048 in addition to the smaller sizes. The API accepts any WIDTHxHEIGHT with each dimension a multiple of 64 between 256 and 2048. Default remains 1024×1024. These are native generations at the requested size, not a resized 1024 image.

Above 1024 on either side, the server automatically enables MFLUX VAE tiling for both reference encoding and output decoding. This reduces peak memory in that part of the pipeline; transformer denoising still runs at full resolution. Tiling may introduce subtle differences at tile boundaries. Larger sizes take longer and use more memory; close other GPU-heavy apps if needed. If a large image fails, try 1536 or 1024. The JSON sidecar records `vae_tiling` alongside the requested size.

### Prompt length

The 4,000-character API limit is distinct from the installed MFLUX tokenizer's 512-token sequence limit. Chat formatting consumes some of those tokens. Token counts depend on language and punctuation; a character count cannot guarantee the full prompt fits. MFLUX truncates longer sequences and logs a warning when the limit is filled. The API currently does not return a token count or truncation warning. Put the subject and essential composition first and keep prompts concise. The official Diffusers pipeline also defaults to 512 tokens: https://huggingface.co/docs/diffusers/api/pipelines/z_image . This is the configured runtime limit, not a claim that every possible implementation has the same architectural maximum.

### Generate from text

```sh
curl --fail-with-body --max-time 1800 http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A red fox in a snowy pine forest, soft winter light, wildlife photograph","size":"512x512","steps":9,"seed":42,"response_format":"url"}'
```

### Generate from a reference and choose where to save

Replace the example reference path with an existing photo on this Mac:

```sh
curl --fail-with-body --max-time 1800 http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A watercolor painting of a fox among snowy pine trees","size":"512x512","steps":9,"seed":42,"reference_image":"/absolute/path/to/reference.jpg","image_strength":0.6,"output_dir":"/absolute/path/to/output","response_format":"url"}'
```

This uses the photo as the starting image for image-to-image generation. Describe the desired final image. It is not a dedicated instruction editor or identity-preserving model. The output automatically follows the oriented reference aspect ratio, using the longer requested size dimension as its output long edge. Try 0.6 first; raise it to retain more of the source, lower it for more change. The step count is a full sampling schedule; reference preservation skips part of it, so actual denoising iterations are fewer and strength changes can be discrete. This convention is different from APIs where a higher denoising strength means more change.

## Automatic reference aspect ratio

With a reference photo, the output automatically follows its displayed aspect ratio (including camera EXIF orientation). `size` still accepts WIDTHxHEIGHT on the 64-pixel grid, but **only its longer dimension sets the output long edge**; its requested shape is ignored. For example, a 1600×900 reference with `size:1024x1024` produces a 1024×576 PNG, and a portrait reference produces a portrait PNG. The webpage switches to a **Reference resolution** control and previews the dimensions for picked photos. Local paths and **Make a variation** use the same rule.

The source is scaled proportionally, with the short edge rounded to the nearest whole pixel. Where needed, a temporary copy is padded to the model’s 64-pixel grid and minimum 256px canvas; that padding is trimmed after generation. No source content is cropped or stretched to fill the canvas. Saved reference outputs therefore need not have dimensions divisible by 64, and a very wide/tall reference can have a short edge below 256px. The original file is unchanged. Local reference paths, like uploaded photos, are limited to 40 megapixels.

Read `data[0].size`, `width`, and `height` for the **actual saved dimensions**, rather than assuming they equal the request. `generation_size` reports the internal model canvas. The JSON sidecar additionally records `requested_size`, actual `size`, `generation_size`, and oriented `reference_size`. Without a reference, requested dimensions work exactly as before.

### Choose a reference photo in the webpage

The **Reference photo** section appears directly below the prompt. Click **Choose photo**, pick one PNG/JPEG/WebP photo, and check its thumbnail. Use **Change photo** or the × remove button to replace or remove it. **Keep the original** controls preservation (default 0.6). **Or use a file path** accepts an existing absolute path without making an upload copy. Selecting a photo and typing a path are alternatives; the most recent choice is used.

The selected photo is uploaded only when you generate. It is saved as an oriented RGB PNG copy in `<project>/references/`; the original is unchanged. The browser reuses that saved reference for subsequent generations until you choose another photo or reload the page. Copies remain in the project for reuse, including when a generation fails. Removing a photo in the form clears its selection, not its saved copy. There is no cloud upload. Maximum upload: 20 MiB and 40 megapixels. PNG, JPEG and WebP are recommended; HEIC is not supported by the installed decoder.

After generating, click **Make a variation** under the result. The existing PNG becomes the reference (no extra upload), with the previous prompt selected for rewriting and preservation reset to 0.6. Describe the desired final image, adjust **Keep the original**, and generate again. Each pass saves a new PNG/JSON; earlier versions stay on disk. This is whole-image image-to-image refinement, so small edits may alter other details. Masked inpainting and exact instruction edits are not exposed by this model wrapper.

Agents may also upload a reference via `POST /v1/reference-images` with **raw image bytes**, not multipart or JSON:

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/reference-images \
  -H 'Content-Type: image/jpeg' \
  --data-binary '@/absolute/path/to/reference.jpg'
```

The 201 response contains `file_path`, `url`, `width`, and `height`. Pass its `file_path` as `reference_image` in the normal generation request. The URL serves the normalized reference PNG while the server is running. Invalid images return 400; oversized files/images return 413. This upload route does not generate an image or load another model.

### Response and files

Example shape (illustrative values):

```json
{
  "created": 1792886400,
  "data": [{
    "seed": 42,
    "size": "512x512",
    "width": 512,
    "height": 512,
    "generation_size": "512x512",
    "file_path": "/absolute/path/to/output/0123456789abcdef0123456789abcdef.png",
    "metadata_path": "/absolute/path/to/output/0123456789abcdef0123456789abcdef.json",
    "url": "http://127.0.0.1:8000/images/0123456789abcdef0123456789abcdef.png"
  }],
  "generation_seconds": 19.02
}
```

Use the actual `data[0].file_path` and `metadata_path`, not a constructed filename. With `b64_json`, `data[0].b64_json` replaces `url` and contains base64 PNG bytes (no data-URL prefix). Both absolute paths are always returned. The JSON sidecar records prompt, actual seed, size, requested steps, model source/revision, elapsed seconds, reference path/strength, and output paths. Reference pixels are not copied into metadata.

The server keeps `image-index/<UUID>.json` records in the project so custom-directory image URLs keep working after restart. The server must be running, and the saved image and project storage must remain accessible. Moving or deleting a PNG breaks its URL. Images created before this index feature remain available in the default image directory.

### Readiness, errors, and retries

- `GET /health`: `{"status":"ready","model":"z-image-turbo","busy":false,"loras_enabled":true}` when usable. Startup loads the model before accepting requests; connection refusal during startup is expected.
- `GET /v1/models`: available model ID.
- `GET /images/{UUID}.png`: fetch an existing output, or 404 if absent.
- HTTP 400: reference cannot be read/decoded, or destination cannot be written. Fix the path.
- HTTP 422: invalid request (relative path, unsupported size, invalid seed/steps, extra field, strength without reference). Read `detail` for the field error.
- HTTP 429: another generation is running; wait until `/health` reports `busy:false`. `Retry-After: 10` is supplied.
- HTTP 500: generation failed; inspect server logs, consider a smaller size if memory is exhausted.

One generation is allowed at a time; there is no job queue. Allow several minutes (the examples use a 30-minute client timeout). A client disconnect or timeout does not cancel generation. Do not automatically resubmit after a timeout: check health, then inspect the requested output directory for the new PNG/JSON first to avoid duplicate work. This API has no cancellation or job-status endpoint.

## AI agent skill

The portable skill is `skill/z-image-turbo/SKILL.md`, with optional UI metadata in `skill/z-image-turbo/agents/openai.yaml`. It covers project discovery, prerequisites, first-time setup, verification, server startup, references, adapters, output paths, and retry behavior. Any AI tool can read it as a normal Markdown instruction file; no Codex-specific tool is required to set up or call this service.

If the tool supports installable skills, install/copy the entire `skill/z-image-turbo/` folder using that tool's normal skill installation mechanism. If it does not, ask it to read the file directly as shown in the handoff prompt above. A globally installed copy must locate the actual checkout before running commands; its own skill folder is not the server project. On the original Mac a copy is already installed at `~/.codex/skills/z-image-turbo/`. Once serving, consult `/openapi.json` for the live schema rather than assuming another installation has identical fields.

## Underlying model controls

The API exposes named compatible LoRA adapters and their strength as described above. Additional MFLUX capabilities include alternate denoising schedulers, weight quantization, compute precision, and VAE tile sizes; those are backend settings rather than extra generation-request fields. Adapter changes reload clean weights automatically. Changing quantization/compute precision requires suitable weights and a model reload. Our weights are pinned at 4-bit; switching to 8-bit or full precision requires suitable weights and more memory. The Turbo runtime forces CFG guidance to 0 and skips negative-prompt encoding, so exposing those controls would have no effect. The default denoising schedule here is MFLUX `linear`.

ControlNet structural guidance requires a compatible additional ControlNet model and integration; the normal reference-photo feature uses image-to-image initialization. PiD decoding is another optional runtime feature requiring a separate decoder; it is not enabled. See https://github.com/mflux-community/mflux/blob/main/src/mflux/models/z_image/README.md for runtime capabilities.

## Storage and reproducibility

- Project, weights (`weights/`), caches (`cache/`), default images (`images/`), uploaded reference copies (`references/`), and URL records (`image-index/`) are under `<project>` (derived from the directory containing `config.py`).
- Default Python environment: `~/Documents/models/z-image-turbo/.venv` on internal storage. This also works when the project is on an exFAT SSD, where environment symlinks are unsuitable. Override with `VENV=...` when needed.
- Original model: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo (Apache 2.0).
- Quantized distribution: https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit . Its own repository metadata/model notes are retained in `weights/README.md`.
- Pinned revision: `b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b`.
- Runtime: MFLUX 0.23.0; exact dependency versions in `requirements.lock.txt`.

To reinstall: `make setup`, then `make serve`. Setup includes the base-model download; `make download` remains available separately. Downloads resume into `weights/`, avoiding a second full copy in the home-directory cache. Downloads are the explicit network step; serving is offline. `make test` checks API behavior without loading weights; `make smoke` generates and validates a 512px image against a running server.

On this Apple M6 Mac with 24 GB unified memory and macOS 27.0.1, initial real API tests took 19.02 seconds for 512×512 and 53.73 seconds for 1024×1024 (9 steps, seed 42). A native 2048×2048 run with automatic VAE tiling took 317.1 seconds (5 minutes 17 seconds), also at 9 steps and seed 42. These are individual runs, not a general benchmark; startup is additional. Start at 512×512 or 768×768 if memory is tight. Pillow is pinned to 12.3.0 because 13.0.0 failed to import with the installed Python 3.14 runtime.
