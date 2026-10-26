# ImageGen-Service

One Apple Silicon project for **Z-Image Turbo**, **Chroma1-HD**, and **Chroma1-Flash**, with a shared webpage, API, and searchable image gallery. Select a model when starting the generator; one backend runs at a time. A separate lightweight queue API accepts jobs even while the generator is off. Source repository: [ImageGen-Service](https://github.com/ashwinmahesh/ImageGen-Service).

## UI preview

Generate from a prompt or reference photo, choose immediate priority or ordinary FIFO, and follow saved jobs across reloads.

![Generator with reference-photo and image controls](docs/screenshots/generator.png)

Inspect, pause, remove or clear waiting jobs without interrupting the active image.

![Persistent generation queue and controls](docs/screenshots/queue.png)

Search saved images by prompt or tags, browse pages, and open an image to reuse its settings or edit its tags. The three previews above and below use demonstration data and a simulated backend; displayed timings are not performance benchmarks.

![Searchable tagged image gallery](docs/screenshots/gallery.png)

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

`DATA_DIR` means the **store containing model folders**, not one model's weights folder. On the original Mac, the single serving project is `~/Documents/models/ImageGen-Service`, and the store is `/Volumes/ASH-SSD/models`. Existing model, output, tag and custom-location index paths are reused in place. A fresh clone needs no external SSD or original user's home path. Use the consolidated project for all image models. The former local source projects `~/Documents/models/z-image-turbo`, `chroma1-hd` and `chroma1-flash` have been retired on the original Mac; their same-named SSD model/data folders remain active and must be preserved. The shared project owns its environment and native runtime without depending on the retired source folders.

Chroma HD and Flash share their T5 encoder and VAE in `chroma1-hd/weights`, and one native runtime in `<project>/runtime`. **Flash setup does not download the HD diffusion checkpoint.** Back up the actual SSD data, including indexes and tags, **and the project-local `state/` folder**; symlinks are not backups.

## Start, switch and stop

`make queue` starts or reuses the lightweight queue API in the background at **http://127.0.0.1:55302**; it returns once ready. It needs the installed Python environment, but no model weights, LLM, SSD or GPU. Logs are in ignored `state/queue-api/server.log`. No login/autostart service is installed.

`make serve` ensures that same queue API is running, then runs the selected **generator and UI** in the foreground; keep its terminal open. Ctrl-C finishes active generation and stops the generator. **The queue API stays running**, and other agents can keep submitting jobs. `make stop-queue` stops only the verified queue process and retains every job. The generator reads/checkpoints SQLite directly, so previously accepted jobs can finish even if the queue HTTP process stops; UI requests still read/write SQLite through the generator API while the standalone queue API is down; only clients using its separate origin need it restarted. AI tools should retain the terminal session and use another session for checks. When an AI tool starts a temporary server for maintenance/testing, it should stop that server when checks finish unless asked to leave it running. On the original Mac the user starts ImageGen manually; leave agent-started test instances stopped. Preserve pre-existing user-started services and active generation.

- Webpage: <http://127.0.0.1:5732/>
- All-model image gallery: <http://127.0.0.1:5732/gallery>
- Generator API docs and live schema: <http://127.0.0.1:5732/docs> and `/openapi.json`
- Queue-only API docs: <http://127.0.0.1:55302/docs> (available without generation)
- Readiness: `make health` or `GET /health`

The default is **`http://127.0.0.1:5732/`**, also reachable as `http://localhost:5732/`. `PORT=...` changes the port for serving, health and smoke commands; use the same override for each command. `QUEUE_PORT=...` changes the independent queue API port; use the same value for `make queue`, `make serve` and `make stop-queue`. The Python generator uses `IMAGEGEN_QUEUE_PORT` when invoked directly. Both services bind to loopback. No hosts-file mapping or `http://imagegen` shortcut is installed.

Health reports `status`, active `model`, `busy`, `project_dir`, active `data_dir`, `loras_enabled`, `prompt_expansion`, model `capabilities`, and durable `queue` counts/state/database path. `GET /v1/models` lists all supported models, download markers and the active one. A download marker indicates completed setup, not a full startup integrity check; startup verifies required local assets too. The UI reads model identity, step defaults and limits from health.

To switch, stop the server and run `make serve` again, or run `make serve MODEL=...` in another terminal. The launcher reuses an identical instance or gracefully replaces a **verified managed idle** service. It refuses to interrupt generation or replace unrelated listeners. It never hot-swaps models inside an inference request. A request's `model`, if specified, must equal the active model; mismatches return 422. Serving never downloads models. If assets are missing, run `make setup MODEL=<id>` first.

If the launcher finds the same model/mode already running from this checkout, it prints the URL and exits successfully; the original terminal still owns that server. A model switch leaves its waiting queue persisted until that model starts again. During initial model loading the HTTP listener may not yet be reachable; connection refusal can be normal until startup completes. Read logs if readiness never arrives.

The original Mac's shortcuts `serve-images`, `serve-z-image-turbo`, `serve-chroma1-hd`, `serve-chroma1-flash` and the three `-no-loras` variants all use this one project. Other users can use Make directly without aliases.

## Local prompt expansion (enabled by default)

First `make setup` also downloads **[Qwen3.5-2B 4-bit](https://huggingface.co/mlx-community/Qwen3.5-2B-4bit)** (Apache-2.0) to `<store>/prompt-enhancer/weights`. All image models share it; repeated setup reuses the files. Revision `674aaa7240b91e8012fcad5d791b7dfe5ba90207` and the weights SHA256 are pinned in `enhancer_profile.py`. The text-only loader uses pinned `mlx-lm==0.32.2`. Serving is offline. For an existing installation, run `make setup MODEL=z-image-turbo` to install the new dependency, or use `make download-llm` if dependencies are already installed.

```sh
make serve MODEL=z-image-turbo              # LLM defaults on; terminal asks [Y/n]
make serve MODEL=z-image-turbo NO_LLM=1     # Explicitly disable the LLM
make serve MODEL=chroma1-flash NO_LORAS=1    # Same default LLM mode, no adapters
make download-llm                           # Download/resume shared LLM only
```

Prompt expansion capability is **enabled by default** for every startup model. In a terminal, startup asks `Enable local LLM prompt expansion? [Y/n]:` after selecting the model: Enter/Y keeps it on, N explicitly turns it off for that run. Without a terminal (or with `NONINTERACTIVE=1`), omission enables it without prompting. `NO_LLM=1` is the only LLM startup opt-out and skips that question. These choices are independent of adapters and model selection, and an interactive N is not saved as the next run's default. The Make setting maps to `--no-llm` in the Python launchers; `server.py` defaults on while `launch.py` offers the same terminal question.

When enabled, prompt assets are checked **before** replacing an idle managed service. Missing/incomplete LLM assets fail clearly with `make download-llm`/setup guidance and the explicit `NO_LLM=1` alternative; startup never silently disables the LLM. Explicitly disabled serving does not require its weights. The LLM still loads only in a short-lived child when expansion/planning is requested, not as a second permanently resident startup model. `/health.prompt_expansion.enabled` and `/health.capabilities.prompt_expansion` describe the current mode. Switching modes requires restarting; the launcher refuses a busy switch.

The LLM has explicit image-specific instructions to expand the description using evocative, concrete visual language. It develops the requested medium/style/look/aesthetic through compatible line work, shading, palette, materials, atmosphere, lighting, framing and depth. It is told to preserve subjects, counts, actions, colors, composition, quoted text and exclusions, and to avoid conflicting styles or generic quality filler. **Your original text is kept verbatim**, followed by added visual detail, with style development first. The LLM returns separate `style_details` and optional `additional_composition` fields; their accepted text is combined after the original. See [research and policy decisions](docs/prompt-expansion.md). This is expansion, with no compression pass. Check the preview when exact creative choices matter: model-generated additions can still be imperfect. The LLM sees only text and does not analyze reference photos.

When enabled at startup, the webpage shows **Expand description with local LLM** (initially checked). Turn it off for any image. **Preview expansion** returns the expanded text without creating an image job; **Use this description** copies it into the editable prompt and turns automatic expansion off to avoid expanding it twice. Generating with expansion checked saves original and expanded prompts together. Accepting a preview manually makes that edited text the new request prompt. Gallery settings reuse an existing expanded prompt with automatic expansion off.

API generation routes accept common strict boolean **`expand_prompt`**, default **false**, independently of priority:

```sh
curl -sS http://127.0.0.1:5732/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Three cloaked adventurers in a misty gothic village, old-school cel animation, no text","expand_prompt":true,"generate_immediately":true,"size":"512x512","response_format":"url"}'

# Text preview only; returns original_prompt, expanded_prompt, model/revision, seconds and token counts.
curl -sS http://127.0.0.1:5732/v1/prompts/expand \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A fox in a snowy forest, watercolor"}'
```

The preview optionally accepts `lora`/`lora_strength` so token counts include its configured trigger. It returns409 while the shared accelerator/another inference is busy; retry after it finishes. It does not create a durable job. Expansion requests while the feature is disabled return422. Automatic expansion runs as part of the durable image job, under the shared accelerator lease; its result is committed to SQLite **before** rendering and reused after an interrupted render. Restarting without LLM mode preserves waiting expansion jobs at the head until that capability is enabled or the job is removed. Failed expansion marks that job failed without rendering or silently using the original; inspect the saved error and retry or submit without expansion.

Expanded text must fit the API's 4000-character cap. For Z-Image, the exact local image tokenizer checks the **512-token** budget including chat formatting and adapter trigger. Almost-full original prompts and over-budget expansions fail clearly; neither is compressed or truncated by this feature. Chroma's native GGUF T5 tokenizer is not counted by the Python preview; token counts/limit are null there, and its backend prompt limits still apply. Guidance is selected automatically for Z-Image, Chroma HD or Chroma Flash. `/health.capabilities["<model-id>"].prompt_expansion_guidance` reports the profile; Flash currently inherits HD content guidance because fewer diffusion steps alone do not establish different prompting needs. Records include `image_model`, `guidance_profile`, `guidance_version`, `policy_version`, `policy_sha256`, `sampling` and structured `details`. The LLM uses greedy non-thinking decoding, at most 320 generated tokens per expansion and a 180-second total timeout. Varied batches load the LLM once, then plan each option in that child. It runs in a short-lived subprocess; its weights/KV cache are released before image rendering. The selected image backend can remain resident.

Shared-expansion job detail/list responses expose `prompt_expansion` once available. Varied batches expose ordered `candidate_expansions[]` instead; each completed image carries its own `prompt_expansion`. Each completed `result.data[i].prompt_expansion` or synchronous `data[i].prompt_expansion` records the expansion actually used for that option. The sidecar keeps `prompt` as the original, `expanded_prompt` as expanded text, `effective_prompt` as actual expanded text plus any adapter trigger, and `prompt_expansion` with provenance/token counts/timing. `generation_seconds` measures image rendering; `prompt_expansion.seconds` is additional latency. Default non-expanded requests keep their legacy idempotency fingerprints. Optional `peak_mlx_memory_gb` and `peak_process_rss_gb` report decimal GB for the LLM child; these overlapping counters must not be added. In one local test, expansion took8.9s and peaked at1.96GB process RSS (including1.63GB MLX allocation). The two512px,9-step same-seed images rendered in14.8s and14.2s, respectively; these are individual measurements, not general performance guarantees.

## Models and effective settings

| Startup ID | Runtime/checkpoint | Default steps | Allowed steps | Sampling |
| --- | --- | ---: | ---: | --- |
| `z-image-turbo` | MFLUX/MLX 4-bit | 9 | 1–12 | MFLUX Turbo defaults |
| `chroma1-hd` | stable-diffusion.cpp Metal, Q4_K_M | 40 | 1–80 | Euler, CFG 3 |
| `chroma1-flash` | stable-diffusion.cpp Metal, Q4_K_M | 8 | 1–40 | Heun, CFG 1, beta schedule, flow shift 1 |

Pinned Hugging Face sources and revisions are in `model_profiles.py`: [Z-Image Turbo](https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit), [Chroma1-HD](https://huggingface.co/silveroxides/Chroma1-HD-GGUF), [Chroma1-Flash](https://huggingface.co/silveroxides/Chroma1-Flash-GGUF), [shared T5 encoder](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf), and [VAE](https://huggingface.co/lodestones/Chroma1-HD). Native runtime release, download URL and checksums are pinned there too. Download scripts resume Hugging Face transfers; Chroma files and the runtime archive are SHA256-verified before recording completion. See upstream cards/licenses for usage terms.

Only request fields listed below are accepted. Negative prompts, guidance/CFG, custom samplers/schedulers, masks and instruction-editing controls are not exposed as request options; Chroma sampling values above are fixed by its backend profile. No model-specific API option namespace is implemented yet.

Z-Image keeps a loaded instance; Chroma loads GPU weights per CLI generation and releases them afterward. Chroma can be substantially slower on this Mac. Native sampling progress, when available, appears in `/health` while busy. Resolution, reference preservation, and steps affect latency. Do not infer Mac performance from CUDA benchmarks.

## Generation API

```sh
curl --fail-with-body http://127.0.0.1:5732/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"z-image-turbo","prompt":"A fox in a snowy forest, soft morning light","size":"1024x1024","seed":42,"tags":["Wildlife","Winter"],"response_format":"url"}'
```

`model` and `steps` may be omitted to use the active model/default. Response fields include `created`, `generation_seconds`, and `data[]` items containing `seed`, `tags`, `size`, `width`, `height`, `generation_size`, **absolute `file_path` and `metadata_path`**, plus `url` or `b64_json`. These paths refer to the server's filesystem. PNG and JSON metadata are always saved, including for base64 responses. URLs remain resolvable after restarts/model switches while files and indexes remain available.

| Request field | Default / constraints |
| --- | --- |
| `prompt` | Required nonblank text; max 4000 characters. Z-Image processes the first 512 tokens including formatting and truncates excess. Put essential details first. |
| `size` | `1024x1024`; each dimension a multiple of 64, 256–2048. Large images need more time/memory. VAE tiling is used above 1024 for Z-Image and always for Chroma. |
| `steps` | Active model's default and range above; references require at least 2. |
| `seed` | Omit/null for random; integer 0–4294967295. Returned and saved. |
| `n` | Integer 1–5, default 1. Sequential image options for the same prompt; each gets a distinct seed. |
| `vary_prompt` | Boolean, default false in API. Requires `expand_prompt:true` and `n>=2`; distinct saved LLM composition expansions per option. The UI enables it by default for eligible batches. |
| `generate_immediately` | UI defaults true. true runs next, ahead of ordinary FIFO jobs; equal-priority jobs remain FIFO. Never preempts the active image. POST `/v1/jobs` defaults false; the synchronous compatibility endpoint defaults true. |
| `response_format` | `b64_json` or `url`; default `b64_json`. |
| `output_dir` | Default `<store>/<active-model>/images`; optional absolute folder or `~/...`. Created if writable; unique filenames never overwrite originals. |
| `reference_image` | Optional absolute local path to one Pillow-readable photo; HTTP URLs are not accepted. |
| `image_strength` | Default 0.6; 0 inclusive to 1 exclusive, only with a reference. Higher preserves more of the original and runs fewer denoising steps. 0 starts from noise. Chroma's native inverse denoising parameter is translated internally. |
| `lora` | Optional available ID from `/v1/loras`, otherwise null/base model. One adapter, compatible with the active family. |
| `lora_strength` | 0–2; requires `lora`; omitted uses catalog default. |
| `tags` | Up to 32 text labels, each 1–64 normalized characters; duplicates removed case-insensitively. No commas/control characters. |

Unknown fields and invalid combinations return 422. Invalid input/adapter selections return 400; unavailable queue storage/worker or admission storage failures return 503. Base-only mode rejects new adapter requests with 422. A saved failed job remains readable with HTTP 200 and `status:"failed"`; the synchronous generation endpoint instead returns its generation error status. The synchronous compatibility endpoint waits for its persisted job and returns the existing image response; backend failures surface there with the job error. Use a long timeout (e.g. 1800 seconds). **Disconnecting or timing out does not cancel a persisted job**. Prefer the asynchronous job API below for promptly returned IDs and polling, and reuse an `Idempotency-Key` after uncertain submissions. Both routes use the same queue worker; a busy worker accepts pending work without starting overlapping inference.

Generation metadata records requested/effective dimensions, prompt, effective trigger prompt, active model/revision/backend, sampler settings, seed, steps, timing, reference path, adapter, initial tags, and output paths. The original file and generation record remain unchanged when tags are edited later.

## Generate several image options

Choose **Number of images** in the webpage: 1–5, default 1. The images share your original description, model, size, reference and adapter settings, and render sequentially. With LLM expansion enabled, **Different compositions with LLM** is checked by default for 2–5 options: each gets its own expansion, filling only unspecified framing/layout/depth/light choices while preserving the stated subjects and style. Uncheck it to use one shared expansion, or disable expansion for seed-only alternatives. Without the LLM, the control explains how to enable it: `make serve` (keep the LLM enabled). Each option has its own PNG, sidecar and gallery entry. Select an option's thumbnail to preview it, download it, reuse its seed or make a variation.

```sh
curl -sS http://127.0.0.1:5732/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A watercolor fox beneath an oak tree","n":3,"seed":42,"response_format":"url"}'
```

For stronger variety through the API, send `"expand_prompt":true,"vary_prompt":true,"n":3` to either generation route while serving with the default LLM capability enabled. `vary_prompt` defaults to false for API compatibility and requires expansion plus at least two images; invalid combinations return 422. All candidate expansions are validated and committed in SQLite before the first image is rendered, with their composition goals and policy provenance. Restart recovery reuses those exact prompts. Selecting a thumbnail shows that option’s grey expanded description; adopting it or making a variation reuses the actual chosen prompt. The single-expansion preview button is disabled while composition variation is selected; candidate expansions are available on the job once planned and on each result.

The model still determines how strongly it follows those directions. [Z-Image’s official card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) rates Turbo’s diversity as low; different seeds alone may yield similar layouts. Distinct composition briefs help but do not guarantee perceptually different results, particularly for tightly specified framing or strongly preserved references. Lower **Keep the original** to allow larger reference-photo changes when desired. Merely choosing numerically distant/random seeds is not a reliable diversity control.

`n` is a strict integer, 1–5, on both generation routes. `seed` sets the first option; later options use `(seed + zero_based_index) % 4294967296`, so this example uses 42, 43, 44. Omitting seed picks and persists a random first seed. The synchronous endpoint waits for the whole batch and returns all images in `data[]`; an asynchronous job returns them in `result.data[]` once succeeded. Each item includes `candidate_index` (one-based), `candidate_count`, seed, actual paths and URL/base64. `generation_seconds` is the sum of rendering times for all options, excluding LLM planning/expansion time. More options take proportionally more rendering time without requiring concurrent model instances.

One request is one durable queue job. `completed_images` and `total_images` report progress; detail responses may include `partial_result.data[]` containing only fully saved options while running or after failure. `result` is reserved for a fully successful batch. List responses omit image result blobs. A mid-batch failure preserves completed files/catalog entries; inspect the job or gallery. An explicit retry creates a new full batch with a new identity. Restart recovery of an interrupted job reuses its completed options, seeds and saved expansion(s) and renders only missing options. IDs remain stable across recovery. A running batch finishes before the next job; priority/pause/clear never interrupt it. All options remain ordinary gallery images with searchable prompts/tags. The queue history's **View** action reopens the whole completed batch.

The option picker below shows a real two-image Z-Image Turbo test at 512 × 512 with distinct seeds and LLM composition prompts. Select a thumbnail to change the preview and its actions.

![Two generated options with distinct seeds and selectable previews](docs/screenshots/image-options.png)

## Independent queue API (generator may be off)

There are two processes: **queue API** (`55302`) handles agent-facing SQLite CRUD and metadata discovery; **generator** (`5732`) owns the webpage, catalog, LLM, image generation and queue consumption. The webpage uses only the generator API, which reads/writes the same SQLite databases directly through the shared storage layer. It does not proxy its queue operations through the standalone queue API. No worker registry, heartbeat, remote worker selection or per-job model switching is involved. Start the generator with the model whose queue you want to drain.

```sh
make queue
# Agents can enqueue even with no generator running. Omit model for Z-Image Turbo.
curl --fail-with-body http://127.0.0.1:55302/v1/jobs \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: agent-forest-1' \
  -d '{"request":{"prompt":"A lantern in a misty forest","size":"512x512","n":2,"expand_prompt":true,"response_format":"url"}}'
# Route a request to another model explicitly (it need not be set up yet).
curl --fail-with-body http://127.0.0.1:55302/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"model":"chroma1-flash","request":{"prompt":"A watercolor mountain village","response_format":"url"}}'

curl 'http://127.0.0.1:55302/v1/jobs?model=chroma1-flash&status=pending&limit=24&offset=0'
curl 'http://127.0.0.1:55302/v1/jobs/<job-id>?model=chroma1-flash'
curl --request PATCH 'http://127.0.0.1:55302/v1/jobs/<job-id>?model=chroma1-flash' \
  -H 'Content-Type: application/json' -d '{"request":{"prompt":"A snowy mountain village","n":2}}'
curl --request DELETE 'http://127.0.0.1:55302/v1/jobs/<job-id>?model=chroma1-flash'
curl --request DELETE 'http://127.0.0.1:55302/v1/jobs?model=chroma1-flash'
curl --request POST 'http://127.0.0.1:55302/v1/jobs/<job-id>/retry?model=chroma1-flash'
curl 'http://127.0.0.1:55302/v1/queue?model=chroma1-flash'
curl --request PATCH 'http://127.0.0.1:55302/v1/queue?model=chroma1-flash' \
  -H 'Content-Type: application/json' -d '{"paused":true}'

# Later, when willing to use the accelerator:
make serve MODEL=z-image-turbo
# Ctrl-C leaves the queue API running; optionally stop it separately:
make stop-queue
```

`POST /v1/jobs` on the **queue port** takes an envelope: `{"model":"z-image-turbo","request":{...image settings...}}`. Model is optional and defaults to **Z-Image Turbo** (`z-image-turbo`), independent of the running generator. Other operations take an optional `?model=<id>` query with that same default. Supported IDs are `z-image-turbo`, `chroma1-hd`, `chroma1-flash`; invalid IDs are rejected. Every job response includes the resolved top-level `model`. A nested `request.model`, if supplied, must match the selected queue. `GET /v1/models` and `/health` on the queue port report `default_model`.

Discover adapters on the queue port without starting the generator:

```sh
curl 'http://127.0.0.1:55302/v1/loras?model=z-image-turbo'
curl 'http://127.0.0.1:55302/v1/loras?model=chroma1-flash'
# Omit model to use Z-Image Turbo:
curl 'http://127.0.0.1:55302/v1/loras'
```

`GET /v1/loras` returns `object:"list"`, resolved `model`, absolute adapter `directory` and `data[]` with IDs/names, compatible models, `available`, `default_strength`, `trigger_prompt`, source and recorded checksum. Use an entry with **`available:true`** as `request.lora`; optional `request.lora_strength` overrides its default. The selected model's local `catalog.json` takes precedence over the shipped public catalog. Compatible configured entries with missing files/checksums remain listed with `available:false`. Chroma HD and Flash use the existing shared HD adapter folder, filtered separately by model compatibility. Unregistered files are not usable adapter IDs and are not listed.

Discovery reads metadata and checks file presence; it never loads weights, hashes adapter files, initializes SQLite or downloads anything. Inference verifies the checksum before use. A missing configured volume or unreadable/invalid catalog returns503 instead of pretending no adapters exist; queue CRUD stays available. Unknown models return422. Availability is independent of generator startup mode, so this endpoint has no `enabled` flag; a generator started with `NO_LORAS=1` leaves adapter-dependent queue entries waiting. Private catalogs and weights stay outside Git.

The lightweight process checks the envelope, model, JSON values and strict `generate_immediately` boolean, then commits opaque image settings. **202 means persisted, not model-validated or generated.** The generator later validates image parameters, resolves the selected model's defaults/output folder, pins adapter checksums and freezes any random seed in SQLite **before** expansion/rendering. Invalid requests become failed records with errors; subsequent jobs continue. Keep referenced files accessible until consumption. Jobs needing an unavailable optional LLM/adapter mode remain waiting until that mode is enabled or they are cancelled. Other model queues wait until their generator is selected. Queue startup/restart never recovers a running job or takes accelerator/worker ownership.

Use PATCH with the **complete replacement request**, not a partial merge. It is allowed only for queued jobs with `attempts=0`; running/interrupted/finished entries return409. PATCH cannot move a job to another model. The ID/order/creation time stay the same, while priority and the idempotency comparison update to the new settings. DELETE and clear cancel waiting entries and preserve history/media, including partial batch files. Retry creates a new FIFO identity. Idempotency keys are model-scoped, so the same key can be used independently for different models.

Queue detail/list responses contain raw persisted `request`, optional `prompt_expansion`, `result` metadata (single-image paths or batch `images[]`), timestamps/attempts/error and state. A partial checkpoint may be present while running or failed: only `status:"succeeded"` means a fully completed request. Pagination returns totals and navigation, with null offsets at boundaries. The queue process does **not** provide image/base64 download routes, uploads, the catalog or a UI; those stay on the generator. `GET /` on the queue port returns404. Use the generator's job endpoint for the existing `result.data[]` download response when it is running.

## If the standalone queue API goes down

The browser never opens SQLite itself. It submits, lists, edits, cancels, retries and polls jobs through the **generator API**, whose handlers use the same shared `JobStore`/SQLite transactions as the standalone queue API. Those operations, synchronous generation, saved results and the gallery keep working while the generator remains up. Accepted jobs continue; no second queue, HTTP write fallback or automatic service restart is involved.

Generator `GET /health` includes `queue_api.available`, `url`, `recovery_command` and an optional `error`. A bounded one-second on-demand probe handles connection refusal, timeouts, dropped/invalid responses and wrong-service/project replies. An unavailable standalone API does **not** change a healthy generator's `status:"ready"` or mark its consumer failed. The queue panel displays an informational recovery notice while its controls remain usable. Restore agent access with `make queue` (with the same `QUEUE_PORT` override, if used); health/UI checks recognize recovery without restarting the generator.

A local SQLite failure is different: affected generator endpoints return HTTP503 with `code:"queue_storage_unavailable"`, a useful `detail` and `Retry-After:2`. Check project-state storage and permissions. A lost submission response is not proof that no row was committed: retry the same request with its original `Idempotency-Key`, as the UI already does. Existing jobs/history/media remain intact. Queue HTTP restart never recovers or reclaims the generator's active work.

## Persistent generation queue

Each startup model has its own SQLite database at **`<project>/state/queues/<model-id>/jobs.sqlite3`**. On the original Mac, Z-Image uses `~/Documents/models/ImageGen-Service/state/queues/z-image-turbo/jobs.sqlite3`. SQLite databases, journals, backups and worker leases stay in the serving project, even when model weights and media use an external SSD. `DATA_DIR` does not relocate SQLite. The same queue is read automatically whenever that model starts. Changing the startup model leaves the other model's waiting jobs persisted until it is served again; models are not hot-swapped per job.

The worker executes one image at a time. Ordinary jobs are FIFO. **`generate_immediately:true` means the next free slot ahead of ordinary jobs**, with FIFO order among immediate jobs. It never interrupts the active image/batch. The webpage checks “Generate immediately” by default; uncheck it to add ordinary FIFO work. You can keep submitting while an image runs. The queue panel lists jobs, removes waiting entries, clears waiting work, retries failed/cancelled jobs, and provides pagination and a persistent pause toggle. Pause stops ordinary jobs only; the active batch finishes and explicitly immediate jobs may still run.

```sh
# Persist an ordinary FIFO job and promptly receive its ID (HTTP 202).
curl --fail-with-body http://127.0.0.1:5732/v1/jobs \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: my-first-image' \
  -d '{"prompt":"A fox beneath snowy pine trees","size":"512x512","response_format":"url","generate_immediately":false}'

# Submit to the next immediate slot instead (same settings, distinct request key).
curl --fail-with-body http://127.0.0.1:5732/v1/jobs \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: my-next-image' \
  -d '{"prompt":"A lantern in a misty forest","size":"512x512","response_format":"url","generate_immediately":true}'

curl 'http://127.0.0.1:5732/v1/jobs?status=pending&limit=24&offset=0'
curl 'http://127.0.0.1:5732/v1/jobs/<job-id>'
curl --request DELETE 'http://127.0.0.1:5732/v1/jobs/<job-id>'
curl --request DELETE 'http://127.0.0.1:5732/v1/jobs'
curl --request POST 'http://127.0.0.1:5732/v1/jobs/<job-id>/retry'
curl --request PATCH 'http://127.0.0.1:5732/v1/queue' \
  -H 'Content-Type: application/json' -d '{"paused":true}'
```

These routes on the **generator port (5732)** read/write the shared SQLite database directly and operate on the startup-selected model, even while the standalone queue API is down. They keep the existing flat image request and image-specific response format; they do not use the queue-port envelope:

| Method / path | Use |
| --- | --- |
| POST `/v1/jobs` | Persist a job; return 202 and an ID after the SQLite commit |
| GET `/v1/jobs?status=pending&limit=24&offset=0` | List running/waiting jobs with pagination; omit status for history |
| GET `/v1/jobs/{id}` | Read one job; completed jobs include `result` |
| PATCH `/v1/jobs/{id}` | Replace a never-started waiting request (complete flat image settings) |
| DELETE `/v1/jobs/{id}` | Cancel waiting work; running/finished409, unknown404 |
| DELETE `/v1/jobs` | Cancel all waiting entries; return `{"cancelled": count}` |
| POST `/v1/jobs/{id}/retry` | New ordinary FIFO job for a failed/cancelled entry |
| PATCH `/v1/queue` | Persist `{"paused":true}` or `{"paused":false}` for ordinary FIFO work |

A job response includes `id`, `model`, `status`, normalized `request`, `generate_immediately`, timestamps, `attempts`, optional `error`, and `result` when complete. Statuses are `queued`, `running`, `succeeded`, `failed`, `cancelled`. A 202 acknowledges the database commit, not image completion. Poll GET `/v1/jobs/{id}`; a successful `result` has the usual generation response with actual image/metadata paths and URL/base64 according to `response_format`. List responses omit result blobs and include `data`, `total`, `limit`, `offset`, `has_more` and `has_previous`. `next_offset` / `previous_offset` are present only when that direction exists; optional null fields are omitted. `status=pending` selects waiting/running jobs; other statuses select that exact state. Omit status for history. Limit is 1–100 (default24); offset is nonnegative. Running comes first, waiting work follows execution order, then finished history newest first. An initial 202 or a queued/running status does **not** contain finished image paths. Once `status` is `succeeded`, iterate **`result.data[]`** for each `file_path`, `metadata_path`, and `url` for `response_format:"url"`. Prefer URL responses for polling; base64 detail responses reread/embed the PNG and can be large. An accepted job may still fail during inference or output writing; inspect its terminal status/error.

DELETE of one entry cancels **waiting work only**; deleting a running/finished job returns409. Clearing returns `{"cancelled": count}`, cancels all waiting work for the active model and preserves the running image, completed files and job history. This removes entries from the active queue, not from the audit history. Failed/cancelled jobs can be explicitly retried as new ordinary FIFO jobs with the saved seed/settings and a new ID; restore missing references/adapters first. Failed jobs do not block later jobs or retry forever. A base-only server preserves a queued adapter job at the head until adapters are enabled or that waiting entry is removed.

`Idempotency-Key` is optional on both submission routes (1–128 ASCII letters/digits/`._:-`). Reuse the **same key and settings** after a network timeout to get the original job, with its one frozen random seed once generation starts. An unconsumed request may still omit the seed or show null. A changed request with that key returns409. A new key intentionally creates another image. The UI remembers its unacknowledged submission key in session storage, and its followed job ID across page reloads. Idempotency does not deduplicate intentionally different keys.

Generator shutdown finishes the active image and leaves waiting jobs on disk; the separate queue API keeps accepting requests. If the process is abruptly killed during inference, startup retries the interrupted job using its original ID, effective settings and seed; it starts sampling again, not from an intermediate diffusion step. Outputs are published from private working files. If a complete matching PNG and metadata were published before the database success commit, recovery validates/adopts that artifact instead of generating another image. The custom-output index is restored if needed. References and adapter files must remain available; image bytes are not stored inside SQLite. Native subprocesses inherit worker/GPU leases so killing their Python parent cannot permit overlapping native inference; a restarting worker waits for the old native process to exit before recovery. Completed history is not automatically rerun.

Upgrading from the former `<store>/<model-id>/queue/jobs.sqlite3` layout automatically migrates that model at startup, before recovery/inference. Stop its old server and let native inference finish first. Migration holds both worker leases, uses SQLite backup (including committed WAL rows), verifies integrity/content and retains a standalone `legacy-backup-*.sqlite3` beside the new database before retiring the old database and sidecars. Identical copies after an interrupted migration are recognized; differing databases cause a clear error without merging or overwriting them. No external queue fallback is used. Keep `state/` as well as the media store when moving/backing up a project; a source-only clone starts with an empty local queue.

Queue schema3 adds a one-time preparation checkpoint for deferred defaults/seeds, upgrading schema1/2 in place without changing job IDs/order. Existing admissions retain their resolved defaults, seeds and adapter snapshots. Older servers refuse schema3 rather than processing requests they cannot prepare; keep the updated server when restarting. SQLite uses short transactions, a rollback journal and `synchronous=FULL`; inference runs outside database transactions. The database, journals and lock files are ignored by Git. Back up the database while the service is stopped or with SQLite's backup API, together with output/reference/index files. This protects against process restarts; it does not replace backups or guarantee survival of disk failure. Missing media SSDs block startup/inference without creating fallback weights/media; the independent queue API can still submit/list/update/cancel local jobs with the media SSD disconnected.

The worker cooperates with other local services through a configurable filesystem GPU lock. `GPU_LOCK_FILE=/absolute/path/gpu.lock` or ignored `storage.json` key `gpu_lock_file` selects it; the portable default is `<store>/.model-services/gpu.lock`. This Mac explicitly shares `/Volumes/ASH-SSD/models/.voice-studio/gpu.lock` with the voice services. A contended accelerator makes pending jobs wait, rather than fail; they remain removable until inference begins. Model loading also holds this lock. Serializing inference does not evict idle resident weights; stop/unload unused models if memory is tight. `/health.queue.stage` distinguishes waiting for the accelerator from generating, and storage failures appear as errors without discarding accepted jobs.

## Reference photos and variations

The webpage accepts a photo upload or an absolute path. The API also accepts raw bytes:

```sh
curl --fail-with-body http://127.0.0.1:5732/v1/reference-images \
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
curl 'http://127.0.0.1:5732/v1/images?q=forest&tag=Winter&limit=24&offset=0&sort=newest'
curl 'http://127.0.0.1:5732/v1/images/0123456789abcdef0123456789abcdef'
curl --request PATCH 'http://127.0.0.1:5732/v1/images/0123456789abcdef0123456789abcdef/tags' \
  -H 'Content-Type: application/json' -d '{"tags":["Campaign","Village"]}'
```

`q` is a case-insensitive substring. Repeated `tag` parameters require all exact labels, case-insensitively; they combine with search. `limit` is 1–100 (default 24), `offset` is nonnegative, `sort` is `newest` or `oldest`. The response returns `total`, `limit`, `offset`, `has_previous`, `has_more`, and `data`; `previous_offset` / `next_offset` are omitted when unavailable. Missing files are skipped; old/corrupt/missing sidecars do not prevent browsing valid PNGs. Tag edits are stored atomically in the owning model's `image-index`; `[]` clears tags. The index also resolves custom-folder images after restart. Back it up.

## Optional adapters

Default setup downloads no adapters. **Realism for Z-Image Turbo is the only bundled optional preset**:

```sh
make setup MODEL=z-image-turbo WITH_REALISM=1
make download-realism MODEL=z-image-turbo
make install-lora MODEL=z-image-turbo FILE=/absolute/path/adapter.safetensors LORA_ID=realism
```

Public downloads use only tracked `lora-catalog.json`, never personal catalogs. Compatible personal catalogs/checkpoints may live under ignored per-model `loras` folders; HD/Flash share `chroma1-hd/loras`. `GET /v1/loras` returns only the active model's entries and availability. The loader applies an adapter directly; a configured trigger phrase is prepended once automatically. Do not add `<lora:...>` tags to prompts. `--no-loras` / `NO_LORAS=1` hides the selector, returns an empty disabled catalog and rejects adapter requests. Adapter files and personal presets are never included in Git/default setup.

## Verification, recovery and AI handoff

`make test` runs fake-backend/storage/download tests without weights/GPU work. `make smoke` calls the running model, creates a real 512px image with its default steps, decodes it and rejects a blank result. This can be slow for Chroma. Health and gallery are responsive during inference. Queue checks cover ordering, restart recovery, artifact reconciliation, idempotency, cross-process ownership, shared-GPU contention and waiting-only delete/clear. Native generation logs are under the active model's `logs/`.

If a download is interrupted, rerun `make setup MODEL=<id>` or `make download MODEL=<id>`; files are reused/resumed. An incomplete setup has no new completion marker. For a missing SSD, reconnect it; do not remove broken links or choose a fallback to hide the problem. A port conflict should be resolved with `PORT=...` or by stopping the identified service. Avoid multiple GPU services on a memory-constrained Mac. If setup cannot create symlinks, use a symlink-capable project filesystem or project-local storage. Local model paths and the native runtime are verified before inference; startup does not fetch network resources.

For another AI tool:

> Read README.md and skill/image-model-server/SKILL.md in this checkout. Run make setup for the desired model (default Z-Image Turbo), choosing the requested model store or Enter for local storage. Run make test. SQLite queues stay in project-local state/queues/<model-id>/jobs.sqlite3 independently of DATA_DIR. Start make serve MODEL=<id> in a retained session, wait for /health ready and verify the active model and serving mode. Report http://127.0.0.1:5732/ (or the explicit PORT override). For enqueueing while generation is off, run make queue and use http://127.0.0.1:55302/v1/jobs with {model,request}; omitted model defaults to z-image-turbo and query operations use ?model=<id>. make serve starts/reuses that queue API before the foreground generator/UI. On the generator port, prefer POST /v1/jobs with the existing flat image settings, reuse Idempotency-Key plus the same request after an uncertain submission, poll GET /v1/jobs/{id}, and iterate result.data[] only after succeeded. Preserve existing jobs and data; keep all model weights, media, tags, personal adapters, settings, SQLite state and storage symlinks outside Git.

No Codex plugin, original paths, shell aliases, or chat history is required. The service accepts local file paths and is intended for trusted local clients; it binds to `127.0.0.1` with host checks and no permissive CORS. Do not expose it to a network without an access-control design. The source-only repository ignores model folders, environments, runtime binaries, output/reference images, indexes, logs, credentials and local settings. The only tracked image files are the four approved README screenshots under docs/screenshots/. Cleanup should preserve actual data unless explicitly requested.
