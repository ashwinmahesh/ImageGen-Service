# ImageGen-Service

One Apple Silicon project for **Z-Image Turbo**, **Chroma1-HD**, and **Chroma1-Flash**, with a shared webpage, API, and searchable image gallery. Select a model when starting the server; one backend runs at a time. Source repository: [ImageGen-Service](https://github.com/ashwinmahesh/ImageGen-Service).

## UI preview

Generate from a prompt or reference photo, choose immediate priority or ordinary FIFO, and follow saved jobs across reloads.

![Generator with reference-photo and image controls](docs/screenshots/generator.png)

Inspect, pause, remove or clear waiting jobs without interrupting the active image.

![Persistent generation queue and controls](docs/screenshots/queue.png)

Search saved images by prompt or tags, browse pages, and open an image to reuse its settings or edit its tags. Screenshots use demonstration data and a simulated backend; displayed timings are not performance benchmarks.

![Searchable tagged image gallery](docs/screenshots/gallery.png)

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

Without a terminal, or with `NONINTERACTIVE=1`, an omitted model defaults to Z-Image Turbo. `MODEL=...` works for setup, download, serving and adapter commands. `IMAGE_MODEL` is the equivalent Python environment setting. `PORT=5733` changes the endpoint. Use `BOOTSTRAP_PYTHON=/absolute/path/to/python3.14` if `python3.14` is installed elsewhere. `VENV=/absolute/internal/path/.venv` overrides the environment; use the same value consistently.

## Model storage

First setup also asks for a model store folder. **Blank/Enter uses the project directory**; another absolute directory (or `~/...`) stores data there and creates per-model symlinks in the project. The chosen store contains one folder per model:

```text
<store>/z-image-turbo/{weights,images,references,image-index,cache,loras,logs}
<store>/chroma1-hd/{weights,images,references,image-index,cache,loras,logs}
<store>/chroma1-flash/{weights,images,references,image-index,cache,loras,logs}
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

`DATA_DIR` means the **store containing model folders**, not one model's weights folder. On the original Mac, the single serving project is `~/Documents/models/ImageGen-Service`, and the store is `/Volumes/ASH-SSD/models`. Existing model, output, tag and custom-location index paths are reused in place. A fresh clone needs no external SSD or original user's home path. Prior separate source folders can be retained as backups; use the consolidated project for new changes.

Chroma HD and Flash share their T5 encoder and VAE in `chroma1-hd/weights`, and one native runtime in `<project>/runtime`. **Flash setup does not download the HD diffusion checkpoint.** Back up the actual SSD data, including indexes and tags, **and the project-local `state/` folder**; symlinks are not backups.

## Start, switch and stop

`make serve` runs in the foreground; keep the terminal open. Ctrl-C stops it gracefully. AI tools should retain the terminal session and use another session for checks.

- Webpage: <http://127.0.0.1:5732/>
- All-model image gallery: <http://127.0.0.1:5732/gallery>
- API docs and live schema: <http://127.0.0.1:5732/docs> and `/openapi.json`
- Readiness: `make health` or `GET /health`

The default is **`http://127.0.0.1:5732/`**, also reachable as `http://localhost:5732/`. `PORT=...` changes the port for serving, health and smoke commands; use the same override for each command. No hosts-file mapping or `http://imagegen` shortcut is installed.

Health reports `status`, active `model`, `busy`, `project_dir`, active `data_dir`, `loras_enabled`, model `capabilities`, and durable `queue` counts/state/database path. `GET /v1/models` lists all supported models, download markers and the active one. A download marker indicates completed setup, not a full startup integrity check; startup verifies required local assets too. The UI reads model identity, step defaults and limits from health.

To switch, stop the server and run `make serve` again, or run `make serve MODEL=...` in another terminal. The launcher reuses an identical instance or gracefully replaces a **verified managed idle** service. It refuses to interrupt generation or replace unrelated listeners. It never hot-swaps models inside an inference request. A request's `model`, if specified, must equal the active model; mismatches return 422. Serving never downloads models. If assets are missing, run `make setup MODEL=<id>` first.

If the launcher finds the same model/mode already running from this checkout, it prints the URL and exits successfully; the original terminal still owns that server. A model switch leaves its waiting queue persisted until that model starts again. During initial model loading the HTTP listener may not yet be reachable; connection refusal can be normal until startup completes. Read logs if readiness never arrives.

The original Mac's shortcuts `serve-images`, `serve-z-image-turbo`, `serve-chroma1-hd`, `serve-chroma1-flash` and the three `-no-loras` variants all use this one project. Other users can use Make directly without aliases.

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

`model` and `steps` may be omitted to use the active model/default. Response fields include `created`, `generation_seconds`, and `data[0]` containing `seed`, `tags`, `size`, `width`, `height`, `generation_size`, **absolute `file_path` and `metadata_path`**, plus `url` or `b64_json`. These paths refer to the server's filesystem. PNG and JSON metadata are always saved, including for base64 responses. URLs remain resolvable after restarts/model switches while files and indexes remain available.

| Request field | Default / constraints |
| --- | --- |
| `prompt` | Required nonblank text; max 4000 characters. Z-Image processes the first 512 tokens including formatting and truncates excess. Put essential details first. |
| `size` | `1024x1024`; each dimension a multiple of 64, 256–2048. Large images need more time/memory. VAE tiling is used above 1024 for Z-Image and always for Chroma. |
| `steps` | Active model's default and range above; references require at least 2. |
| `seed` | Omit/null for random; integer 0–4294967295. Returned and saved. |
| `n` | Only 1. |
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

## Persistent generation queue

Each startup model has its own SQLite database at **`<project>/state/queues/<model-id>/jobs.sqlite3`**. On the original Mac, Z-Image uses `~/Documents/models/ImageGen-Service/state/queues/z-image-turbo/jobs.sqlite3`. SQLite databases, journals, backups and worker leases stay in the serving project, even when model weights and media use an external SSD. `DATA_DIR` does not relocate SQLite. The same queue is read automatically whenever that model starts. Changing the startup model leaves the other model's waiting jobs persisted until it is served again; models are not hot-swapped per job.

The worker executes one image at a time. Ordinary jobs are FIFO. **`generate_immediately:true` means the next free slot ahead of ordinary jobs**, with FIFO order among immediate jobs. It never interrupts the current image. The webpage checks “Generate immediately” by default; uncheck it to add ordinary FIFO work. You can keep submitting while an image runs. The queue panel lists jobs, removes waiting entries, clears waiting work, retries failed/cancelled jobs, and provides pagination and a persistent pause toggle. Pause stops ordinary jobs only; the active image finishes and explicitly immediate jobs may still run.

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

The queue APIs operate on the startup-selected model only:

| Method / path | Use |
| --- | --- |
| POST `/v1/jobs` | Persist a job; return 202 and an ID after the SQLite commit |
| GET `/v1/jobs?status=pending&limit=24&offset=0` | List running/waiting jobs with pagination; omit status for history |
| GET `/v1/jobs/{id}` | Read one job; completed jobs include `result` |
| DELETE `/v1/jobs/{id}` | Cancel waiting work; running/finished409, unknown404 |
| DELETE `/v1/jobs` | Cancel all waiting entries; return `{"cancelled": count}` |
| POST `/v1/jobs/{id}/retry` | New ordinary FIFO job for a failed/cancelled entry |
| PATCH `/v1/queue` | Persist `{"paused":true}` or `{"paused":false}` for ordinary FIFO work |

A job response includes `id`, `model`, `status`, normalized `request`, `generate_immediately`, timestamps, `attempts`, optional `error`, and `result` when complete. Statuses are `queued`, `running`, `succeeded`, `failed`, `cancelled`. A 202 acknowledges the database commit, not image completion. Poll GET `/v1/jobs/{id}`; a successful `result` has the usual generation response with actual image/metadata paths and URL/base64 according to `response_format`. List responses omit result blobs and include `data`, `total`, `limit`, `offset`, `has_more` and `has_previous`. `next_offset` / `previous_offset` are present only when that direction exists; optional null fields are omitted. `status=pending` selects waiting/running jobs; other statuses select that exact state. Omit status for history. Limit is 1–100 (default24); offset is nonnegative. Running comes first, waiting work follows execution order, then finished history newest first. An initial 202 or a queued/running status does **not** contain finished image paths. Once `status` is `succeeded`, read **`result.data[0].file_path`**, `result.data[0].metadata_path`, and `result.data[0].url` for `response_format:"url"`. Prefer URL responses for polling; base64 detail responses reread/embed the PNG and can be large. An accepted job may still fail during inference or output writing; inspect its terminal status/error.

DELETE of one entry cancels **waiting work only**; deleting a running/finished job returns409. Clearing returns `{"cancelled": count}`, cancels all waiting work for the active model and preserves the running image, completed files and job history. This removes entries from the active queue, not from the audit history. Failed/cancelled jobs can be explicitly retried as new ordinary FIFO jobs with the saved seed/settings and a new ID; restore missing references/adapters first. Failed jobs do not block later jobs or retry forever. A base-only server preserves a queued adapter job at the head until adapters are enabled or that waiting entry is removed.

`Idempotency-Key` is optional on both submission routes (1–128 ASCII letters/digits/`._:-`). Reuse the **same key and settings** after a network timeout to get the original job, including its originally chosen random seed. A changed request with that key returns409. A new key intentionally creates another image. The UI remembers its unacknowledged submission key in session storage, and its followed job ID across page reloads. Idempotency does not deduplicate intentionally different keys.

Shutdown finishes the active image and leaves waiting jobs on disk. If the process is abruptly killed during inference, startup retries the interrupted job using its original ID, effective settings and seed; it starts sampling again, not from an intermediate diffusion step. Outputs are published from private working files. If a complete matching PNG and metadata were published before the database success commit, recovery validates/adopts that artifact instead of generating another image. The custom-output index is restored if needed. References and adapter files must remain available; image bytes are not stored inside SQLite. Native subprocesses inherit worker/GPU leases so killing their Python parent cannot permit overlapping native inference; a restarting worker waits for the old native process to exit before recovery. Completed history is not automatically rerun.

Upgrading from the former `<store>/<model-id>/queue/jobs.sqlite3` layout automatically migrates that model at startup, before recovery/inference. Stop its old server and let native inference finish first. Migration holds both worker leases, uses SQLite backup (including committed WAL rows), verifies integrity/content and retains a standalone `legacy-backup-*.sqlite3` beside the new database before retiring the old database and sidecars. Identical copies after an interrupted migration are recognized; differing databases cause a clear error without merging or overwriting them. No external queue fallback is used. Keep `state/` as well as the media store when moving/backing up a project; a source-only clone starts with an empty local queue.

SQLite uses short transactions, a rollback journal and `synchronous=FULL`; inference runs outside database transactions. The database, journals and lock files are ignored by Git. Back up the database while the service is stopped or with SQLite's backup API, together with output/reference/index files. This protects against process restarts; it does not replace backups or guarantee survival of disk failure. Missing media SSDs block startup/inference without creating fallback weights/media; queue listing/removal on an already-running service uses its local database.

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

> Read README.md and skill/image-model-server/SKILL.md in this checkout. Run make setup for the desired model (default Z-Image Turbo), choosing the requested model store or Enter for local storage. Run make test. SQLite queues stay in project-local state/queues/<model-id>/jobs.sqlite3 independently of DATA_DIR. Start make serve MODEL=<id> in a retained session, wait for /health ready and verify the active model and serving mode. Report http://127.0.0.1:5732/ (or the explicit PORT override). Prefer POST /v1/jobs, reuse Idempotency-Key plus the same request after an uncertain submission, poll GET /v1/jobs/{id}, and read result.data[0] only after succeeded. Preserve existing jobs and data; keep all model weights, media, tags, personal adapters, settings, SQLite state and storage symlinks outside Git.

No Codex plugin, original paths, shell aliases, or chat history is required. The service accepts local file paths and is intended for trusted local clients; it binds to `127.0.0.1` with host checks and no permissive CORS. Do not expose it to a network without an access-control design. The source-only repository ignores model folders, environments, runtime binaries, output/reference images, indexes, logs, credentials and local settings. The only tracked image files are the three approved README screenshots under docs/screenshots/. Cleanup should preserve actual data unless explicitly requested.
