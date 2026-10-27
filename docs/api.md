# ImageGen API reference

This is the human-readable reference for **all HTTP endpoints in ImageGen-Service**. Use it to build a client, submit images from another tool, manage saved work or browse results. For your first request, follow the [quick-start walkthrough](../QUICKSTART.md#5-generate-through-the-api).

## Contents

- [Services and conventions](#services-and-conventions)
- [Health, models and LoRAs](#health-models-and-loras)
- [Generation settings](#generation-api)
- [Job and queue endpoints](#job-and-queue-endpoints)
- [Job and image responses](#job-and-image-responses)
- [Standalone queue requests](#independent-queue-api-generator-may-be-off)
- [Reference photos](#reference-photos-and-variations)
- [Prompt expansion](#local-prompt-expansion-enabled-by-default)
- [Gallery, tags and pagination](#searchable-gallery-tags-and-pagination)
- [Observability](#observability)
- [Media, webpages and API documentation](#media-webpages-and-api-documentation)
- [Errors and safe retries](#errors-and-safe-retries)
- [Outages](#if-the-standalone-queue-api-goes-down) and [restart recovery](#persistent-generation-queue)

## Services and conventions

| Service | Default base URL | Start it | Available while the generator is off? |
| --- | --- | --- | --- |
| **Generator** | `http://localhost:5742` | `make serve` | No. It owns image generation, the webpage, uploads and gallery. |
| **Queue** | `http://localhost:55302` | `make queue` | Yes. It accepts and manages persisted jobs without loading models. |

`make serve` starts the queue API if necessary. `PORT` changes the generator port; `QUEUE_PORT` changes the queue port. Use overrides consistently. Both bind to loopback, accept trusted local clients and have no API-key authentication. No permissive cross-origin access is configured. Local file paths refer to the server's Mac.

Unless stated otherwise, send JSON with `Content-Type: application/json`; responses are JSON. Request-body booleans must be JSON `true`/`false`. Generation integers such as `n`, `steps` and `seed` must be numbers, not strings or booleans. Unknown fields are rejected on typed request bodies. The queue's nested `request` is deliberately open for later generator validation.

Supported model IDs are `z-image-turbo`, `chroma1-hd`, `chroma1-flash`, `flux2-klein-4b`, `flux2-klein-9b` and `qwen-image-2.1-turbo`:

Both services use **one shared queue** at `state/queue/jobs.sqlite3`. Every job stores a resolved model. Omitted submission `model` uses the saved default selected with `make serve MODEL=<id>`; a fresh installation prefers setup Z-Image Turbo, otherwise the first setup model. The default does not change when the active backend switches. Discovery and new submissions support **setup models only**, using the same required-assets/runtime preflight. Unready models return422 with setup guidance. Setup itself still offers the complete registry.

GET list and DELETE clear accept an optional `model` filter; omission spans every model. Detail and mutation routes address unique job IDs directly. Pause, queue positions, moves and Run Now are global. A model that becomes unavailable after admission stays queued with a blocked reason while other eligible jobs proceed.

Path placeholders such as `{identifier}` mean a returned job or image ID: 32 lowercase hexadecimal characters. Job and image IDs can differ in a batch; use returned URLs and gallery IDs. Timestamps are Unix seconds; job timestamps may include fractions. Optional response fields may be null or omitted depending on the route. Examples show representative values and may omit optional fields.

## Health, models and LoRAs

| Endpoint | Service | Success | Purpose |
| --- | --- | --- | --- |
| GET `/health` | Both | 200 | Process, model and queue state. |
| GET `/v1/models` | Both | 200 | Discover supported model IDs. |
| GET `/v1/loras` | Both | 200 | Discover compatible configured adapters. |

### GET /health

The generator also reports an independent `observability` status when available; a metrics failure does not make generation unavailable.

No body or query parameters.

**Generator response:**

| Field | Type | Meaning |
| --- | --- | --- |
| `status` | string | `loading`, `ready` or `error`. Check this value as well as HTTP status. Model loading is deferred to jobs; the listener remains available during loading and switching. |
| `service`, `version`, `queue_scope` | strings | `imagegen-generator`, `4.0.0`, `shared`. |
| `runtime_model_switching` | boolean | Always true for this generator contract. |
| `default_model`, `active_model`, `model_loaded` | nullable string, nullable string, boolean | Saved submission default; selected/initializing backend or null; whether its adapter is initialized. Native per-image runtimes release their weights after each image. |
| `project_dir`, `data_dir` | strings | Checkout and default model’s data folder. Each image response has its actual file path/model. |
| `busy` | boolean | Image inference, prompt preview or a running queue job is active. |
| `loras_enabled` | boolean | Whether this generator accepts adapters. |
| `prompt_expansion` | object | Availability, mode, LLM `model`/`revision`/`license`, `text_only`, `guidance_profile`, `batch_variations` and usage `note`. |
| `capabilities` | object | Backend/name, default/max steps, default size, dimension/prompt/count limits, adapter/LLM mode, queue URL and persistent/immediate support. The default model-ID key contains `prompt_expansion_guidance`. |
| `queue` | object, when initialized | Counts/pause state below, plus `stage`, optional `blocked_reason` and consumer `error`. |
| `queue_api` | object | Standalone queue availability: `available`, `url`, `recovery_command`, optional `error`. Its outage alone does not make the generator unready. |
| `progress` | object, when supported | Backend-specific inference progress while busy. Its fields depend on the backend. |

The generator's `capabilities` object exposes these named controls:

| Keys | Types | Meaning |
| --- | --- | --- |
| `model`, `name`, `backend` | strings | Default model identity; active backend is reported separately as `active_model`. |
| `default_steps`, `max_steps`, `default_size` | integers, string | Defaults and step ceiling for the requested model. |
| `prompt_note`, `max_prompt_characters`, `min_dimension`, `max_dimension` | string, integers | Prompt behavior and canvas limits. |
| `default_image_count`, `max_image_count` | integers | Batch default and ceiling. |
| `loras_enabled`, `persistent_queue`, `immediate_generation` | booleans | Available modes. |
| `tasks`, `photo_editing` | array, object | Supported tasks and edit capabilities: eligible model IDs, required source, Auto reference sizing, maximum long edge, mask support, no composition variation, text-only edit refinement and human setup guidance. The guidance can name installable editors; discovery entries and accepted requests still include setup models only. |
| `reference_mode`, `image_strength_supported` | string, boolean | `variation` with preservation strength for Z-Image/Chroma; `edit` without that control for Klein/Qwen. |
| `prompt_expansion`, `queue_api_url` | object, string | LLM capability metadata and standalone queue origin. |
| `<model-id>` | object | For Klein/Qwen, also `quantization` and `license:{id,url,notice}`. `prompt_expansion_guidance` with profile `id`, `version`, `soft_max_new_words`, nullable `image_token_limit`, `verified_token_count` and `sources[]`. |

**Queue response:** `status`, `service:"imagegen-queue"`, `project_dir`, nullable `default_model`, `queue_scope:"shared"` and `version:"4.0.0"`. This checks the queue HTTP process; it does not load a model or guarantee that a later storage write will succeed.

Queue counts/state contain integer `queued`, `running`, `succeeded`, `failed`, `cancelled`; boolean `paused`; string `database_path`; nullable `next_job_id` (head waiting job before setup/capability checks, null while paused without immediate work); and boolean `next_requires_llm` / `next_requires_lora`. Generator stages include `idle`, `waiting_for_accelerator`, `loading_model`, `expanding_prompt`, `planning_variations`, `generating` and per-option generation. `next_model` labels the next waiting request.

### GET /v1/models

No body or query parameters.

Both responses list **only models whose required local assets and runtime files pass preflight**, without loading GPU weights or hashing them. Generator: `{object:"list",default_model,active_model,data:[...]}`; queue: `{default_model,data:[...]}`. Generator entries include `id`, `name`, `downloaded:true`, `output_dir`, per-model default/max steps, task/reference/mask capabilities, LLM guidance and license metadata under the exact model ID. Queue entries include ID/name/tasks/reference/mask/license metadata. `active_model` may be null before the first job. A preflight pass is not an inference benchmark or full checksum verification.

### GET /v1/loras

Both accept optional query `model`; omission uses the shared saved default. An unready model returns422.

| Response | Shape |
| --- | --- |
| Generator | `{object:"list", model, enabled:boolean, data:[...]}`. `NO_LORAS=1` returns `enabled:false` and an empty list. |
| Queue | `{object:"list", model, directory, data:[...]}`. `directory` is the absolute adapter folder. No `enabled` flag: generator mode is independent. |

Each entry describes `id`, `name`, `filename`, compatible `models[]`, `available`, optional `default_strength`, `trigger_prompt`, `source` and `sha256`; creator metadata may add fields. Select an entry with **`available:true`** and send its `id` as `lora`. Null/omitted `lora` uses the base model.

Availability checks file presence and recorded checksum metadata. Inference verifies the actual checksum. Discovery does not load/hash weights or download files. A model's local catalog overrides the shipped public catalog; compatible missing files remain listed as unavailable. Chroma HD and Flash share the HD adapter directory with compatibility filtering. Queue discovery returns **422** for unknown models and **503** for an unavailable configured volume or invalid/unreadable catalog; independent queue CRUD remains available.

## Generation API

These image settings are accepted directly by the generator's **POST `/v1/jobs`**, **PATCH `/v1/jobs/{identifier}`** and **POST `/v1/images/generations`**. On the standalone queue API they go inside `request`.

Only `prompt` is required for a valid image request. Defaults below apply when the generator validates the request; the queue API can accept an opaque request earlier.

| Field | Type | Default and rules |
| --- | --- | --- |
| `task` | string | `"generate"` by default. `"edit"` requires a setup editing-capable model and a source photo. Composition variation is rejected for edits. |
| `preserve` | string or null | Edit only: explicit preservation requirements, max 1000 characters. Instruction plus constraints must fit the final 4000-character prompt budget. |
| `mask_image` | string or null | Qwen edit only: absolute/`~/...` mask path, max 4096 characters. Must match the oriented source dimensions; white edits and black preserves the resized source. |
| `model` | string | Any setup model; omitted uses the shared saved default. On queue submissions use the outer `model`; a nested value, if present, must match. |
| `prompt` | string | Required, nonblank, 1–4000 characters. `<lora:...>` tags are rejected; use adapter fields. Z-Image processes the first 512 tokens including formatting/trigger, so put essentials first. |
| `size` | string | Without a reference: `"1024x1024"` by default. With a reference: omit or use `"auto"` to detect oriented source dimensions without upscaling, capped at a 2048px long edge. An explicit `WIDTHxHEIGHT` overrides the long edge; each supplied dimension must be a multiple of 64 from 256–2048. The source sets the aspect ratio; saved dimensions need not be grid multiples. Auto without a reference returns 422. |
| `n` | integer | 1–5, default 1. Sequential options in one job. |
| `steps` | integer | Model default/range in the table below. Reference generation requires at least 2. |
| `seed` | integer or null | Omit/null for random; otherwise 0–4294967295. First option's seed; later options increment modulo 2^32. |
| `expand_prompt` | boolean | false in API; enabled UI initially checks it. Requires LLM-enabled serving. Saves the original and expanded text. |
| `vary_prompt` | boolean | false. Requires `expand_prompt:true` and `n>=2`; creates a distinct expansion for each option. |
| `generate_immediately` | boolean | false for async jobs, true for synchronous generation. True gets the next free slot before ordinary waiting work; never interrupts active work. |
| `response_format` | string | `"b64_json"` or `"url"`; default `"b64_json"`. URL format is convenient for polling and downloads. Both save files. |
| `output_dir` | string | `<store>/<requested-model>/images`. Absolute folder or `~/...`, 1–4096 characters; missing folders are created if writable. Unique files do not overwrite originals. |
| `reference_image` | string or null | Omitted/null for text-only generation. One server-local absolute/`~/...` path, max 4096 characters; HTTP URLs are not accepted. |
| `image_strength` | number | Z-Image/Chroma variations only: default 0.6, [0,1), higher preserves more. Only send with a reference. Klein/Qwen reject this field even if set to 0.6; omit it for instruction edits. |
| `lora` | string or null | Omitted/null for none. Compatible installed catalog ID; lowercase letters, digits and hyphens, starting alphanumeric, max 80 characters. One adapter. |
| `lora_strength` | number or null | Requires `lora`; 0–2. Omit/null for its catalog default (Realism: 0.8). |
| `tags` | array of strings | `[]`. Up to 32 labels, each 1–64 characters after whitespace normalization. No commas/control characters; case-insensitive duplicates are removed. |

| Model | Default steps | Allowed steps | Effective runtime sampling |
| --- | ---: | ---: | --- |
| `z-image-turbo` | 9 | 1–12 | MFLUX Turbo defaults. |
| `chroma1-hd` | 40 | 1–80 | Euler, CFG 3. |
| `chroma1-flash` | 8 | 1–40 | Heun, CFG 1, beta schedule, flow shift 1. |
| `flux2-klein-4b` | 4 | 1–4 | Distilled flow-match Euler, guidance 1; MLX Q8. |
| `flux2-klein-9b` | 4 | 1–4 | Distilled flow-match Euler, guidance 1; MLX Q4. |
| `qwen-image-2.1-turbo` | 8 | 1–8 | Checkpoint Turbo sigma schedule, guidance 1; MLX Q4. |

Negative prompts, CFG/guidance, custom samplers/schedulers and multiple references are **not exposed** as request fields. Qwen photo edits accept `mask_image`; Klein does not. Klein/Qwen instruction editing uses the common `reference_image` and `prompt` fields. No model-specific request-option namespace is implemented. Larger images/options/step counts cost more time and memory. Z-Image uses VAE tiling above 1024; Chroma and Qwen always use it. Klein uses the runtime's default strip decoder. Klein/Qwen use low-RAM child processes and release all model memory afterward.

### Submit and poll an image

```sh
curl --fail-with-body http://localhost:5742/v1/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: forest-1' \
  -d '{"prompt":"A watercolor fox in a snowy forest","size":"512x512","response_format":"url"}'
```

The **202** response is a job record with `id`. Read **GET `/v1/jobs/{identifier}`** until `status` is `succeeded`, `failed` or `cancelled`. For example, replace the value below with the returned ID:

```sh
IMAGEGEN_JOB_ID="paste-the-returned-id-here"
curl --fail-with-body "http://localhost:5742/v1/jobs/$IMAGEGEN_JOB_ID"
```

A completed generator job returns images in **`result.data[]`**. An accepted or running job is not a finished image.

### Generate and wait synchronously

| Endpoint | Service | Success | Purpose |
| --- | --- | --- | --- |
| POST `/v1/images/generations` | Generator | 200 | Persist the same job, wait for completion and return an image response directly. |
| POST `/v1/images/edits` | Generator | 200 | Require an editing model and source photo; wait for the durable edit and return an image response. |

Accepts the full image-settings object and optional `Idempotency-Key` header. The only default difference is `generate_immediately:true`. Response is `{created, generation_seconds, data:[...]}`, without the job wrapper. The shared saved default is used if omitted.

```sh
curl --fail-with-body --max-time 1800 http://localhost:5742/v1/images/generations \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: forest-sync-1' \
  -d '{"prompt":"A watercolor fox in a snowy forest","size":"512x512","response_format":"url"}'
```

Use a long client timeout; disconnecting does not cancel the persisted job. Failed generation returns an HTTP error (the recorded error status, or 500); cancellation returns 400. Prefer async jobs when you need a promptly returned ID and restart-safe polling. This endpoint is a limited compatibility route, not a claim of full OpenAI API compatibility.

## Edit an existing photo

**POST `/v1/images/edits`** is a JSON, synchronous endpoint using the same durable worker and `GenerateResponse` as generation. It defaults to `task:"edit"` and immediate priority. It requires `reference_image` and a setup Klein/Qwen model; missing photos, unsupported models, or composition variation return **422**. For a promptly returned job ID, use **POST `/v1/jobs`** with `task:"edit"`. The standalone queue accepts the same settings inside `request`; model-specific validation happens when consumed.

```json
{
  "model": "qwen-image-2.1-turbo",
  "prompt": "Change the jacket to green.",
  "reference_image": "/absolute/path/photo.png",
  "preserve": "Face, pose, and background",
  "size": "1024x1024",
  "expand_prompt": true,
  "response_format": "url"
}
```

All three editing entries still support text-to-image generation through `/v1/images/generations` and `/v1/jobs` with the default task. Edit settings do not change that workflow. The requested setup model determines validation and runtime selection.

### Selected-area Qwen edits

Add `"mask_image":"/absolute/path/mask.png"` to the edit request. The mask must match the **oriented original source dimensions**, not the resized output canvas. White (255) repaints, black (0) preserves; intermediate grayscale values blend. An entirely black mask or mismatched dimensions returns **400** at generator admission; invalid opaque queue requests fail when consumed. Upload a mask PNG through the existing raw-byte **POST `/v1/reference-images`** and pass its returned local path if needed.

Selected areas are supported by **both API servers**. The generator accepts a flat request on port 5742; request model must be Qwen (set it up first):

```sh
curl --fail-with-body http://localhost:5742/v1/jobs \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: jacket-area-1' \
  -d '{"task":"edit","prompt":"Change the jacket to green.","reference_image":"/absolute/path/photo.png","mask_image":"/absolute/path/mask.png","preserve":"Face, pose and background","size":"auto","response_format":"url"}'
```

The independent queue accepts the same selection inside `request` on port 55302, even while the generator is off:

```sh
curl --fail-with-body http://localhost:55302/v1/jobs \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: jacket-area-2' \
  -d '{"model":"qwen-image-2.1-turbo","request":{"task":"edit","prompt":"Change the jacket to green.","reference_image":"/absolute/path/photo.png","mask_image":"/absolute/path/mask.png","preserve":"Face, pose and background","size":"auto"}}'
```

Both return **202** and a job ID. Poll the generator's `/v1/jobs/{id}`, or the queue's `/v1/jobs/{id}`. Run `make serve` to consume queued edits with automatic model switching. For a synchronous result, send the flat masked request to generator **POST `/v1/images/edits`** instead. Keep photo and mask files accessible on the generator's Mac through completion and any restart recovery; the independent queue does not upload, copy or serve media. Use a new idempotency key when changing the mask path or settings, and avoid modifying the referenced files while jobs wait.

Reference resolution defaults to Auto in the webpage and when `size` is omitted from a reference request on either API. `size:"auto"` also selects this mode explicitly. The generator reads the oriented source dimensions at consumption, keeps them without upscaling, and limits the long edge to 2048px. Arbitrary source dimensions (including small photos) are padded to the inference grid and trimmed afterward. Explicit dimensions still override resolution. The queue API stores Auto without opening source files.

Mask and photo are aligned, resized to the requested long edge and padded together. Qwen receives the mask during inference. The service then composites unselected source pixels back before trimming padding, preserving them exactly **at the normalized output resolution**. Original source files are unchanged. Klein models do not receive mask inputs; unsupported mask/model/task combinations return **422**. The UI provides a rectangle selection for uploaded/gallery photos; API clients may supply arbitrary masks. A path-only reference can still be edited, but must have a browser preview before drawing an area selection.

### Edit instructions and saved settings

Edit requests append deterministic scope/preservation guidance even with expansion disabled. With `expand_prompt:true`, a separate text-only editing policy refines only the requested change and preservation requirements. It does not run the generation style/composition expansion policy, inspect the photo, or add composition alternatives. `n` still requests 1–5 attempts at the same edit; `vary_prompt:true` is rejected. Explicit `image_strength` is also rejected for these instruction editors.

**POST `/v1/prompts/expand`** also accepts `task:"edit"` and optional `preserve` for an editing preview; no photo is required because it reads text only. Expansion records report `task:"edit"`, an `<model-id>-edit-v2` guidance profile, the original instruction, refined instruction and preservation details. Each generated image response reports `task`. Edit sidecars also retain `preserve`, `mask_image`, `edit_scope` and the actual effective prompt. Missing `task` on older generation metadata means generation. Queue preparation/recovery keeps selected task, source, mask and saved refinement.

## Job and queue endpoints

In this table **Both** means the endpoint exists on each port. Both services manage the shared cross-model queue. Optional model filters apply to list and clear; job-ID routes need no model query.

| Endpoint | Service | Success | Body and behavior |
| --- | --- | --- | --- |
| POST `/v1/jobs` | Both | 202 | Generator: flat image settings. Queue: `{model?,request,adapter_sha256?}`. Returns a job record after committing it. |
| GET `/v1/jobs` | Both | 200 | List jobs using optional `status`, `limit`, `offset`; returns a page. |
| GET `/v1/jobs/{identifier}` | Both | 200 | Read one job and any saved result. |
| PATCH `/v1/jobs/{identifier}` | Both | 200 | Replace a never-started waiting request. Generator: complete flat settings. Queue: `{request:{...}}`. Returns the updated job. |
| POST `/v1/jobs/{identifier}/move` | Both | 200 | Body `{"position":1}`. Move a waiting job to a one-based position across the whole waiting queue. Returns that job. |
| POST `/v1/jobs/{identifier}/run-now` | Both | 200 | No body. Move a waiting job to position 1 with immediate scheduling priority; returns that same job. |
| DELETE `/v1/jobs/{identifier}` | Both | 200 | Cancel a waiting job; returns its cancelled record. No request body. |
| DELETE `/v1/jobs` | Both | 200 | Cancel all waiting jobs across the shared queue; returns `{cancelled:integer}`. No body. |
| POST `/v1/jobs/{identifier}/retry` | Both | 202 | Retry a failed/cancelled job as a new ordinary job. No body. Returns the new record, with `retry_of`. |
| GET `/v1/queue` | Queue | 200 | Read global counts/pause state. On the generator, read `queue` in GET `/health`. |
| PATCH `/v1/queue` | Both | 200 | Body `{"paused":true}` or `{"paused":false}`. Returns counts/state. Queue response also includes `queue_scope:"shared"`. |

### Listing and pagination

| Query | Default | Rules |
| --- | --- | --- |
| `status` | omitted | `queued`, `running`, `succeeded`, `failed`, `cancelled`, or `pending` for queued + running. Omit for all history. |
| `limit` | 24 | Integer 1–100. |
| `offset` | 0 | Nonnegative integer. |
| `model` | omitted | Both ports: filter jobs by model; positions remain global. |

Order is running first, then waiting work in its saved execution order, then finished history newest first. Each waiting job exposes `queue_position`, starting at 1 across the entire waiting queue, independent of page size, offset and status filters. Position is absent/null for active and finished work. A page contains `data[]`, matching `total`, `limit`, `offset`, `has_more`, `has_previous`, and boundary offsets `next_offset` / `previous_offset`. Generator pages omit unavailable offsets; queue pages return them as null and echo the optional `model` filter.

Generator job lists omit full/partial image result bodies; fetch a job's detail for those. Queue lists return raw job records, including saved metadata when present.

```sh
curl --fail-with-body 'http://localhost:5742/v1/jobs?status=pending&limit=24&offset=0'
curl --fail-with-body 'http://localhost:55302/v1/jobs?model=chroma1-flash&limit=24&offset=0'
```

### Reordering and Run Now

**POST `/v1/jobs/{identifier}/move`** accepts exactly `{"position":N}`: a strict integer from 1 to the current waiting-job count. Move up by sending the job's current `queue_position - 1`, move down with `queue_position + 1`, or send any destination. The move and the other positions update together in one SQLite transaction, including across pagination boundaries. The response is the same job with its updated position. Moves apply to any waiting job, including an interrupted/recovered attempt, and preserve immediate status, request settings, seed, saved expansions, checkpoints and idempotency comparison.

**POST `/v1/jobs/{identifier}/run-now`** takes no body. It moves the same waiting job to position 1 and sets its top-level `generate_immediately:true`. It does not create a duplicate, start the generator, enable disabled LoRAs/LLM, bypass the shared accelerator lock or interrupt an active batch. The original `request.generate_immediately` is retained as provenance and may differ from this current scheduling flag. Repeating Run Now while the job is still waiting keeps that identity and puts it first again; once claimed it returns **409**.

Both operations return **200** after the scheduling change commits, **404** for an unknown/wrong-model ID, **409** when the target is already running/finished, and normal storage **503** errors. Move returns **422** for a missing/noninteger/nonpositive position or extra fields; a position beyond the current waiting count returns **409**. Another client or worker can change the list, so refresh after a conflict. A response may become stale as soon as the worker claims the next job.

Ordinary admissions append to the queue. New `generate_immediately:true` admissions insert immediately before the first ordinary waiting job, retaining the order of any leading immediate jobs. **Manual moves then define execution order, even across priority jobs**. While paused, only immediate jobs are eligible, in their saved relative order; Run Now makes its target eligible. Once resumed, all waiting jobs follow the saved order. A recovered active batch is returned to the front so its saved progress resumes. Order and pause state survive restarts.

The header’s **Queue** button opens a modal with active/waiting counts, pause/clear, filters and pagination. **Monitor** follows a job in the main preview; **Open image** displays completed results, and **Metrics** opens resource history. Waiting jobs have green **Run Now** and red **Remove** actions. The modal supports Close, Escape and backdrop dismissal. It provides a grip on the left of each waiting row. Drag it above/below another visible waiting job, or focus it and press ↑ / ↓ to move one position (including across page boundaries). **Run Now** follows that job's progress in the preview. Active and finished jobs have no scheduling controls. Both services use the same SQL operations; generator/UI controls work even while standalone queue HTTP is offline.

```sh
# Replace this value with a waiting job ID.
IMAGEGEN_JOB_ID="paste-the-job-id-here"
curl --fail-with-body --request POST "http://localhost:5742/v1/jobs/$IMAGEGEN_JOB_ID/move" \
  -H 'Content-Type: application/json' -d '{"position":1}'
curl --fail-with-body --request POST "http://localhost:5742/v1/jobs/$IMAGEGEN_JOB_ID/run-now"
# Job IDs address the same shared queue on either service.
curl --fail-with-body --request POST "http://localhost:55302/v1/jobs/$IMAGEGEN_JOB_ID/move" \
  -H 'Content-Type: application/json' -d '{"position":2}'
```

### Editing, cancellation, retries and pause

PATCH is a **complete replacement**, not a partial merge. It is allowed only when `status:"queued"` and `attempts:0`. Omitted settings use defaults when validated; include all settings you want to retain. ID and creation time stay the same. The model cannot change; priority and the idempotency comparison follow the new settings. Position stays the same unless `generate_immediately` changes: switching it on inserts before the first ordinary waiting job, switching it off appends to the queue. Running, previously attempted/interrupted and finished jobs return **409**.

DELETE cancels waiting work only. Deleting a running/finished job returns **409**; an unknown ID returns **404**. Clear cancels all waiting work across all models (or an explicit clear filter), preserving active work, history and saved media. There is no permanent job-history or image-deletion API.

Retry is allowed only for failed/cancelled entries (**409** otherwise), with a new ID and ordinary priority. It reuses saved settings and any frozen seed. A retry is a new full batch; completed files from the old job remain. Restore missing references/adapters first. Generator-side retry can return **400/422** if those saved settings/inputs are unusable. Repeating the retry call intentionally creates another job; it has no idempotency header parameter.

Pause persists across restarts. The active batch finishes; explicitly immediate jobs may still run while ordinary queued work is paused. No cancellation endpoint interrupts active inference.

```sh
curl --fail-with-body --request PATCH http://localhost:5742/v1/queue \
  -H 'Content-Type: application/json' -d '{"paused":true}'
```

## Job and image responses

### Job records

| Common field | Type | Meaning |
| --- | --- | --- |
| `id`, `model` | strings | Stable job ID and resolved queue/model. |
| `status` | string | `queued`, `running`, `succeeded`, `failed` or `cancelled`. |
| `request` | object | Saved settings. Queue admissions may initially lack defaults/seed; the generator freezes effective values before inference. |
| `generate_immediately` | boolean | Current scheduling priority; Run Now may change it without changing the saved request. |
| `queue_position` | integer or null | One-based position among all waiting jobs; absent/null for running/finished jobs. |
| `created`, `started`, `finished` | numbers | Unix seconds; started/finished can be null or omitted until applicable. |
| `attempts` | integer | Number of times inference ownership was claimed. |
| `retry_of` | string or null | Original ID when this record was created by retry. |
| `error` | string or null | Terminal failure or recovery message; inspect it together with status. |
| `prompt_expansion` | object or null | Saved shared expansion; queue records can also hold raw batch-planning metadata. |
| `result` | object or null | Shape depends on the service below. |

**Generator-specific fields:** `total_images` and `completed_images` are integers. `candidate_expansions[]` contains per-option plans for varied batches. Shared expansion instead uses `prompt_expansion`. `result` is present only for complete success; `partial_result` can contain saved options while running/failed. Neither pending state nor partial files establish full success.

**Queue-specific fields:** integer `sequence` records immutable admission order (use `queue_position` for waiting execution order), and nullable integer `error_status` records the failure's HTTP status. Raw `result` can hold a partial checkpoint before success. Queue responses do not provide the generator's candidate-count/partial-result wrapper. Always check **`status:"succeeded"`** before treating either response as complete.

### Generator image response

Synchronous generation returns this object directly. Async detail puts it in `result` or, for incomplete work with saved images, `partial_result`:

```json
{
  "created": 1793000000,
  "generation_seconds": 14.2,
  "data": [
    {
      "candidate_index": 1,
      "candidate_count": 1,
      "tags": ["Forest"],
      "seed": 42,
      "size": "512x512",
      "width": 512,
      "height": 512,
      "generation_size": "512x512",
      "file_path": "/absolute/store/z-image-turbo/images/0123456789abcdef0123456789abcdef.png",
      "metadata_path": "/absolute/store/z-image-turbo/images/0123456789abcdef0123456789abcdef.json",
      "url": "http://localhost:5742/images/0123456789abcdef0123456789abcdef.png"
    }
  ]
}
```

Values above are illustrative. `created` is integer Unix seconds; `generation_seconds` is total rendering time, excluding expansion time. Each `data[]` item contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `candidate_index`, `candidate_count` | integers | One-based option index and requested batch count. |
| `seed` | integer | Actual option seed. |
| `tags` | array of strings | Tags saved at generation time. Use the gallery API for later editable tags. |
| `size`, `width`, `height` | string, integers | Actual saved dimensions; references can differ from the requested canvas. |
| `generation_size` | string | Internal canvas, padded to the model's grid before reference output is trimmed. |
| `file_path`, `metadata_path` | strings | Absolute server-local PNG and JSON sidecar paths. Files are saved even for base64 output. |
| `url` | string, when requested | Download/preview URL for `response_format:"url"`. |
| `b64_json` | string, when requested | Base64 PNG for `response_format:"b64_json"`. |
| `prompt_expansion` | object, when used | The expansion actually applied to this option, described below. |

### Generate several image options

Set `n` to 2–5. The first seed is random if omitted, then persisted; option `i` uses `(first_seed + i) % 4294967296` for zero-based `i`. Images render sequentially in one durable job. With `expand_prompt:true`, expansion is shared unless `vary_prompt:true` requests distinct composition plans.

All candidate prompts are committed before rendering and reused after interruption. Recovery adopts completed files and renders missing options only. Explicit retry creates a new full batch. Different seeds/prompts can still give similar layouts for constrained prompts or strongly preserved references. See the [README example](../README.md#explore-multiple-options).

## Independent queue API (generator may be off)

The standalone service stores requests for every model in the shared SQLite database. Start it with `make queue`; it needs the Python environment but no image/LLM weights or mounted media drive for CRUD.

### Submission envelope

POST `http://localhost:55302/v1/jobs` accepts:

| Field | Type | Default and meaning |
| --- | --- | --- |
| `request` | object | Required. Image settings from the generation table. |
| `model` | string or omitted | Shared saved default if omitted. Only setup models are admitted. |
| `adapter_sha256` | string or null | Optional lowercase 64-character hex checksum. Advanced admission pin for an adapter; omit for ordinary submissions. |

```sh
curl --fail-with-body http://localhost:55302/v1/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: queued-forest-1' \
  -d '{"model":"z-image-turbo","request":{"prompt":"A watercolor fox in a forest","size":"512x512","expand_prompt":true,"response_format":"url"}}'
```

The envelope rejects unknown outer fields. Nested `request.model`, if present, must match the outer model. `request.generate_immediately` defaults false and must be a strict boolean. JSON values must be finite (no NaN/infinity). Other image settings are validated by the generator when consumed: **202 means persisted, not model-validated or generated**. Bad image settings become failed jobs without blocking later work.

For PATCH, send only the complete replacement envelope `{"request":{...image settings...}}`; address the job ID directly. Only list/clear offer an optional model filter; see the shared endpoint table.

A queue response has resolved top-level `model` and the raw job fields above. Single-image `result` is its saved generation metadata, including `file_path`, `metadata_path`, `seed`, `size`, `prompt`, `model`, `created`, `seconds` and other settings. Batch `result` is `{created,seconds,images:[...saved metadata...]}`. These raw objects have no generator download URLs or base64 field, even if `request.response_format` asks for URL format. Use the generator's detail endpoint for `result.data[]` and image URLs when it is running. A queue response's raw result may be partial before success.

One generator consumes all setup models from the shared queue. Jobs whose models are unavailable or whose LoRA/expansion capability is disabled stay waiting with an explanation while later eligible work continues. Referenced files/adapters must remain available until processing. The queue process never selects a worker, starts inference or recovers running jobs.

## Reference photos and variations

| Endpoint | Service | Success | Purpose |
| --- | --- | --- | --- |
| POST `/v1/reference-images` | Generator | 201 | Store an oriented PNG copy of uploaded photo bytes. |

Send **raw bytes**, not multipart or JSON. Documented content types are `image/png`, `image/jpeg`, `image/webp` and `application/octet-stream`. Uploads are limited to **20 MiB** and **40,000,000 decoded pixels**; exceeding either returns **413**. Invalid/unsavable images return **400**; unavailable configured storage can return **503**.

```sh
curl --fail-with-body http://localhost:5742/v1/reference-images \
  -H 'Content-Type: image/jpeg' --data-binary @/absolute/path/reference.jpg
```

Response: `{file_path:string, url:string, width:integer, height:integer}`. The PNG copy is saved in `<store>/<active-model>/references`, with orientation normalized and original metadata stripped. It is retained for reuse. The original photo is unchanged. Pass the returned `file_path` in an image request, or use an existing server-local path:

```json
{
  "prompt": "The village at dusk, watercolor washes and warm lantern light",
  "reference_image": "/absolute/path/reference.jpg",
  "image_strength": 0.6,
  "size": "1536x1536",
  "response_format": "url"
}
```

Output preserves the oriented reference's aspect ratio. Omitted `size` or `"auto"` detects source resolution without upscaling, capped at a 2048px long edge; only the larger dimension of an explicit `WIDTHxHEIGHT` sets the overridden long edge. Model padding is trimmed, so saved dimensions can be nonmultiples of 64 and the short edge can be below 256. Local-path references also have the 40-megapixel check; the 20-MiB limit applies to the upload endpoint. Z-Image/Chroma produce variations controlled by `image_strength`. Klein and Qwen instead perform **instruction-based editing**; describe the change in `prompt` and **omit `image_strength`**. The service exposes one reference. Whole-photo edits do not guarantee unchanged pixels or exact identity. Qwen photo edits can use a mask as described below. For example, while serving `flux2-klein-9b`:

```json
{
  "model": "flux2-klein-9b",
  "prompt": "Change the jacket to green. Keep the face, pose and background.",
  "reference_image": "/absolute/path/portrait.png",
  "size": "1024x1024",
  "expand_prompt": false,
  "response_format": "url"
}
```

The same shape works for `qwen-image-2.1-turbo` when that model is serving. Its default is 8 steps with the checkpoint's Turbo schedule. Use around 1024px initially; smaller Turbo outputs may lose detail. Qwen PNGs retain RGBA output. The shared prompt LLM remains text-only and does not inspect the reference.

## Local prompt expansion (enabled by default)

| Endpoint | Service | Success | Purpose |
| --- | --- | --- | --- |
| POST `/v1/prompts/expand` | Generator | 200 | Preview an expanded description without creating an image job. |

Body: required `prompt` (nonblank string, max 4000 characters), optional `lora` (ID/null), `lora_strength` (0–2/null, requires a LoRA), `task` (`generate` default or `edit`) and `preserve` (edit only, max 1000 characters). Other fields are rejected. Adapter selection accounts for its configured trigger in token budgeting.

```sh
curl --fail-with-body http://localhost:5742/v1/prompts/expand \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A watercolor fox in a snowy forest"}'
```

This endpoint is synchronous and **not queued**. It returns **409** if another preview/inference/shared GPU owner is busy; **422** if expansion is disabled, invalid or over budget; **400** for an invalid adapter; and **503** for unavailable worker/assets/storage. Retry a busy preview after the active work finishes. No image is rendered on failure.

### Expansion response

| Fields | Types | Meaning |
| --- | --- | --- |
| `original_prompt`, `expanded_prompt` | strings | Original text and text with appended visual detail. Original is retained verbatim. |
| `mode` | string | `"expand"`. |
| `model`, `revision` | strings | Prompt LLM identity, distinct from the image model. |
| `image_model` | string/null | Image backend the guidance targets. |
| `policy_version`, `policy_sha256` | integer, string/null | Instruction provenance. |
| `guidance_profile`, `guidance_version` | string/null, integer/null | Backend guidance identity/version. |
| `details` | object/null | `style_details` and `additional_composition` strings. |
| `sampling` | object/null | LLM decoding settings used. |
| `style_anchor` | string/null | Requested style span when identified. |
| `original_image_tokens`, `expanded_image_tokens`, `image_token_limit` | integers/null | Exact image-tokenizer counts for Z-Image; unknown/null for native Chroma. |
| `seconds` | number | Expansion time, additional to image rendering time. |
| `variation_index`, `variation_count`, `variation_goal` | integers/null, string/null | Per-option planning metadata for varied batches. |
| `peak_mlx_memory_gb`, `peak_process_rss_gb` | numbers/null | LLM-child peak memory in decimal GB. Counters overlap; don't add them. The image model can consume additional memory. |

Capability is on by default at startup, with `NO_LLM=1` as the opt-out. Requests still need `expand_prompt:true` (API default false). To generate from an adopted preview, use its expanded text as `prompt` with expansion false to avoid another pass. The LLM reads text only and does not inspect references.

Z-Image expansions check the exact **512-token** image budget including formatting and adapter trigger. Over-budget inputs/expansions fail; the expander never compresses or truncates them. Native Chroma and Klein/Qwen counts are not measured by the expander; the 4000-character API limit still applies. Klein/Qwen have concise, edit-preserving guidance with a 60-new-word soft cap. This is not an encoder-budget guarantee. Guidance is selected by backend; Flash currently shares HD's content guidance. The LLM uses a short-lived child, greedy non-thinking decoding, at most 320 generated tokens per expansion and a 180-second child timeout (including all options in varied planning).

Automatic expansion is part of the durable image job. Shared plans appear in `prompt_expansion`; varied plans in generator `candidate_expansions[]`; each image records its actual plan. The sidecar keeps original `prompt`, `expanded_prompt`, and `effective_prompt` including the adapter trigger. Expansion failure fails the job without silently rendering the original. See [design and research](prompt-expansion.md).

## Searchable gallery, tags and pagination

The gallery's **Edit photo** action opens the generator webpage at `/?generation=<image-id>&mode=edit`. This is a UI link, not a generation endpoint: it loads the saved PNG into the Edit preview, leaves the edit instruction blank and defaults reference resolution to Auto without submitting a job. It uses any setup editing model, independently of the model that created the PNG. It prefers an eligible original model, then an eligible saved default, then the sole setup editor; multiple remaining choices leave the selector empty. With no setup editor, the action is disabled with setup guidance, and direct links report the missing capability. Existing `mode=reference` and `mode=settings` links retain variation/settings reuse. Submission then uses the normal generator job API with `task:"edit"`, the selected editor's `model` and the PNG's `file_path` as `reference_image`.

| Endpoint | Service | Success | Purpose |
| --- | --- | --- | --- |
| GET `/v1/images` | Generator | 200 | Search/list saved images across model roots and indexed custom locations. |
| GET `/v1/images/{identifier}` | Generator | 200 | Read one image's settings, paths and URLs. |
| PATCH `/v1/images/{identifier}/tags` | Generator | 200 | Replace its editable tags; returns the updated detail. |

### GET /v1/images

| Query | Default | Rules |
| --- | --- | --- |
| `q` | empty string | Case-insensitive substring, max 4000 characters. Searches prompt, tags, model, adapter ID/name, seed and dimensions. |
| `tag` | omitted | Repeat up to 32 exact label filters. All must match, case-insensitively; combines with `q`. Same tag rules as generation. |
| `sort` | `newest` | `newest` or `oldest`. |
| `limit`, `offset` | 24, 0 | Integer limit 1–100 and nonnegative offset. |

```sh
curl --fail-with-body 'http://localhost:5742/v1/images?q=forest&tag=Winter&limit=24&offset=0&sort=newest'
```

Response has `object:"list"`, `data[]`, matching `total`, `limit`, `offset`, `has_previous`, `has_more`, and available `previous_offset` / `next_offset` (omitted at boundaries). References are excluded; no inference runs. Missing images are skipped. Valid PNGs can remain browsable even when metadata is missing/unreadable. Unavailable active image storage returns **503**.

### Gallery image objects

| Fields | Types | Meaning |
| --- | --- | --- |
| `id`, `created` | string, integer | Image ID and Unix timestamp; older images can use modification time. |
| `url`, `thumbnail_url` | strings | PNG and cached WebP preview URLs. |
| `file_path`, `file_size` | string, integer | Absolute PNG path and byte count. |
| `tags` | array of strings | Current editable tags, including later changes. |
| `metadata_available` | boolean | Whether a readable JSON sidecar exists. |
| `metadata_path`, `metadata_url` | strings, optional | Sidecar location and download URL when readable. |
| `prompt`, `model`, `size` | strings, optional | Saved generation details. |
| `width`, `height`, `seed`, `steps` | integers, optional | Image dimensions and saved generation settings. |
| `generation_seconds` | number, optional | Saved render time. |
| `lora` | object, optional | Adapter `id`, `name` and `strength` when recorded. |
| `metadata` | object, detail only | Full saved JSON generation record, when readable. Not included by the list schema. |

GET detail returns **404** for unknown/missing images. PATCH tags requires `{"tags":["Winter","Campaign"]}`; `{"tags":[]}` clears them. It uses the same normalization/limits as initial tags. Changes are stored atomically in the owning model's image index without modifying PNG/sidecar. Unknown images return **404**, invalid tags **422**, and unwritable storage **503**. No endpoint deletes images.

```sh
curl --fail-with-body --request PATCH \
  http://localhost:5742/v1/images/0123456789abcdef0123456789abcdef/tags \
  -H 'Content-Type: application/json' -d '{"tags":["Winter","Campaign"]}'
```

## Observability

These read-only endpoints are on the **generator (5742)**, with a webpage at `/observability`. Metrics read separate project-local resource history; the job-log route reads the shared queue record and attributable log files; the queue API (55302) does not expose them. Recording begins with generator startup, including model loading, and ends when it stops. Old activity cannot be backfilled.

| Method / path | Parameters and behavior |
| --- | --- |
| GET `/v1/observability` | Sampler `recording`, startup `model`, session ID, `last_sample`, optional error, source availability, cadence/retention, model IDs/names, window IDs and measurement definitions. |
| GET `/v1/observability/metrics` | `window`: `5m` (default), `30m`, `1h`, `6h`, `24h`, `3d`, `7d`. Optional `model=<id>` filters history; omission includes all models/sessions. Optional `end` is a finite current/past Unix timestamp (default now). `max_points`: 30–1200, default 600. |
| GET `/v1/observability/jobs` | Paginated measured **attempts**, newest first. Optional `model`; `limit` 1–100 (default 20), `offset` >=0 (default 0). Returns `data`, `total`, `limit`, `offset`, `has_more`. Waiting/unmeasured jobs do not appear. |
| GET `/v1/observability/jobs/{identifier}` | One job’s recorded attempts, plus its graph. Optional `model` disambiguates the model. `window`: `full` (default), `1m`, `5m`, `15m`; `max_points` as above. Short windows end at the latest completion, or now during execution. Only execution is sampled; waiting/restart intervals between attempts appear as gaps. |
| GET `/v1/observability/jobs/{identifier}/logs` | Available generation-log tails for a persisted queue job, independent of resource sampling. Optional `model` must match the job; `max_bytes` 1024–262144, default 65536 total raw bytes. At most ten runtime attempts and five native image logs. |

Example requests:

```sh
curl 'http://127.0.0.1:5742/v1/observability'
curl 'http://127.0.0.1:5742/v1/observability/metrics?window=30m&model=z-image-turbo&max_points=300'
curl 'http://127.0.0.1:5742/v1/observability/jobs?limit=20&offset=0'
curl "http://127.0.0.1:5742/v1/observability/jobs/$JOB_ID?model=z-image-turbo&window=full"
curl "http://127.0.0.1:5742/v1/observability/jobs/$JOB_ID/logs?max_bytes=65536"
```

Graph responses have `start`/`end` Unix seconds, `bucket_seconds`, `points[]` and `summary`. Each point contains the actual first/last sample timestamps, midpoint `timestamp`, sample `count`, recorded `models[]`, `sessions[]`, `stages[]`, `values` (bucket means) and `peaks` (sampled maxima). Missing counters are **null**, not zero; absent buckets are not fabricated. Means are sample means, not elapsed-time-weighted averages. The UI breaks lines across missing intervals and session/model changes.

| Metric key | Unit and scope |
| --- | --- |
| `service_cpu_percent` | Generator plus its current child processes, normalized to **total logical CPU capacity**, 0–100%. Includes prompt/native inference children and baseline/API overhead; excludes the independent queue process. |
| `system_cpu_percent` | Whole-machine CPU busy percentage, 0–100%. CPU metrics need two snapshots; the first is null. |
| `service_rss_bytes` | Summed resident memory across that generator process tree. Shared pages can be counted more than once; not unique physical memory or a model-only allocation. |
| `system_memory_used_bytes` | Whole-machine total RAM minus available RAM. |
| `system_memory_total_bytes` | Machine RAM capacity. |
| `gpu_utilization_percent` | Apple GPU driver `Device Utilization %`, **whole device**, including other apps. Not per-process GPU attribution. |
| `gpu_memory_bytes` | Whole-device driver `In use system memory`. On Apple Silicon this shares unified RAM; **do not add it to RSS/system RAM**. |

`summary` has `sample_count`, first/last sample timestamps, and one entry per metric with `mean`, `peak`, and valid-value `count`. Job detail also returns `runs[]`: `session`, `job_id`, `attempt`, actual job `model`, `started`, `finished`, `status` (`running`, `succeeded`, `failed`, `interrupted`) and that attempt’s summary. Each recovery attempt remains identifiable. A recovered previous execution closes at its last observed sample; it is not reported as successful. In full job view, the UI can combine saved attempt summaries even when raw points have expired. Short-window summaries use only retained points in that window.

Samples are taken every **1s during a job / 5s idle**, plus execution boundaries. Raw samples expire after **8 days**; finished/interrupted summaries remain in `state/observability/metrics.sqlite3`. Retention and restarts can therefore leave a job with a saved summary and no raw graph points. This database, WAL/journals and all resource history are private local state, not Git assets. No prompts, images or command lines are copied into it.

GPU counters are best-effort read-only IORegistry diagnostics, verified locally without sudo. Unavailable counters remain blank and are retried with backoff. Summed RSS can double-count shared pages; CPU snapshots can miss short-lived children and 1s sampling can miss brief spikes. Per-job plots correlate measured resources with execution and are not an exclusive GPU/cost profiler. The page’s tooltips state these limitations. [psutil documentation](https://psutil.io/) describes process/system readings; the GPU [collector example](https://github.com/XReyRobert/macos-telegraf-gpumonitoring) documents the driver counter names.

Bad models/windows/ranges return **422**. An unknown/unmeasured job returns **404**, including jobs still waiting or predating instrumentation. A missing sampler or unavailable telemetry store returns **503** with `detail.code: observability_unavailable` and `Retry-After: 2`. These failures are independent of queue storage and do not change generator readiness or prevent generation. `/health.observability` reports sampler/source availability when initialized, or a startup error if sampling could not start.

### Generation-log response

`GET /v1/observability/jobs/{identifier}/logs` returns `job_id`, `model`, queue `status`, nullable job `error`, `available`, `sources`, `max_bytes` and a `note` explaining missing/legacy files. Each source has `kind` (`runtime` or `native`), `label`, nullable `attempt` and `candidate_index` (1-based), absolute `file_path`, plain `text`, total `size_bytes`, tail `bytes_read`, `truncated`, and Unix `modified`. The budget is shared across the bounded source list; actual text can be smaller when files are absent. ANSI escapes/control characters are removed and carriage returns normalized. Live views refresh every five seconds and show latest available tails regardless of the graph time window.

New scheduler/Python logs are appended separately for each attempt under `state/job-logs/<job-id>/attempt-<number>.log`, including model selection, stages and failures. Native CLI logs remain in the configured model's `logs/` folder; retries may overwrite them, so they are labelled as latest contents without claiming an exact attempt. Job capture failure does not fail inference. Logs may contain prompts, local paths or other runtime output; keep them private. They persist until locally removed and are independent of the eight-day metrics retention.

A known job without attributable files returns **200** with `available:false` and empty `sources`. Old shared runtime logs are never exposed as a job's logs. Invalid/unknown job IDs and a mismatched valid model return **404**; unknown model or out-of-range byte limits return **422**. Queue-record/storage failure returns **503** with `detail.code:job_logs_unavailable` and `Retry-After:2`; individual missing/unreadable files leave other sources usable. Sampling and queue-HTTP outages do not prevent log reads when the shared SQLite and files are readable. This endpoint does not accept a file path or mutate queue state.

## Media, webpages and API documentation

The URLs below are read-only. They are useful for previews/downloads and cover routes that may not appear in OpenAPI. Follow returned URLs when possible.

| Endpoint | Service | Success | Content |
| --- | --- | --- | --- |
| GET `/images/{identifier}.png` | Generator | 200 | `image/png` for a saved generation, including an indexed custom destination. |
| GET `/images/{identifier}.json` | Generator | 200 | `application/json` download of its original sidecar. |
| GET `/thumbnails/{identifier}.webp` | Generator | 200 | Cached `image/webp` preview; private cache lifetime 86400 seconds. Returned URLs can include a cache-version `v` query. |
| GET `/references/{identifier}.png` | Generator | 200 | `image/png` preview of an uploaded reference. |
| GET `/` | Generator | 200 | Generator webpage, `text/html`. |
| GET `/gallery` | Generator | 200 | Searchable gallery webpage, `text/html`. |
| GET `/observability` | Generator | 200 | Resource history and measured-job webpage, `text/html`. |
| GET `/assets/observability.css` | Generator | 200 | Resource-page stylesheet, `text/css`. |
| GET `/assets/observability.js` | Generator | 200 | Local interactive resource graphs, `text/javascript`. |
| GET `/assets/help.css` | Generator | 200 | Shared tooltip stylesheet, `text/css`. |
| GET `/assets/help.js` | Generator | 200 | Shared tooltip interactions, `text/javascript`. |
| GET `/assets/shell.css` | Generator | 200 | Common header and queue modal styling, `text/css`. |
| GET `/assets/shell.js` | Generator | 200 | Shared runtime status, queue controls and polling, `text/javascript`. |
| GET `/docs` | Both | 200 | Swagger UI for that service. |
| GET `/redoc` | Both | 200 | Alternative API reference UI. |
| GET `/openapi.json` | Both | 200 | Machine-readable OpenAPI schema for that service. |
| GET `/docs/oauth2-redirect` | Both | 200 | Framework-provided Swagger helper page; no authentication is configured. |

The four framework documentation routes (`/docs`, `/redoc`, `/openapi.json`, `/docs/oauth2-redirect`) also support HEAD for headers without the response body.

Missing/invalid media IDs or unreadable metadata return **404**. PNGs/custom destinations remain resolvable after restarts while their files/indexes remain available. URLs require a running generator. The queue API has no root webpage, gallery, uploads or media routes; requesting them there returns **404**. Generator, gallery and observability pages share the generator header, live status and queue modal. Queue actions call the same generator APIs on every page. Monitor/Open image/Run Now from another page open `/?job=<job-id>` to show or follow that job; this UI link does not submit or rerun work by itself. UI pages, shared assets and catalog JSON use `Cache-Control:no-store`.

## Errors and safe retries

Most application errors have `{"detail":"message"}`. Request-schema errors use FastAPI's `{"detail":[...]}` validation entries (including location, message and type). Endpoint success status alone does not establish generator readiness or job completion.

| Status | Typical meaning | What to do |
| --- | --- | --- |
| 200 | Successful read/update or synchronous completion. A job read can still have `status:"failed"`. | Inspect `status`/`error` on job and health records. |
| 201 | Reference copy stored. | Use returned `file_path`. |
| 202 | Job committed, including retry or idempotent resubmission. | Poll its ID until terminal state. |
| 400 | Unreadable reference, bad/uninstalled adapter, unsavable upload, or cancelled synchronous job. | Correct input/files and inspect `detail`. |
| 404 | Unknown ID, missing media or unsupported route. | Check the service port, selected model and ID. |
| 405 | HTTP method is not supported at that path. | Check the endpoint table. |
| 409 | Edit/cancel/retry conflicts, reused idempotency key with different settings, or busy prompt preview. | Respect job state; use a new key only for new work, or retry a preview later. |
| 413 | Reference upload exceeds byte/pixel limits. | Resize/re-encode the source. |
| 422 | Schema/model/range/combination error, disabled optional capability, or failed expansion. | Correct fields/mode. Queue image validation can be deferred and reported on the job instead. |
| 503 | Queue storage/consumer, media store, catalog or required assets unavailable. | Inspect health, paths/mounts and the specific error; preserve accepted work. |
| 500 | Unhandled/backend generation failure. | Inspect job error/logs before resubmitting. |

### Idempotency-Key

Optional on both services' **POST `/v1/jobs`**, and generator **POST `/v1/images/generations`** / **POST `/v1/images/edits`**. It accepts 1–128 ASCII letters/digits and `._:-`. Keys are scoped to the model database. The same key/settings returns the original job; changed settings return **409**. PATCH associates that key with the replacement settings.

After an uncertain submission, retry the **same API route, key and settings**. A lost response is not proof that no job was committed. Different keys create new work. Client disconnect/timeouts do not cancel accepted jobs. The synchronous route waits on the same durable worker; the async route is preferable for polling and large/slow batches.

## If the standalone queue API goes down

The browser calls generator APIs; those handlers use the shared SQLite store directly. Submission, queue controls, polling, saved results, the gallery and accepted generation continue while the generator and local storage are healthy. There is no HTTP write fallback, second database or automatic queue-service restart.

`queue_api` in GET `/health` uses a bounded one-second probe and reports a recovery notice without disabling controls or marking an otherwise healthy generator unready. Run `make queue` with the original `QUEUE_PORT` override to restore outside-agent access, without restarting the generator or resubmitting jobs.

A genuine generator SQLite failure instead returns **503**, `Retry-After: 2`, and:

```json
{
  "detail": "Project-local queue storage is unavailable; check its path and permissions, then retry with the same Idempotency-Key for unconfirmed submissions.",
  "code": "queue_storage_unavailable"
}
```

Other 503 errors do not necessarily include that code/header. The queue API's storage error uses a `detail` message. Fix local storage/permissions before retrying; keep the same idempotency key for an unconfirmed submission.

## Persistent generation queue

The shared SQLite database is **`<project>/state/queue/jobs.sqlite3`**, with journals, backups and worker leases alongside. `DATA_DIR` relocates weights/media, not SQLite. Both HTTP services share this state; only the leased generator owns inference and running-job recovery. There is no worker registration/selection API.

Graceful shutdown finishes the active batch and leaves waiting jobs persisted. Abrupt shutdown recovery reuses job IDs, frozen defaults/seeds, saved expansions and completed options, rendering only missing images. Sampling of an interrupted image restarts; intermediate diffusion state is not saved. Matching fully published PNG/sidecars can be adopted after a crash before the success commit. Completed history is not rerun automatically. Native children retain ownership leases until they finish.

References, adapters and output/index files are external to SQLite and must remain accessible. A missing media SSD blocks generation without creating fallback downloads; standalone queue CRUD remains available. Failed work does not retry forever or block subsequent valid jobs. GPU-lock contention waits with pending jobs still removable; it does not unload idle resident models. `GPU_LOCK_FILE` or ignored `storage.json` setting `gpu_lock_file` selects the shared lock; the portable default is `<store>/.model-services/gpu.lock`.

### Queue durability and backups

The new shared queue starts empty. Restart both services against version4; no old per-model database migration or API compatibility is provided. SQL uses short transactions and `synchronous=FULL`; inference runs outside database transactions. Back up project `state/` using a stopped service or SQLite's backup API, together with media/references/indexes. This protects work across process restarts; it does not replace backups for disk failure. A source-only clone starts with an empty shared queue.

### Runtime dispatch and setup readiness

The generator processes eligible jobs in saved global order, one complete batch at a time, and never groups jobs by model behind the caller’s back. Switching stops the old backend child before launching another. Consecutive Z-Image jobs reuse the resident model; native Chroma/Klein/Qwen retain per-image loading. The HTTP API remains available while `queue.stage` reports loading/expansion/rendering. A backend failure fails that job and later eligible jobs continue. Shutdown waits for the current batch, then releases the owned runtime.

A previously queued job whose model files are missing remains queued; both job APIs expose `blocked_reason` for missing model files; generator responses also explain disabled optional capabilities. Disabled LLM/LoRA requirements also remain queued rather than silently changing their settings. Models missing setup cannot be newly submitted or retried until restored. Both services can still read/cancel existing history if model storage is unavailable. The queue API uses no inference imports and never starts a model. The fresh shared queue intentionally does not import former per-model databases; existing media/catalogs are retained.

Every generator image result includes `model` in addition to its absolute `file_path` and `metadata_path`. Preview expansion accepts optional `model`, using that model’s guidance/tokenizer; omission uses the shared default. Discovery is the source of per-model limits/defaults. To update the saved default, run `make serve MODEL=<ready-id>`; an otherwise matching running generator is reused even if busy, without changing already-resolved job models. Startup mode changes still require it to be idle.

The webpage enables **Edit photo** when at least one setup editor exists. Its model choice starts empty on the first visit; select an editing model before sending a request. Later visits restore that tab's selection. An empty edit selector never falls back to the default generation model. With no setup editor, the tab stays disabled and its tooltip explains which models to set up. Generate and Edit retain separate in-memory drafts, references, masks, expansion previews and selected results. Followed jobs update the tab matching their request's `task`, even while the other tab is visible; the queue itself remains shared. Ordinary tab switches never transfer photos. Explicit result/gallery reuse can prefill the target workflow. Form drafts are not persisted across reloads; accepted jobs remain durable.
