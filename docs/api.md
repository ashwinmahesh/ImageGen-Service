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
- [Media, webpages and API documentation](#media-webpages-and-api-documentation)
- [Errors and safe retries](#errors-and-safe-retries)
- [Outages](#if-the-standalone-queue-api-goes-down) and [restart recovery](#persistent-generation-queue)

## Services and conventions

| Service | Default base URL | Start it | Available while the generator is off? |
| --- | --- | --- | --- |
| **Generator** | `http://localhost:5732` | `make serve` | No. It owns image generation, the webpage, uploads and gallery. |
| **Queue** | `http://localhost:55302` | `make queue` | Yes. It accepts and manages persisted jobs without loading models. |

`make serve` starts the queue API if necessary. `PORT` changes the generator port; `QUEUE_PORT` changes the queue port. Use overrides consistently. Both bind to loopback, accept trusted local clients and have no API-key authentication. No permissive cross-origin access is configured. Local file paths refer to the server's Mac.

Unless stated otherwise, send JSON with `Content-Type: application/json`; responses are JSON. Request-body booleans must be JSON `true`/`false`. Generation integers such as `n`, `steps` and `seed` must be numbers, not strings or booleans. Unknown fields are rejected on typed request bodies. The queue's nested `request` is deliberately open for later generator validation.

Supported model IDs are `z-image-turbo`, `chroma1-hd` and `chroma1-flash`:

- **Generator:** omitted `model` means its startup-selected model. An explicit different model returns **422**; switch models at startup.
- **Queue:** omitted `model` always means **`z-image-turbo`**, independent of the running generator. Submission selects it in the JSON envelope; other operations select it with `?model=<id>`.

Path placeholders such as `{identifier}` mean a returned job or image ID: 32 lowercase hexadecimal characters. Job and image IDs can differ in a batch; use returned URLs and gallery IDs. Timestamps are Unix seconds; job timestamps may include fractions. Optional response fields may be null or omitted depending on the route. Examples show representative values and may omit optional fields.

## Health, models and LoRAs

| Endpoint | Service | Success | Purpose |
| --- | --- | --- | --- |
| GET `/health` | Both | 200 | Process, model and queue state. |
| GET `/v1/models` | Both | 200 | Discover supported model IDs. |
| GET `/v1/loras` | Both | 200 | Discover compatible configured adapters. |

### GET /health

No body or query parameters.

**Generator response:**

| Field | Type | Meaning |
| --- | --- | --- |
| `status` | string | `loading`, `ready` or `error`. Check this value as well as HTTP status. Initial model loading may happen before the listener is available. |
| `model`, `project_dir`, `data_dir` | strings | Active model, checkout and active model's data folder. |
| `busy` | boolean | Image inference, prompt preview or a running queue job is active. |
| `loras_enabled` | boolean | Whether this generator accepts adapters. |
| `prompt_expansion` | object | Availability, mode, LLM `model`/`revision`/`license`, `text_only`, `guidance_profile`, `batch_variations` and usage `note`. |
| `capabilities` | object | Backend/name, default/max steps, default size, dimension/prompt/count limits, adapter/LLM mode, queue URL and persistent/immediate support. The active model-ID key contains `prompt_expansion_guidance`. |
| `queue` | object, when initialized | Counts/pause state below, plus `stage`, optional `blocked_reason` and consumer `error`. |
| `queue_api` | object | Standalone queue availability: `available`, `url`, `recovery_command`, optional `error`. Its outage alone does not make the generator unready. |
| `progress` | object, when supported | Backend-specific inference progress while busy. Its fields depend on the backend. |

The generator's `capabilities` object exposes these named controls:

| Keys | Types | Meaning |
| --- | --- | --- |
| `model`, `name`, `backend` | strings | Selected backend identity. |
| `default_steps`, `max_steps`, `default_size` | integers, string | Defaults and step ceiling for the active model. |
| `prompt_note`, `max_prompt_characters`, `min_dimension`, `max_dimension` | string, integers | Prompt behavior and canvas limits. |
| `default_image_count`, `max_image_count` | integers | Batch default and ceiling. |
| `loras_enabled`, `persistent_queue`, `immediate_generation` | booleans | Available modes. |
| `prompt_expansion`, `queue_api_url` | object, string | LLM capability metadata and standalone queue origin. |
| `<active-model-id>` | object | `prompt_expansion_guidance` with profile `id`, `version`, `soft_max_new_words`, nullable `image_token_limit`, `verified_token_count` and `sources[]`. |

**Queue response:** `status`, `service:"imagegen-queue"`, `project_dir`, `default_model:"z-image-turbo"` and `version`. This checks the queue HTTP process; it does not load a model or guarantee that a later storage write will succeed.

Queue counts/state contain integer `queued`, `running`, `succeeded`, `failed`, `cancelled`; boolean `paused`; string `database_path`; nullable `next_job_id`; and boolean `next_requires_llm` / `next_requires_lora`. Generator stages are `idle`, `waiting_for_accelerator` and `generating`.

### GET /v1/models

No body or query parameters.

- **Generator:** `{object:"list", active_model, data:[...]}`. Each entry has `id`, `object:"model"`, `owned_by:"local"`, `name`, `active`, `downloaded`, `default_steps` and `max_steps`. `downloaded` checks the completion marker; it is not a full startup integrity check.
- **Queue:** `{default_model, data:[{id,name}, ...]}`. Lists every supported model even before setup, so tools can route future work.

### GET /v1/loras

The **queue** accepts optional query `model` with default `z-image-turbo`. The **generator** uses its active model, with no model selector.

| Response | Shape |
| --- | --- |
| Generator | `{object:"list", enabled:boolean, data:[...]}`. `NO_LORAS=1` returns `enabled:false` and an empty list. |
| Queue | `{object:"list", model, directory, data:[...]}`. `directory` is the absolute adapter folder. No `enabled` flag: generator mode is independent. |

Each entry describes `id`, `name`, `filename`, compatible `models[]`, `available`, optional `default_strength`, `trigger_prompt`, `source` and `sha256`; creator metadata may add fields. Select an entry with **`available:true`** and send its `id` as `lora`. Null/omitted `lora` uses the base model.

Availability checks file presence and recorded checksum metadata. Inference verifies the actual checksum. Discovery does not load/hash weights or download files. A model's local catalog overrides the shipped public catalog; compatible missing files remain listed as unavailable. Chroma HD and Flash share the HD adapter directory with compatibility filtering. Queue discovery returns **422** for unknown models and **503** for an unavailable configured volume or invalid/unreadable catalog; independent queue CRUD remains available.

## Generation API

These image settings are accepted directly by the generator's **POST `/v1/jobs`**, **PATCH `/v1/jobs/{identifier}`** and **POST `/v1/images/generations`**. On the standalone queue API they go inside `request`.

Only `prompt` is required for a valid image request. Defaults below apply when the generator validates the request; the queue API can accept an opaque request earlier.

| Field | Type | Default and rules |
| --- | --- | --- |
| `model` | string | Generator's active model. Must match it. On queue submissions use the outer `model`; a nested value, if present, must match. |
| `prompt` | string | Required, nonblank, 1–4000 characters. `<lora:...>` tags are rejected; use adapter fields. Z-Image processes the first 512 tokens including formatting/trigger, so put essentials first. |
| `size` | string | `"1024x1024"`. Format `WIDTHxHEIGHT`; each dimension a multiple of 64 from 256–2048. With a reference, the larger dimension sets its long edge and the source sets the aspect ratio. |
| `n` | integer | 1–5, default 1. Sequential options in one job. |
| `steps` | integer | Model default/range in the table below. Reference generation requires at least 2. |
| `seed` | integer or null | Omit/null for random; otherwise 0–4294967295. First option's seed; later options increment modulo 2^32. |
| `expand_prompt` | boolean | false in API; enabled UI initially checks it. Requires LLM-enabled serving. Saves the original and expanded text. |
| `vary_prompt` | boolean | false. Requires `expand_prompt:true` and `n>=2`; creates a distinct expansion for each option. |
| `generate_immediately` | boolean | false for async jobs, true for synchronous generation. True gets the next free slot before ordinary FIFO work; never interrupts active work. |
| `response_format` | string | `"b64_json"` or `"url"`; default `"b64_json"`. URL format is convenient for polling and downloads. Both save files. |
| `output_dir` | string | `<store>/<active-model>/images`. Absolute folder or `~/...`, 1–4096 characters; missing folders are created if writable. Unique files do not overwrite originals. |
| `reference_image` | string or null | Omitted/null for text-only generation. One server-local absolute/`~/...` path, max 4096 characters; HTTP URLs are not accepted. |
| `image_strength` | number | 0.6; from 0 inclusive to 1 exclusive. Only send it with a reference. Higher preserves more of the reference and runs fewer denoising steps. |
| `lora` | string or null | Omitted/null for none. Compatible installed catalog ID; lowercase letters, digits and hyphens, starting alphanumeric, max 80 characters. One adapter. |
| `lora_strength` | number or null | Requires `lora`; 0–2. Omit/null for its catalog default (Realism: 0.8). |
| `tags` | array of strings | `[]`. Up to 32 labels, each 1–64 characters after whitespace normalization. No commas/control characters; case-insensitive duplicates are removed. |

| Model | Default steps | Allowed steps | Effective runtime sampling |
| --- | ---: | ---: | --- |
| `z-image-turbo` | 9 | 1–12 | MFLUX Turbo defaults. |
| `chroma1-hd` | 40 | 1–80 | Euler, CFG 3. |
| `chroma1-flash` | 8 | 1–40 | Heun, CFG 1, beta schedule, flow shift 1. |

Negative prompts, CFG/guidance, custom samplers/schedulers, masks and instruction-editing options are **not exposed** as request fields. No model-specific request-option namespace is implemented. Larger images/options/step counts cost more time and memory. Z-Image uses VAE tiling above 1024; Chroma always uses it.

### Submit and poll an image

```sh
curl --fail-with-body http://localhost:5732/v1/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: forest-1' \
  -d '{"prompt":"A watercolor fox in a snowy forest","size":"512x512","response_format":"url"}'
```

The **202** response is a job record with `id`. Read **GET `/v1/jobs/{identifier}`** until `status` is `succeeded`, `failed` or `cancelled`. For example, replace the value below with the returned ID:

```sh
IMAGEGEN_JOB_ID="paste-the-returned-id-here"
curl --fail-with-body "http://localhost:5732/v1/jobs/$IMAGEGEN_JOB_ID"
```

A completed generator job returns images in **`result.data[]`**. An accepted or running job is not a finished image.

### Generate and wait synchronously

| Endpoint | Service | Success | Purpose |
| --- | --- | --- | --- |
| POST `/v1/images/generations` | Generator | 200 | Persist the same job, wait for completion and return an image response directly. |

Accepts the full image-settings object and optional `Idempotency-Key` header. The only default difference is `generate_immediately:true`. Response is `{created, generation_seconds, data:[...]}`, without the job wrapper. The active model is used if omitted.

```sh
curl --fail-with-body --max-time 1800 http://localhost:5732/v1/images/generations \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: forest-sync-1' \
  -d '{"prompt":"A watercolor fox in a snowy forest","size":"512x512","response_format":"url"}'
```

Use a long client timeout; disconnecting does not cancel the persisted job. Failed generation returns an HTTP error (the recorded error status, or 500); cancellation returns 400. Prefer async jobs when you need a promptly returned ID and restart-safe polling. This endpoint is a limited compatibility route, not a claim of full OpenAI API compatibility.

## Job and queue endpoints

In this table **Both** means the endpoint exists on each port. The generator operates only on its active model. Queue-port operations accept optional `?model=<id>` with default `z-image-turbo`.

| Endpoint | Service | Success | Body and behavior |
| --- | --- | --- | --- |
| POST `/v1/jobs` | Both | 202 | Generator: flat image settings. Queue: `{model?,request,adapter_sha256?}`. Returns a job record after committing it. |
| GET `/v1/jobs` | Both | 200 | List jobs using optional `status`, `limit`, `offset`; returns a page. |
| GET `/v1/jobs/{identifier}` | Both | 200 | Read one job and any saved result. |
| PATCH `/v1/jobs/{identifier}` | Both | 200 | Replace a never-started waiting request. Generator: complete flat settings. Queue: `{request:{...}}`. Returns the updated job. |
| DELETE `/v1/jobs/{identifier}` | Both | 200 | Cancel a waiting job; returns its cancelled record. No request body. |
| DELETE `/v1/jobs` | Both | 200 | Cancel all waiting jobs for that model; returns `{cancelled:integer}`. No body. |
| POST `/v1/jobs/{identifier}/retry` | Both | 202 | Retry a failed/cancelled job as a new ordinary FIFO job. No body. Returns the new record, with `retry_of`. |
| GET `/v1/queue` | Queue | 200 | Read that model's counts/pause state. On the generator, read `queue` in GET `/health`. |
| PATCH `/v1/queue` | Both | 200 | Body `{"paused":true}` or `{"paused":false}`. Returns counts/state. Queue response also includes `model`. |

### Listing and pagination

| Query | Default | Rules |
| --- | --- | --- |
| `status` | omitted | `queued`, `running`, `succeeded`, `failed`, `cancelled`, or `pending` for queued + running. Omit for all history. |
| `limit` | 24 | Integer 1–100. |
| `offset` | 0 | Nonnegative integer. |
| `model` | `z-image-turbo` | Queue port only; selects a supported model's database. |

Order is running first, then waiting work in execution order (immediate priority, FIFO within priority), then finished history newest first. A page contains `data[]`, matching `total`, `limit`, `offset`, `has_more`, `has_previous`, and boundary offsets `next_offset` / `previous_offset`. Generator pages omit unavailable offsets; queue pages return them as null and include top-level `model`.

Generator job lists omit full/partial image result bodies; fetch a job's detail for those. Queue lists return raw job records, including saved metadata when present.

```sh
curl --fail-with-body 'http://localhost:5732/v1/jobs?status=pending&limit=24&offset=0'
curl --fail-with-body 'http://localhost:55302/v1/jobs?model=chroma1-flash&limit=24&offset=0'
```

### Editing, cancellation, retries and pause

PATCH is a **complete replacement**, not a partial merge. It is allowed only when `status:"queued"` and `attempts:0`. Omitted settings use defaults when validated; include all settings you want to retain. ID/order/creation time stay the same. The model cannot change; priority and the idempotency comparison follow the new settings. Running, previously attempted/interrupted and finished jobs return **409**.

DELETE cancels waiting work only. Deleting a running/finished job returns **409**; an unknown ID returns **404**. Clear cancels all waiting work in that model's queue, preserving active work, history and saved media. There is no permanent job-history or image-deletion API.

Retry is allowed only for failed/cancelled entries (**409** otherwise), with a new ID and ordinary FIFO priority. It reuses saved settings and any frozen seed. A retry is a new full batch; completed files from the old job remain. Restore missing references/adapters first. Generator-side retry can return **400/422** if those saved settings/inputs are unusable. Repeating the retry call intentionally creates another job; it has no idempotency header parameter.

Pause persists across restarts. The active batch finishes; explicitly immediate jobs may still run while ordinary FIFO work is paused. No cancellation endpoint interrupts active inference.

```sh
curl --fail-with-body --request PATCH http://localhost:5732/v1/queue \
  -H 'Content-Type: application/json' -d '{"paused":true}'
```

## Job and image responses

### Job records

| Common field | Type | Meaning |
| --- | --- | --- |
| `id`, `model` | strings | Stable job ID and resolved queue/model. |
| `status` | string | `queued`, `running`, `succeeded`, `failed` or `cancelled`. |
| `request` | object | Saved settings. Queue admissions may initially lack defaults/seed; the generator freezes effective values before inference. |
| `generate_immediately` | boolean | Saved priority. |
| `created`, `started`, `finished` | numbers | Unix seconds; started/finished can be null or omitted until applicable. |
| `attempts` | integer | Number of times inference ownership was claimed. |
| `retry_of` | string or null | Original ID when this record was created by retry. |
| `error` | string or null | Terminal failure or recovery message; inspect it together with status. |
| `prompt_expansion` | object or null | Saved shared expansion; queue records can also hold raw batch-planning metadata. |
| `result` | object or null | Shape depends on the service below. |

**Generator-specific fields:** `total_images` and `completed_images` are integers. `candidate_expansions[]` contains per-option plans for varied batches. Shared expansion instead uses `prompt_expansion`. `result` is present only for complete success; `partial_result` can contain saved options while running/failed. Neither pending state nor partial files establish full success.

**Queue-specific fields:** integer `sequence` preserves order, and nullable integer `error_status` records the failure's HTTP status. Raw `result` can hold a partial checkpoint before success. Queue responses do not provide the generator's candidate-count/partial-result wrapper. Always check **`status:"succeeded"`** before treating either response as complete.

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
      "url": "http://localhost:5732/images/0123456789abcdef0123456789abcdef.png"
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

The standalone service routes settings to a model's SQLite database. Start it with `make queue`; it needs the Python environment but no image/LLM weights or mounted media drive for CRUD.

### Submission envelope

POST `http://localhost:55302/v1/jobs` accepts:

| Field | Type | Default and meaning |
| --- | --- | --- |
| `request` | object | Required. Image settings from the generation table. |
| `model` | string | `z-image-turbo`. A supported model can be queued before setup. |
| `adapter_sha256` | string or null | Optional lowercase 64-character hex checksum. Advanced admission pin for an adapter; omit for ordinary submissions. |

```sh
curl --fail-with-body http://localhost:55302/v1/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: queued-forest-1' \
  -d '{"model":"z-image-turbo","request":{"prompt":"A watercolor fox in a forest","size":"512x512","expand_prompt":true,"response_format":"url"}}'
```

The envelope rejects unknown outer fields. Nested `request.model`, if present, must match the outer model. `request.generate_immediately` defaults false and must be a strict boolean. JSON values must be finite (no NaN/infinity). Other image settings are validated by the generator when consumed: **202 means persisted, not model-validated or generated**. Bad image settings become failed jobs without blocking later work.

For PATCH, send only the complete replacement envelope `{"request":{...image settings...}}`; select the queue with `?model=<id>`. Other queue operations also use that query; see the shared endpoint table.

A queue response has resolved top-level `model` and the raw job fields above. Single-image `result` is its saved generation metadata, including `file_path`, `metadata_path`, `seed`, `size`, `prompt`, `model`, `created`, `seconds` and other settings. Batch `result` is `{created,seconds,images:[...saved metadata...]}`. These raw objects have no generator download URLs or base64 field, even if `request.response_format` asks for URL format. Use the generator's detail endpoint for `result.data[]` and image URLs when it is running. A queue response's raw result may be partial before success.

Only the selected model's generator consumes its queue. Jobs requiring disabled LoRAs or expansion stay waiting until that capability is enabled or the job is cancelled; an incompatible head job can hold up later work. Referenced files/adapters must remain available until processing. The queue process never selects a worker, starts inference or recovers running jobs.

## Reference photos and variations

| Endpoint | Service | Success | Purpose |
| --- | --- | --- | --- |
| POST `/v1/reference-images` | Generator | 201 | Store an oriented PNG copy of uploaded photo bytes. |

Send **raw bytes**, not multipart or JSON. Documented content types are `image/png`, `image/jpeg`, `image/webp` and `application/octet-stream`. Uploads are limited to **20 MiB** and **40,000,000 decoded pixels**; exceeding either returns **413**. Invalid/unsavable images return **400**; unavailable configured storage can return **503**.

```sh
curl --fail-with-body http://localhost:5732/v1/reference-images \
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

Output preserves the oriented reference's aspect ratio. Only the larger dimension of `size` sets the long edge. Model padding is trimmed, so saved dimensions can be nonmultiples of 64 and the short edge can be below 256. Local-path references also have the 40-megapixel check; the 20-MiB limit applies to the upload endpoint. These are variations, with no masks or reliable instruction-based localized editing.

## Local prompt expansion (enabled by default)

| Endpoint | Service | Success | Purpose |
| --- | --- | --- | --- |
| POST `/v1/prompts/expand` | Generator | 200 | Preview an expanded description without creating an image job. |

Body: required `prompt` (nonblank string, max 4000 characters), optional `lora` (ID/null) and `lora_strength` (0–2/null, requires a LoRA). Other fields are rejected. Adapter selection accounts for its configured trigger in token budgeting.

```sh
curl --fail-with-body http://localhost:5732/v1/prompts/expand \
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

Z-Image expansions check the exact **512-token** image budget including formatting and adapter trigger. Over-budget inputs/expansions fail; the expander never compresses or truncates them. Native Chroma counts are unknown; the 4000-character API limit still applies. Guidance is selected by backend; Flash currently shares HD's content guidance. The LLM uses a short-lived child, greedy non-thinking decoding, at most 320 generated tokens per expansion and a 180-second child timeout (including all options in varied planning).

Automatic expansion is part of the durable image job. Shared plans appear in `prompt_expansion`; varied plans in generator `candidate_expansions[]`; each image records its actual plan. The sidecar keeps original `prompt`, `expanded_prompt`, and `effective_prompt` including the adapter trigger. Expansion failure fails the job without silently rendering the original. See [design and research](prompt-expansion.md).

## Searchable gallery, tags and pagination

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
curl --fail-with-body 'http://localhost:5732/v1/images?q=forest&tag=Winter&limit=24&offset=0&sort=newest'
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
  http://localhost:5732/v1/images/0123456789abcdef0123456789abcdef/tags \
  -H 'Content-Type: application/json' -d '{"tags":["Winter","Campaign"]}'
```

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
| GET `/docs` | Both | 200 | Swagger UI for that service. |
| GET `/redoc` | Both | 200 | Alternative API reference UI. |
| GET `/openapi.json` | Both | 200 | Machine-readable OpenAPI schema for that service. |
| GET `/docs/oauth2-redirect` | Both | 200 | Framework-provided Swagger helper page; no authentication is configured. |

The four framework documentation routes (`/docs`, `/redoc`, `/openapi.json`, `/docs/oauth2-redirect`) also support HEAD for headers without the response body.

Missing/invalid media IDs or unreadable metadata return **404**. PNGs/custom destinations remain resolvable after restarts while their files/indexes remain available. URLs require a running generator. The queue API has no root webpage, gallery, uploads or media routes; requesting them there returns **404**. Generator/gallery pages and catalog JSON use `Cache-Control:no-store`.

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

Optional on both services' **POST `/v1/jobs`**, and generator **POST `/v1/images/generations`**. It accepts 1–128 ASCII letters/digits and `._:-`. Keys are scoped to the model database. The same key/settings returns the original job; changed settings return **409**. PATCH associates that key with the replacement settings.

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

Each model's SQLite database is **`<project>/state/queues/<model-id>/jobs.sqlite3`**, with journals, backups and worker leases alongside. `DATA_DIR` relocates weights/media, not SQLite. Both HTTP services share this state; only the leased generator owns inference and running-job recovery. There is no worker registration/selection API.

Graceful shutdown finishes the active batch and leaves waiting jobs persisted. Abrupt shutdown recovery reuses job IDs, frozen defaults/seeds, saved expansions and completed options, rendering only missing images. Sampling of an interrupted image restarts; intermediate diffusion state is not saved. Matching fully published PNG/sidecars can be adopted after a crash before the success commit. Completed history is not rerun automatically. Native children retain ownership leases until they finish.

References, adapters and output/index files are external to SQLite and must remain accessible. A missing media SSD blocks generation without creating fallback downloads; standalone queue CRUD remains available. Failed work does not retry forever or block subsequent valid jobs. GPU-lock contention waits with pending jobs still removable; it does not unload idle resident models. `GPU_LOCK_FILE` or ignored `storage.json` setting `gpu_lock_file` selects the shared lock; the portable default is `<store>/.model-services/gpu.lock`.

### Existing queue migration and backups

Older `<store>/<model-id>/queue/jobs.sqlite3` databases migrate at generator startup under both worker leases. Stop the old worker and let native inference finish first. Migration uses SQLite backup including committed WAL data, checks integrity/content and retains a project-local `legacy-backup-*.sqlite3` before retiring the old database. Differing source/destination databases cause a clear error without overwriting or merging them; identical copies after interruption are recognized.

Schema 3 adds deferred preparation checkpoints and upgrades schemas 1/2 without changing job IDs/order or already-frozen settings. Older servers refuse schema 3. SQL uses short transactions and `synchronous=FULL`; inference runs outside database transactions. Back up project `state/` using a stopped service or SQLite's backup API, together with media/references/indexes. This protects work across process restarts; it does not replace backups for disk failure. A source-only clone starts with an empty queue.
