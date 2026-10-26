---
name: image-model-server
description: Set up and use the shared local Apple Silicon image service for Z-Image Turbo, Chroma1-HD, or Chroma1-Flash, including reference variations and the searchable tagged gallery.
---

# Operate ImageGen-Service

Use this skill to set up or call the service. For repository edits, follow `AGENTS.md` in the checkout. The human-facing `README.md` provides the quick start; `docs/setup.md` covers advanced configuration; `docs/api.md` covers request fields, results, queue semantics and recovery. Read the reference relevant to the task. Each live `/openapi.json` is authoritative for its service. No Codex plugin, shell alias or chat history is required.

Locate the provided checkout containing `Makefile`, `server.py` and `model_profiles.py`. On the original Mac it is `~/Documents/models/ImageGen-Service`, with model/media storage at `/Volumes/ASH-SSD/models`. Elsewhere use the supplied checkout and chosen store. Do not recreate retired per-model source projects. All backends share one source tree and environment.

## Setup and startup

Run `make setup` in a terminal: Enter chooses Z-Image Turbo, then project-local storage on a fresh checkout. Another absolute folder creates reusable per-model symlinks. Automation: `make setup MODEL=<id> DATA_DIR=/absolute/store NONINTERACTIVE=1`; omit DATA_DIR to reuse saved/default storage. IDs: `z-image-turbo`, `chroma1-hd`, `chroma1-flash`. DATA_DIR is the parent store containing model folders, not one weights folder. Reuse existing assets; fail on missing mounts instead of creating fallback data. Setup downloads selected image assets and the shared prompt LLM, no LoRAs. Realism is the only optional public preset.

Start `make serve MODEL=<id>` in a retained foreground session when requested. Plain `make serve` numbers ready models only, lists unready setup commands and prefers ready Z-Image on Enter. Serving never downloads. LLM capability defaults on; Enter/Y keeps it, N disables it for that run. `NO_LLM=1` is the single startup opt-out. `NO_LORAS=1` disables adapters independently. Missing requested assets fail before replacing an idle server. Never silently change modes or interrupt busy/unrelated services.

The generator/UI is `http://127.0.0.1:5732/`, with `/gallery`, `/docs` and `/health`. Use PORT overrides consistently. Wait for ready and confirm model, paths, `loras_enabled` and `prompt_expansion.enabled`; initial loading may precede the HTTP listener. One startup-selected backend serves requests. An identical managed instance may be reused. Ctrl-C finishes the active batch and stops the generator; the queue stays up.

`make queue` starts/reuses a detached API at `http://127.0.0.1:55302`, using the installed environment without model/LLM/SSD/GPU requirements. `make serve` ensures it is up too. `make stop-queue` stops only the verified queue process, preserving jobs. Use QUEUE_PORT overrides consistently. No OS autostart is installed.

## Choose the API

| API | Request and default model | Results |
| --- | --- | --- |
| Generator, port 5732 | POST `/v1/jobs` with flat image settings. Omitted model uses its startup model; a different explicit model returns422. | Completed `result.data[]` has paths and URLs/base64. |
| Queue, port 55302 | POST `/v1/jobs` with `{"model":"z-image-turbo","request":{...image settings...}}`. Omitted model always means Z-Image Turbo. | Raw persisted metadata, local paths; batch `images[]`. No media/UI routes. |

For both, HTTP 202 means committed, not generated. Preserve the ID and poll GET `/v1/jobs/{id}` until `succeeded`, `failed` or `cancelled`. Only succeeded is a complete result; partial checkpoints may exist earlier. Reuse the same `Idempotency-Key` and settings after uncertain submission; changed settings with that key return409. Prefer `response_format:"url"` on the generator to avoid embedding PNGs repeatedly, and iterate every result item.

Example generator body:

```json
{"prompt":"A watercolor fox beneath an oak tree","n":3,"expand_prompt":true,"vary_prompt":true,"response_format":"url"}
```

For the queue API, wrap it in `request`. List/detail/update/cancel/clear/retry/pause routes there take optional `?model=<id>` (default Z-Image). GET `/v1/models` lists all supported IDs even before setup. GET `/v1/loras?model=<id>` discovers compatible adapters without a generator; select available IDs and optional strength. Presence is not checksum verification or a generator-mode check. A missing configured volume/bad catalog can return503 while queue CRUD remains usable. Generator `/v1/loras` describes its own active model and enabled mode.

Queue admission validates routing/JSON/priority; the consumer later validates image settings and freezes defaults/seeds/adapter snapshots before inference. Other-model jobs and requests requiring disabled optional capabilities wait for compatible startup. Inspect failed job errors; HTTP200 from a status read is not generation success. The synchronous `/v1/images/generations` waits on the same durable worker and needs a long timeout. Disconnecting never cancels persisted work.

## Generation controls

- `n`: integer 1–5, default1. Sequential options share settings; seeds increment from the persisted first seed modulo 2^32. Omit seed for a random first seed. Each result has seed, paths and candidate index/count.
- `expand_prompt:true`: API defaultfalse; enabled UI initiallychecked. Text-only LLM appends visual details preserving original text/style/constraints. Preview with POST `/v1/prompts/expand` when useful. Adopting preview text should turn expansion off to avoid a second pass.
- `vary_prompt:true`: requires expansion and n>=2, giving each option a saved composition expansion. UI enables it for eligible batches. All expansions persist before rendering; recovery reuses saved options/prompts/seeds and renders only missing images. Constrained prompts/strong references can limit visible variety.
- `size`: default1024x1024, dimensions multiple64 from256–2048. Default/max steps: Z9/12, HD40/80, Flash8/40. Prompt max4000characters; Z's backend processes512tokens including formatting/trigger. Expanded Z prompts use exact tokenizer budget checks and fail rather than compress/truncate. Chroma native token counts are unavailable.
- `reference_image`: absolute server-local path, or upload raw bytes to POST `/v1/reference-images` (not multipart;20MiB/40MP). Output follows oriented reference aspect ratio, using size's larger dimension as long edge. Trimmed dimensions need not be multiples64. `image_strength` [0,1), default.6; higher preserves more, at least2steps required. This is variation, not precise editing/inpainting.
- `lora` / `lora_strength`: one available compatible ID, strength0–2. Configured triggers are inserted automatically; don't add `<lora:...>` prompt tags. Base-only mode rejects new adapter requests.
- `output_dir`: absolute server-local folder or `~/...`, otherwise `<store>/<model>/images`. PNG and JSON sidecar are always saved; original references are unchanged. Paths belong to the server, not the caller.

Read `docs/api.md` for remaining fields, errors and provenance. Negative prompt, CFG, custom sampler/scheduler and mask controls are not exposed. No model-specific option namespace exists yet.

## Queue and gallery

POST `/v1/jobs` defaults to the queue end; UI defaults `generate_immediately:true` (next free slot, never preemption). New immediate admissions insert before the first ordinary job, after any leading immediate jobs; explicit moves define saved execution order. GET `/v1/jobs?status=pending&limit=24&offset=0` lists active/waiting work; omit status for history, limit1–100. Lists omit image blobs. PATCH a never-started queued job with a **complete replacement request** (flat on generator, `{request:...}` on queue). Attempts>0/nonqueued returns409; model cannot change.

POST `/v1/jobs/{id}/move` with `{"position":N}` moves a waiting job to a one-based position across the entire waiting queue, independent of pagination. Read `queue_position` first; add/subtract 1 to move down/up. Manual moves may cross immediate jobs and preserve request/priority/checkpoints. POST `/v1/jobs/{id}/run-now` (no body) moves that same job first and enables immediate scheduling, even while paused. It never interrupts an active batch, starts an offline generator or enables missing capabilities. The current top-level priority may differ from the unchanged request's original flag. Both operations return200; active/finished jobs or stale out-of-range positions409, unknown/wrong-model404, malformed moves422. Recovered waiting attempts can move without changing frozen progress. Queue-port operations accept `?model=<id>`; generator operations stay direct SQL during queue HTTP outages. The UI exposes drag grips, keyboard ↑/↓ and Run Now.

DELETE `/v1/jobs/{id}` cancels waiting only; DELETE `/v1/jobs` cancels all waiting work for that model, preserving running work/history/media. POST `/v1/jobs/{id}/retry` creates a new ordinary identity from a failed/cancelled entry. PATCH `/v1/queue` with `{"paused":true}` pauses ordinary work only; immediate jobs can still run. Generator counts/stage/errors appear in `/health.queue`.

GET `/v1/images` on the generator browses all model roots/indexed custom destinations: `q`, repeated exact `tag` filters, `limit`, `offset`, `sort`. Pagination includes totals/navigation; generator null offsets are omitted. GET `/v1/images/{id}` reads settings. PATCH `/v1/images/{id}/tags` with `{"tags":[...]}` replaces labels (`[]` clears); at most32 labels, each1–64 normalized characters, without commas/control characters. Edits do not alter original PNG/sidecar. Gallery reuse fills the form for review rather than submitting.

## State, outages and checks

SQLite is project-local at `state/queues/<model-id>/jobs.sqlite3`, with journals/backups/leases alongside; DATA_DIR never relocates it. Both services use the same JobStore directly. The browser calls only generator APIs and keeps working if standalone queue HTTP stops. `/health.queue_api` reports available/url/recovery_command/error through a bounded one-second probe; the notice is informational and controls stay usable. Restore outside-agent access with `make queue`, no generator restart. Local SQLite failures instead return503/code=`queue_storage_unavailable`/Retry-After2; retain an unconfirmed submission's idempotency key.

Only the leased generator recovers interrupted jobs; restarting queue HTTP does not reclaim work. Graceful shutdown finishes the batch. Abrupt recovery reuses saved settings/seeds/expansions and adopts matching committed artifacts. GPU-lock contention leaves pending work waiting without unloading idle weights. Preserve references/adapters until consumption and back up project `state/` with media/references/indexes. Keep weights/media/private adapters/settings/SQLite/symlinks outside Git.

`make test` uses fakes; `make smoke` requires a running model and creates a real512px image. Do not start models merely to inspect docs/catalogs or change user queues for tests. This user starts ImageGen manually: stop generator/queue instances you started for testing afterward unless asked to keep them up; preserve pre-existing services and active work. Report actual checks, mode, server state and completed output paths.
