---
name: z-image-turbo
description: Use Z-Image Turbo through the shared local Apple Silicon image model server, with generation, photo variations and a tagged gallery.
---

# Z-Image Turbo

Use the shared project at `~/Documents/models/ImageGen-Service` on the original Mac, or the provided checkout elsewhere. Read its `README.md` and `skill/image-model-server/SKILL.md` for setup, API, storage and catalog details. Run `make setup MODEL=z-image-turbo`, then `make serve MODEL=z-image-turbo` in a retained session; `NO_LORAS=1` disables adapters. Use `http://127.0.0.1:5732/` by default, or the explicit PORT override consistently. Verify `/health` is ready with model `z-image-turbo`. Keep weights/media in the configured store and SQLite queues in project-local `state/queues/<model-id>/`; preserve existing jobs, paths, images and tags. The API/UI are shared with Chroma; select the model at startup, not per request.

Text-only prompt expansion capability is on by default with make serve (terminal Enter/Y on, N off). Use NO_LLM=1 to opt out (the Python CLI uses --no-llm). Verify health, then use expand_prompt:true. Read the shared service skill for preview, exact512-token checks, persisted original/expanded text and restart behavior. Default API expansion is off.

Image options: n is a strict integer1–5, default1. The UI provides a count selector and candidate thumbnails; API clients iterate data[] (or completed job result.data[]). Each option uses a distinct persisted seed; expansion is shared by default in the API, or distinct per option with expand_prompt:true,vary_prompt:true and n>=2. The UI enables varied compositions for eligible batches. All planned prompts persist before rendering. Turbo has limited seed-only diversity; inspect the actual outputs rather than assuming unique seeds guarantee distinct compositions. See the shared skill for progress and recovery.


## Independent queue service

`make queue` starts/reuses the lightweight background queue API at `http://127.0.0.1:55302` without model/LLM/SSD/GPU requirements (the Python environment must be installed). `make serve` ensures it is running, then starts the foreground generator/UI on5732. Ctrl-C stops the generator after active work; the queue API stays available. `make stop-queue` stops only that verified queue process, retaining jobs. Set `QUEUE_PORT=...` consistently for all three commands; no OS login service is installed. The UI/gallery/uploads/LLM run only on the generator.

For offline-agent admission use POST `http://127.0.0.1:55302/v1/jobs` with `{"model":"z-image-turbo","request":{"prompt":"A misty forest","expand_prompt":true,"response_format":"url"}}`. **Omitted model defaults to z-image-turbo**, regardless of active generator; explicit supported model IDs choose that queue, even before its assets are set up. List/detail/PATCH/DELETE/clear/retry/pause on that port take optional `?model=<id>` (same default); responses include resolved model. The generator consumes only its startup model's project-local SQLite queue. There is no worker registry/selection. Queue API restart never recovers running work; the generator recovers it under its exclusive lease.

Queue-only admission validates the envelope/routing/finite JSON/strict immediate flag; generation fields are validated by the consumer. 202 is not generation success. Missing defaults/random seeds and adapter checksums are frozen once before expansion/inference. Invalid image settings fail visibly without blocking later jobs. PATCH `/v1/jobs/{id}?model=<id>` with `{"request":{...complete replacement...}}` edits only queued entries with attempts=0; it preserves ID/order, updates priority/idempotency settings, cannot change model, and returns409 for running/interrupted/finished jobs. DELETE/clear preserve history/media; retry makes a new FIFO job. Idempotency keys are scoped to model.

Queue-port results are raw saved metadata, possibly partial before success, with local paths (batch images[]). No image URLs/base64/gallery are served there. Poll until succeeded before using the full result. The generator's existing flat request API and image-specific `result.data[]` responses on5732 remain compatible and proxy CRUD to the queue API. Its omitted model still means its startup model. Read both live `/openapi.json` schemas. Preserve user jobs/data; stop agent-started test services after verification unless asked to leave them running.
