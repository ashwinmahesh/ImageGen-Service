---
name: z-image-turbo
description: Use Z-Image Turbo through the shared local Apple Silicon image model server, with generation, photo variations and a tagged gallery.
---

# z-image-turbo

Use the shared ImageGen-Service checkout, at `~/Documents/models/ImageGen-Service` on the original Mac or the provided folder elsewhere. Read `skill/image-model-server/SKILL.md` and `docs/api.md` there for operating details; `README.md` and `QUICKSTART.md` are human guides. Do not recreate a separate per-model serving project.

Set up with `make setup MODEL=z-image-turbo`. When serving is requested, `make serve MODEL=z-image-turbo` chooses the shared saved default; one generator handles requests for **all setup models** and switches automatically between complete batches. Use a retained foreground session. LLM capability defaults on; `NO_LLM=1` is its sole opt-out. `NO_LORAS=1` independently disables adapters. Verify `/health` reports ready, `service:imagegen-generator`, `queue_scope:shared` and the intended `default_model`; `active_model` may be null before the first job or differ while another model runs. The generator/UI uses localhost5742 with consistent PORT overrides.

Submit flat generator settings with `model:"z-image-turbo"` to POST `/v1/jobs`, or use the independent queue API at localhost55302 with `{"model":"z-image-turbo","request":{...image settings...}}`. Omission uses the saved default on **both** APIs; it is not tied to whichever backend is active. GET `/v1/models` lists only setup models. GET `/v1/loras?model=z-image-turbo` reports compatible adapters. Setup/readiness never implies automatic adapter downloads.

There is one project-local SQLite queue at `state/queue/jobs.sqlite3`, with global ordering/pause/Run Now/moves and one consumer lease. List/clear default to all models and offer an optional model filter; job-ID CRUD needs no model query. Existing queued models that become unavailable stay blocked with an explanation while other eligible jobs continue. HTTP202 means committed, not finished; poll the returned ID. Generator results use `result.data[]`; queue results are raw metadata/paths. The generator/UI accesses SQLite directly and works when queue HTTP is offline. `make queue` starts that independent API; Ctrl-C stops the generator after its active batch while the queue stays up. `make stop-queue` stops only the verified queue process.

Preserve configured external model/media/reference/catalog paths and project-local state. Keep generated images, model/adapters and personal catalogs out of Git. Stop maintenance test services you started unless asked to leave them running; preserve pre-existing services and active work.
