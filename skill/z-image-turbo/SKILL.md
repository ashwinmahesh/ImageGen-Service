---
name: z-image-turbo
description: Use Z-Image Turbo through the shared local Apple Silicon image model server, with generation, photo variations and a tagged gallery.
---

# Z-Image Turbo

Use the shared ImageGen-Service checkout, at `~/Documents/models/ImageGen-Service` on the original Mac or the provided folder elsewhere. Read `skill/image-model-server/SKILL.md` there for operating instructions and `docs/api.md` for request/result details. `README.md` is the human setup guide; `AGENTS.md` covers code changes. Do not recreate a separate Z-Image source project.

For setup use `make setup MODEL=z-image-turbo`, then start `make serve MODEL=z-image-turbo` in a retained session when requested. Confirm `/health` ready with model `z-image-turbo` at `http://127.0.0.1:5732`. Prompt LLM capability defaults on; `NO_LLM=1` opts out independently of `NO_LORAS=1`. API expansion still requires `expand_prompt:true`.

Prefer the generator's POST `/v1/jobs` with flat image settings, `response_format:"url"` and an idempotency key. Poll until succeeded and iterate `result.data[]` for full paths/URLs. Z defaults9steps (max12), n1–5, prompt limit4000characters and backend512tokens including formatting/trigger. For varied batches use n>=2, expand_prompt:true, vary_prompt:true; different seeds alone don't guarantee visible diversity. References produce aspect-preserving variations, not precise edits.

For submission while generation is off, `make queue` starts the separate port55302 API. POST `/v1/jobs` there with `{"model":"z-image-turbo","request":{...settings...}}`; model may be omitted because its default is Z-Image Turbo. Other queue routes use optional `?model=z-image-turbo`. Queue-side results are raw metadata rather than generator `result.data[]`. Generator/UI APIs access shared project-local SQLite directly and continue if standalone queue HTTP stops. Read the shared skill for queue CRUD, LoRA discovery, gallery/tags/pagination and outages.

Keep SQLite under project `state/`, weights/media in the chosen store, and preserve user jobs/data. Stop test services you start afterward unless asked to keep them running; preserve pre-existing services and active generation.
