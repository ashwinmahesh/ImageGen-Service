---
name: z-image-turbo
description: Use Z-Image Turbo through the shared local Apple Silicon image model server, with generation, photo variations and a tagged gallery.
---

# Z-Image Turbo

Use the shared project at `~/Documents/models/ImageGen-Service` on the original Mac, or the provided checkout elsewhere. Read its `README.md` and `skill/image-model-server/SKILL.md` for setup, API, storage and catalog details. Run `make setup MODEL=z-image-turbo`, then `make serve MODEL=z-image-turbo` in a retained session; `NO_LORAS=1` disables adapters. Use `http://127.0.0.1:5732/` by default, or the explicit PORT override consistently. Verify `/health` is ready with model `z-image-turbo`. Keep weights/media in the configured store and SQLite queues in project-local `state/queues/<model-id>/`; preserve existing jobs, paths, images and tags. The API/UI are shared with Chroma; select the model at startup, not per request.

Optional text-only prompt expansion: start with WITH_LLM=1, verify health, then use expand_prompt:true. Read the shared service skill for preview, exact512-token checks, persisted original/expanded text and restart behavior. Default API expansion is off.

Image options: n is a strict integer1–5, default1. The UI provides a count selector and candidate thumbnails; API clients iterate data[] (or completed job result.data[]). Each option uses a distinct persisted seed; prompt expansion is shared once per batch. See the shared skill for progress and recovery.
