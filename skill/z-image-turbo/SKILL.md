---
name: z-image-turbo
description: Use Z-Image Turbo through the shared local Apple Silicon image model server, with generation, photo variations and a tagged gallery.
---

# Z-Image Turbo

Use the shared project at `~/Documents/models/image-model-server` on the original Mac, or the provided checkout elsewhere. Read its `README.md` and `skill/image-model-server/SKILL.md` for setup, API, storage and catalog details. Run `make setup MODEL=z-image-turbo`, then `make serve MODEL=z-image-turbo` in a retained session; `NO_LORAS=1` disables adapters. Verify `/health` is ready with model `z-image-turbo`. Keep weights and private data in the configured store; preserve existing jobs, paths, images and tags. The API/UI are shared with Chroma; select the model at startup, not per request.
