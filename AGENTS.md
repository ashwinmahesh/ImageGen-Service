# Working on ImageGen-Service

This file is for agents changing the repository. An agent operating the service should read `skill/image-model-server/SKILL.md`. Keep `README.md` written for people: explain what the project does and the implemented techniques behind its image workflow; put step-by-step usage in `QUICKSTART.md`. Detailed contracts belong in `docs/api.md` and `docs/setup.md`.

## Project layout

- `server.py`: generator FastAPI app, validation, media, gallery and direct SQLite queue endpoints.
- `queue_api.py` / `queue_launch.py`: independent queue API and background-process launcher.
- `job_queue.py`: shared SQLite `JobStore`, ordering, mutations and durable checkpoints.
- `generation_worker.py` / `queue_storage.py`: exclusive consumer, recovery, accelerator coordination and project-local database migration.
- `model_profiles.py` / `model_assets.py` / `config.py` / `launch.py`: backend defaults, pinned snapshot preflight and startup.
- `photo_workflows.py`: common generate/edit task semantics and preservation scope; masked geometry/compositing lives in `reference_geometry.py`.
- `mflux_engine.py`: offline, per-image low-RAM CLI children for FLUX.2 Klein and Qwen-Image 2.1 Turbo; `engine.py` retains the resident Z-Image path.
- `storage_paths.py`, setup/download scripts: portable storage selection and reproducible assets.
- `adapter_catalog.py` / `lora_catalog.py`: queue discovery and generator adapter validation.
- `prompt_expander.py` / `prompt_policy.py` / `enhancer_profile.py`: local prompt expansion; consult `docs/prompt-expansion.md` before changing policy.
- `ui/index.html` / `ui/gallery.html` / `image_catalog.py`: browser UI and catalog.
- `tests/test_*.py`: fake-backend, storage, API, queue and recovery tests.
- `tests/smoke_test.py`: explicit real HTTP generation check, run by `make smoke` rather than unit-test discovery.

## Architecture to preserve

There are two services: queue API on loopback port **55302**, generator/UI on **5732**. They share per-model SQLite databases. The browser calls generator APIs whose handlers access `JobStore` directly. Do not proxy those operations through the standalone queue API. Its outage is informational for the generator: submission, polling, queue controls, results and generation continue. Use a bounded health probe. Local SQLite failures are different (`503`, `code:queue_storage_unavailable`, `Retry-After:2`).

The queue service owns no GPU, model loading, UI or running-job recovery. It accepts `{model,request}`; omitted model defaults to `z-image-turbo`. Model discovery lists all supported models, even before setup. LoRA discovery reads metadata/presence without loading or hashing weights; a missing store must not prevent unrelated queue CRUD.

The generator selects one model at startup and consumes only its queue under an exclusive lease. Its request body is flat; omitted model means the startup model, and a different model is rejected. No worker registry, heartbeat, remote worker selection or per-request model switching is needed. Preserve both API shapes and their different result formats; see `docs/api.md`.

SQLite databases, journals, backups and worker leases stay under `<project>/state/queues/<model-id>/`, never the external model store. `DATA_DIR` changes weights/media storage, not SQLite. Both services use the same short atomic transactions; inference runs outside them. Only the leased generator recovers interrupted jobs. Preserve schema checks and guarded legacy migration.

A batch contains 1–5 sequential images. Persist effective defaults, adapter snapshots, seeds and all planned expansions before rendering. Recovery reuses them and completed artifacts, rendering only missing options. HTTP 202 is acceptance, not completion; partial checkpoints are not full success. Preserve idempotent resubmission. Immediate priority means the next free slot without interrupting an active batch. Pause affects ordinary work only. Persist explicit one-based waiting positions independently of immutable admission sequence. Both APIs support POST job move and run-now through the same atomic JobStore methods. Manual moves override priority order; Run Now puts the same queued identity first and enables immediate scheduling without changing request/fingerprint/checkpoints. Never move or preempt running work. Schema 4 migrates earlier priority order without recovering active jobs on queue startup. The UI uses left-hand drag grips plus keyboard arrows, and labels the panel Queue. Delete/clear cancel waiting entries while preserving history/media. Request replacement is allowed only before the first attempt.

Keep the shared accelerator lock separate from the queue worker lease. Native children retain ownership until they exit. Preserve existing user jobs and processes when testing or switching modes.

## Setup and startup

`make setup` lists all supported models (Enter: Z-Image Turbo), then asks for storage (Enter: project-local). External choices create reusable per-model symlinks. Never silently download to a fallback when a configured volume is missing, overwrite conflicting data or recreate retired per-model source projects. Chroma Flash shares the HD encoder/VAE without downloading HD's diffusion checkpoint.

`make serve` numbers only ready models; list unready models separately with setup commands. Label editing-capable model names `(edits enabled)` and mask-capable models `(edit selected area enabled)`, using registry capabilities in both sections. Serving never downloads assets. Validate the model/mode before replacing an idle managed server; never replace a busy or unrelated listener. Start/reuse the detached queue API, then run the generator in the foreground. Ctrl-C finishes active work; the queue service stays up. `make stop-queue` stops only the verified queue process.

LLM capability defaults on, with an interactive `[Y/n]` choice. `NO_LLM=1` is the single startup opt-out (`--no-llm` in Python); don't add duplicate flags. API requests default `expand_prompt:false`; the enabled UI checks it by default. Preserve original text/style/constraints, validate expansion budgets and persist provenance. The text-only LLM subprocess exits before rendering. `vary_prompt:true` requires expansion and at least two images. Unique seeds don't guarantee visible diversity.

FLUX.2 Klein 4B uses the pinned Q8 snapshot; Klein 9B and Qwen-Image 2.1 Turbo use pinned Q4 snapshots. Require every indexed shard and Qwen's vision encoder/processor/Turbo schedule before marking setup complete or listing them as ready. Keep child offline/local-model flags and inherited queue/GPU lock descriptors. Both generation and instruction editing are supported; references use the editing pipeline. Keep `task:generate` backward compatible and available for every model. Explicit `task:edit` and `/v1/images/edits` require a source and eligible model, use the dedicated editing-refinement policy, and reject composition variation. Qwen alone accepts aligned masks; pass them through to inference and composite unselected source pixels back at output resolution. Preserve task/mask/preservation/refinement through queue recovery and gallery reuse. The UI filters model choices by task but does not hot-switch the startup backend. Reject `image_strength` on these instruction editors and hide the variation control in the UI. Keep new quantization/license metadata under the exact model-ID namespace and preserve it when recording expansion provenance.

Only Realism is an optional public adapter preset. Default setup downloads no LoRAs. Keep private catalogs, weights and preset names out of tracked files and download defaults. The loader inserts configured triggers; don't require `<lora:...>` prompt tags.

Use accessible switches for boolean UI settings, labeled **Prompt expansion** for the local LLM control. Keep all help icons as grey circled question marks, and inset select chevrons from the right border. Preserve native input checked/disabled semantics and keyboard control.

Disable the Edit photo mode button when the selected model's tasks omit edit. Its hover/focus tooltip must say “Switch to a model that supports edits”. Recompute capability restrictions after model changes and submission cleanup; unsupported models must never enter edit mode through another UI path.

Selected-area API edits must work through both the generator's flat request and the standalone queue's `request` envelope. Keep queue admission independent of source/mask reads and defer model/media validation to consumption. Document complete examples in both schemas/references and verify masks survive preparation, checkpoints and recovery without re-rendering committed artifacts.

## Verification and data boundaries

From the project root, use `make test` for routine checks: fake downloads/inference, no model load. Tests form a package; run a focused module with `.venv/bin/python -m unittest tests.test_queue -v`. Keep cross-test imports under `tests` and resolve source/assets relative to the project root, one directory above the test files. Run focused tests while iterating and relevant complete checks for behavior changes. Documentation-only edits need link/example/schema review rather than GPU generation. `make smoke` requires a running generator and creates a real image; report it separately from unit tests.

Use benign browser fixtures and label simulated screenshots/timings. Do not change user queues, use private media, start a second GPU model or restart busy services just to test. Preserve unrelated and pre-existing user-started processes. After maintenance, stop generator/queue instances you started unless asked to leave them running; this user starts ImageGen manually.

Git contains source, docs and the four approved screenshots under `docs/screenshots/`. Keep weights, runtime binaries, LoRAs, references, outputs, sidecars, indexes/tags, SQLite state, settings, symlinks, environments, caches, logs and credentials ignored. Inspect staged paths before committing. Preserve actual data on cleanup; a source-only clone starts with an empty queue.

## Documentation responsibilities

Update these together when their behavior changes:

- `README.md`: human project overview, supported capabilities and high-level image techniques; link the step-by-step quick start and detailed references.
- `QUICKSTART.md`: step-by-step human setup, webpage/API walkthroughs and concise setting explanations; keep examples and defaults current.
- `docs/setup.md`: exact setup, storage and startup behavior.
- `docs/api.md`: human-readable specifications for every endpoint on both services, including methods, ports, request/response fields, errors, media routes and recovery. Check its endpoint inventory against the registered routes and both live schemas when APIs change.
- `skill/image-model-server/SKILL.md`: portable operating instructions; keep the Z-Image wrapper consistent.
- `AGENTS.md`: contributor guidance and architectural constraints.

For reusable workflow changes, update the user's installed `model-server-gen` skill while preserving other-modality guidance. Synchronize existing Codex/Claude copies and portable exports when maintaining them; repository edits alone do not update installed skills. Do not publish private data or rewrite published history without explicit authorization.
