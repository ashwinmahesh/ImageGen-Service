---
name: z-image-turbo
description: Generate images with the user's locally installed Z-Image Turbo server on this Mac, including reference-photo variations and custom output folders. Use when the user chooses local Z-Image Turbo, not for unrelated image-generation services.
---

# Local Z-Image Turbo

Project: `/Volumes/ASH-SSD/models/z-image-turbo`. Read its `README.md` for the full API contract and current setup; if the service is running, `/openapi.json` is authoritative for accepted fields.

Check `http://127.0.0.1:8000/health`. If unavailable, verify the SSD is mounted and start `make serve` from the project in a retained terminal. Do not launch a duplicate model instance: this Mac has 24 GB of memory. The shell alias `serve-z-image-turbo` may be unavailable in noninteractive shells. The UI is `/`, interactive API docs `/docs`. Chroma1-HD shares port 8000; verify the health model ID. `make serve` can switch from an idle managed Chroma service but refuses to interrupt a generation.

Send JSON to `POST http://127.0.0.1:8000/v1/images/generations`. A useful default is:

```json
{"prompt":"Describe the desired final image here","size":"1024x1024","steps":9,"response_format":"url"}
```

Use 512×512 for a quick draft. Dimensions are multiples of 64, 256–2048 each. The UI includes 1536 and 2048 sizes. Above 1024, VAE tiling is automatic; larger sizes cost more time/memory. Prefer 1024 for a first pass and 1536 for higher resolution on this 24 GB Mac. Only `n:1` is supported. Supply an integer seed to reproduce a variation; otherwise the server selects and returns one. The API accepts 4,000 characters, but the installed tokenizer truncates at 512 tokens including chat formatting; keep essential details first.

For a reference photo, add `reference_image` with one existing absolute local path and `image_strength:0.6`. Higher values preserve MORE of the original and use fewer denoising steps; valid range is 0 inclusive to 1 exclusive. At least 2 steps are required with a reference. This is image-to-image initialization, not precise editing or guaranteed identity preservation. The output aspect ratio follows the oriented source automatically; only the longer requested size dimension sets its long edge. The generation field accepts a file path, not a URL, mask, or list. The webpage has a visible Choose photo picker. To upload programmatically, POST raw image bytes (not multipart) to `/v1/reference-images` (20 MiB, 40 megapixels maximum), then use the returned `file_path` as `reference_image`. Uploaded RGB PNG copies are kept in the project’s `references/` folder; existing local paths avoid an extra copy.

To refine a previously generated image, use its returned `file_path` as `reference_image` in a new generation request. Describe the desired final result and choose preservation strength; each pass saves a new output. The UI’s Make a variation button sets this up directly. This is whole-image refinement; masks and precise instruction editing are not supported.

Default save directory: `/Volumes/ASH-SSD/models/z-image-turbo/images`. If the user chooses another destination, supply `output_dir` as an absolute local directory (`~/` supported). Missing directories are created; unique PNG and JSON filenames are generated. No output filename parameter exists.

Allow up to 30 minutes for the HTTP request. Only one generation runs at a time. On 429, wait for health `busy:false`; on 400/422 fix the reported input. A timeout/disconnect does not cancel generation: check health and the destination for a newly saved image/sidecar before retrying. On 500 inspect logs rather than repeatedly resubmitting.

Read `data[0].file_path` and `metadata_path` from the response; both exist even with `response_format:b64_json`. Verify the PNG and show it using its absolute local path when the client supports image display. Return the actual path and seed, not an invented filename. The `url` works only while the server and saved file are available. The JSON sidecar retains the prompt and generation settings for reproducibility.


For adapters, inspect `GET /v1/loras`: choose an `available:true` ID only when `enabled:true`. Send `lora:<id>` and optional `lora_strength` (0–2, catalog defaults currently 0.8); omit/null disables adapters. One per request. Required trigger text is added automatically; `<lora:...>` prompt tags are rejected. Base-only mode rejects non-null adapters with 422. Adapter or strength changes reload clean model weights.

With `reference_image`, preserve its EXIF-oriented aspect ratio automatically: `size:1024x1024` means a 1024px long edge, not a square crop. The UI offers Reference resolution. The server pads a temporary input to the model grid and trims only that padding from the result; output short edges are rounded to whole pixels and can be below 256 or not divisible by 64. Local references are limited to 40 MP. Read actual `data[0].size`, `width`, `height`; `generation_size` is the internal canvas. Sidecars also record `requested_size` and oriented `reference_size`. Without a reference, size is used unchanged.

For a base-only server, use `serve-z-image-turbo-no-loras` or `make serve-no-loras` (`make serve NO_LORAS=1` is equivalent). The CLI flag is `--no-loras`. Check `/health` `loras_enabled`; false means no adapter can be requested, the UI hides adapter controls, and `/v1/loras` has `enabled:false` and no entries. Non-null `lora` requests fail with 422. Normal `make serve` restores adapter support, switching an idle server if necessary; never interrupt a busy generation.
