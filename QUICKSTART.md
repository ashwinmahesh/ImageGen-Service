# Quick start: generate your first image

Complete steps 1–3, then use the **webpage** in step 4 or the **API** in step 5. Step 6 shows how to queue work while the image generator is off. Everything runs locally on your Mac.

## 1. Check the prerequisites

You need an **Apple Silicon Mac**, Git, Make and **Python 3.14**. The provided setup uses Apple's Metal GPU; it does not support Windows or NVIDIA/CUDA. The bundled Chroma runtime targets macOS 26.6.2 on Apple Silicon; other versions may need a compatible build.

If Git or Make is missing, install Apple's command line tools:

```sh
xcode-select --install
```

If you use Homebrew, install Python with:

```sh
brew install python@3.14
```

## 2. Download the project and set up a model

Open Terminal and run:

```sh
git clone https://github.com/ashwinmahesh/ImageGen-Service.git
cd ImageGen-Service
make setup
```

If you already have the project, open Terminal in that folder and run `make setup`.

Follow the setup questions:

1. **Choose a model.** Press Enter for **Z-Image Turbo**, or select Chroma1-HD or Chroma1-Flash.
2. **Choose a storage folder.** Press Enter to keep model downloads and pictures inside the project. To use an external SSD, enter an absolute path such as `/Volumes/MySSD/models`.
3. **Wait for setup to finish.** It installs dependencies and downloads the chosen image model plus a shared Qwen prompt-expansion model. Setup does not start either server.

The external folder is a parent store: setup creates model folders inside it and links them into the project. Your choice is remembered, so later setup runs reuse it. Keep the drive connected when generating. Queue databases always stay inside the project's `state/` folder.

If setup cannot find Python, rerun it with the full executable path:

```sh
make setup BOOTSTRAP_PYTHON=/opt/homebrew/bin/python3.14
```

## 3. Start the generator

From the project folder, run:

```sh
make serve
```

1. Select an installed model. Enter prefers Z-Image Turbo when it is ready; models you haven't set up appear separately with installation commands.
2. At **Enable local LLM prompt expansion? [Y/n]**, press Enter to keep expansion available. A small AI text model on your Mac can add visual detail to your image descriptions before generation; you can turn this off for each image.
3. Wait for startup to finish. Keep this terminal open while using the generator.

You now have two local services:

| Address | Purpose |
| --- | --- |
| [localhost:5732](http://localhost:5732/) | Generator, webpage, gallery and image API. Uses the selected model. |
| [localhost:55302/docs](http://localhost:55302/docs) | Lightweight queue API. Keeps accepting jobs when the generator is off. |

`make serve` starts the queue service automatically if needed. To check generator readiness from another terminal in the project folder, run `make health`; look for `"status":"ready"`. The browser/API may be unreachable while the model is initially loading.

### Startup choices

These settings are chosen before the generator starts:

| Choice | Example | What it does |
| --- | --- | --- |
| Model | `make serve MODEL=z-image-turbo` | Skips the model menu. Other IDs: `chroma1-hd`, `chroma1-flash`. |
| Prompt expansion | `make serve NO_LLM=1` | Disables the text LLM for this run. Otherwise it defaults on. |
| LoRAs | `make serve NO_LORAS=1` | Disables optional style adapters and their UI selector. |
| Generator port | `make serve PORT=5733` | Changes the webpage/API address to `localhost:5733`. |
| Queue port | `make serve QUEUE_PORT=55303` | Uses another standalone queue port. Use the same override with `make queue` and `make stop-queue`. |

Options can be combined, for example `make serve MODEL=z-image-turbo NO_LLM=1 NO_LORAS=1`. The examples below use the default ports.

To add another model later, stop the generator, run `make setup MODEL=chroma1-flash`, then `make serve MODEL=chroma1-flash`. Each model keeps its own waiting jobs; one model runs at a time.

## 4. Generate from the webpage

1. Open **[http://localhost:5732](http://localhost:5732/)**. Grey question-mark icons beside settings explain them on hover, keyboard focus or tap; press Escape or click elsewhere to dismiss the help.
2. Enter a prompt, such as **A watercolor fox in a snowy forest, soft morning light**.
3. For a first image, choose **Quick draft · 512 × 512**, keep **Number of images** at **1**, and leave the default steps alone.
4. Keep **Prompt expansion** switched on if you want added style, lighting and composition details. **Preview expansion** lets you review the description first. If you adopt it with **Use this description**, automatic expansion turns off to avoid expanding twice.
5. Click **Generate image**. **Generate immediately** is switched on by default: your request gets the next free slot ahead of ordinary queued work, after the active batch finishes.
6. View or download the result. With multiple options, select a thumbnail to choose the image you want.

For a photo variation, click **Choose photo** or expand **Or use a file path**. **Keep the original** defaults to 0.6; lower values allow larger changes. The output follows the photo's aspect ratio. This creates variations, with no guarantee of precise edits or exact identity preservation.

Open **[Gallery](http://localhost:5732/gallery)** to search saved images by prompt or tags, browse pages, edit tags and reuse settings. Images and JSON settings are saved automatically in `<storage folder>/<model>/images`. The webpage footer shows the default save folder; **Save location** lets you override it for a request.

### Settings you can change for each request

The API names below map to the webpage controls.

| Setting | API field | What to know |
| --- | --- | --- |
| Description | `prompt` | Required. Describe subject, medium/style, setting and light. Maximum 4000 characters; Z-Image processes 512 tokens including formatting, so put essential details first. |
| Resolution | `size` | Default `1024x1024`. API dimensions must be multiples of 64 from 256–2048. With a reference, the larger dimension sets the long edge and the photo sets the ratio. |
| Number of options | `n` | 1–5, default 1. Options render in turn with different seeds. More options take longer. |
| Prompt expansion | `expand_prompt` | Uses the text LLM when enabled at startup. Switched on by default in the UI; API default is false. It does not inspect reference photos. |
| Different compositions | `vary_prompt` | Requires `expand_prompt:true` and `n>=2`. Gives each option its own expanded composition. The UI enables it for eligible batches; API default is false. |
| Steps | `steps` | Defaults: Z-Image 9, Chroma HD 40, Flash 8. More steps cost time and do not always improve results. Keep the default to start. |
| Seed | `seed` | Leave blank/omit for random. Reuse a returned seed to compare setting changes; a seed alone does not guarantee the same image across models/settings. |
| Reference photo | `reference_image` | One absolute path on the server's Mac. The UI also uploads photos for you. Upload limits: 20 MiB and 40 megapixels. |
| Keep the original | `image_strength` | With a reference: 0 inclusive to 1 exclusive, default 0.6. Higher preserves more; references require at least 2 steps. |
| Style adapter | `lora`, `lora_strength` | Optional compatible installed LoRA; strength 0–2. Choose an ID from the API catalog. Configured trigger words are added automatically. |
| Tags | `tags` | Optional labels for gallery search. UI: comma-separated text. API: an array such as `["Forest","Watercolor"]`. Editable later. |
| Save location | `output_dir` | Absolute folder or `~/...` on the server's Mac. Omit to use the model's images folder. |
| Priority | `generate_immediately` | True runs next without interrupting active work. UI default true; async API default false to join the end of the queue. |
| API result format | `response_format` | Use `"url"` for easy previews/downloads. Default is `"b64_json"`. Both save the PNG and metadata on disk. |

For an optional Realism LoRA on Z-Image, run `make download-realism MODEL=z-image-turbo` before starting the generator. Normal setup downloads no LoRAs. Different seeds or composition prompts can help variety, but strongly constrained scenes and preserved references may still look similar.

In the **Queue** panel, drag a waiting job by its left-hand grip to change its place. You can also focus the grip and press ↑ / ↓. **Run Now** moves a waiting job to the next free slot, even while the queue is paused; the active batch finishes first. Order is saved across restarts.

## 5. Generate through the API

Keep the generator running. You can try requests directly in **[the interactive API docs](http://localhost:5732/docs)**: expand an endpoint, click **Try it out**, edit its JSON and click **Execute**. Or use the terminal steps below.

### Submit a request

Open another terminal and run:

```sh
curl --fail-with-body http://localhost:5732/v1/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: quickstart-fox-1' \
  -d '{
    "prompt": "A watercolor fox in a snowy forest, soft morning light",
    "size": "512x512",
    "n": 1,
    "expand_prompt": true,
    "generate_immediately": true,
    "tags": ["Forest", "Watercolor"],
    "response_format": "url"
  }'
```

The response returns a job `id` with status `queued` or `running`. **HTTP 202 means saved, not finished.** Leaving out `model` uses the generator's startup-selected model; specifying a different model is rejected. If you started with `NO_LLM=1`, set `expand_prompt` to false.

The idempotency key prevents duplicate work if you retry this same request after losing its response. Use a **new key for each intentional new request**; reusing a key with different settings returns 409.

### Check progress and get the picture

Copy the returned ID into the variable below:

```sh
IMAGEGEN_JOB_ID="paste-the-returned-id-here"
curl --fail-with-body "http://localhost:5732/v1/jobs/$IMAGEGEN_JOB_ID"
```

Run the second command again occasionally until the status is `succeeded`, `failed` or `cancelled`. On success, **`result.data[]`** contains every generated image, with its `url`, full `file_path`, `metadata_path` and seed. Open an image's `url` in your browser, or use the file already saved on disk. On failure, read the job's `error`. Closing your browser or timing out a request does not cancel a saved job.

### Move a waiting job or run it next

Read `queue_position` from a job/list response. It starts at 1 across all waiting jobs, independent of pagination. Send the desired position to move it up/down:

```sh
curl --fail-with-body --request POST "http://localhost:5732/v1/jobs/$IMAGEGEN_JOB_ID/move" \
  -H 'Content-Type: application/json' -d '{"position":1}'
curl --fail-with-body --request POST "http://localhost:5732/v1/jobs/$IMAGEGEN_JOB_ID/run-now"
```

Run Now marks that same job immediate and moves it first; it doesn't interrupt active work. These controls also exist on port 55302 with optional `?model=<id>`, even when the generator is off. Running/finished jobs return 409. If another client changes the queue, refresh positions before trying again.

### Discover models, adapters and saved images

```sh
curl --fail-with-body http://localhost:5732/v1/models
curl --fail-with-body http://localhost:5732/v1/loras
curl --fail-with-body 'http://localhost:5732/v1/images?q=forest&limit=24&offset=0'
```

Use a LoRA entry with `available:true`, setting `lora` to its `id`. Adapter discovery reports disabled/empty when the generator starts with `NO_LORAS=1`. For API reference photos, pass `reference_image` as a server-local absolute path; see [uploading references](docs/api.md#reference-photos-and-variations) if you prefer sending image bytes.

## 6. Optional: queue images while the generator is off

After setup, run **only** the lightweight queue service:

```sh
make queue
```

It runs in the background and returns control to your terminal. It doesn't load image models or the LLM and has no webpage. Its **[API docs](http://localhost:55302/docs)** remain available while the generator is off.

Submit the image settings inside a `request` object:

```sh
curl --fail-with-body http://localhost:55302/v1/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: quickstart-queued-fox-1' \
  -d '{
    "model": "z-image-turbo",
    "request": {
      "prompt": "A watercolor fox in a snowy forest",
      "size": "512x512",
      "response_format": "url"
    }
  }'
```

On this standalone API, omitted `model` **always defaults to Z-Image Turbo**, regardless of which generator you last ran. Ordinary work joins the end of the queue; you can reorder it later. The queue accepts settings for later validation; a saved job may still fail when the generator processes it.

List or inspect queued work with:

```sh
curl --fail-with-body 'http://localhost:55302/v1/jobs?model=z-image-turbo&status=pending&limit=24&offset=0'
IMAGEGEN_JOB_ID="paste-the-queue-job-id-here"
curl --fail-with-body "http://localhost:55302/v1/jobs/$IMAGEGEN_JOB_ID?model=z-image-turbo"
```

When you're ready to generate, run `make serve MODEL=z-image-turbo`. It picks up that model's saved jobs. The queue API returns raw saved metadata and local paths, with batch files under `result.images[]`; it doesn't serve image download URLs. Use the generator's job endpoint for `result.data[]` with URLs, or browse the gallery.

Queue editing, cancelling, clearing, retrying and pausing are described in the [API reference](docs/api.md#independent-queue-api-generator-may-be-off). The UI continues working through the generator API if the standalone queue service stops; `make queue` restores access for outside tools.

## 7. Stop and return later

1. Press **Ctrl-C** in the generator terminal. Its active batch finishes before shutdown; waiting work stays saved.
2. The queue API stays up. To stop it too, run `make stop-queue` from the project folder.
3. Next time, run `make serve` again. No setup is needed unless you're adding a model or repairing an incomplete download.

Pictures and settings live in your chosen storage folder; queue databases live in project `state/`. Back up both to retain images, tags and pending work. Downloaded models and generated media are excluded from Git.

For troubleshooting and advanced configuration, see the [README](README.md#if-something-goes-wrong), [setup reference](docs/setup.md) and [complete API reference](docs/api.md).
