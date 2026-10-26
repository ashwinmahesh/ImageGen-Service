# ImageGen-Service

Generate images on your Apple Silicon Mac, using a simple webpage or a local API. Write a description, optionally add a reference photo, and browse your results in a searchable gallery.

The app supports **Z-Image Turbo**, **Chroma1-HD**, and **Chroma1-Flash**. They share the same interface; you choose which model to use when starting the server. A small local language model can expand your descriptions and suggest different compositions.

![Image generator with prompt and reference-photo controls](docs/screenshots/generator.png)

## Quick start

For a step-by-step walkthrough of setup, the webpage, API requests and configurable settings, follow **[QUICKSTART.md](QUICKSTART.md)**.

You need an **Apple Silicon Mac**, Git, Make, and **Python 3.14**. If you use Homebrew, install Python with `brew install python@3.14`. Z-Image uses Apple's Metal GPU through MLX. The bundled Chroma runtime targets macOS 26.6.2 on Apple Silicon; other macOS versions may need a compatible build. Windows and NVIDIA/CUDA are not supported by this setup. If Git or Make is missing, install Apple's command line tools with `xcode-select --install`.

Open Terminal and run:

```sh
git clone https://github.com/ashwinmahesh/ImageGen-Service.git
cd ImageGen-Service
make setup
make serve
```

Already downloaded the project? Open Terminal in that folder and start with `make setup`.

During setup, you'll make two choices:

1. **Image model:** press Enter for Z-Image Turbo, or select another model from the list.
2. **Storage folder:** press Enter to keep the downloads inside the project, or enter a folder on an external drive.

Setup downloads the selected image model and a shared prompt-expansion model, and installs the required software. The first download can take a while. You only need to do this once for each image model you want to use. Setup doesn't start the server.

When `make serve` asks about prompt expansion, press Enter to keep it enabled. Once startup finishes, open **[http://localhost:5732](http://localhost:5732/)** in your browser. Keep the terminal open while generating.

If setup can't find Python, use its full path:

```sh
make setup BOOTSTRAP_PYTHON=/opt/homebrew/bin/python3.14
```

## Generating images

Enter a description and click **Generate**. Start at 1024 × 1024 with the model's default settings; larger images and more steps take longer and use more memory. The resolution controls support dimensions up to 2048 × 2048.

**Number of images** lets you request 1–5 options for the same description. Each uses a different seed and renders in turn. With prompt expansion enabled, **Different compositions with LLM** gives each option its own expanded description. This helps create variety, although tightly specified scenes or strongly preserved reference photos can still produce similar results.

![Multiple image options with selectable previews](docs/screenshots/image-options.png)

**Expand description with local LLM** adds visual details about your requested style, lighting and composition. Use **Preview expansion** to review it first, or turn it off for a particular image. Your original description is saved alongside the expanded version. The expander reads text only; it doesn't inspect reference photos.

For a variation on an existing picture, upload a reference or enter its local file path. The result automatically follows the reference's aspect ratio. **Keep the original** controls how closely it follows the source: lower values allow bigger changes. This is a way to make variations; precise edits, inpainting and exact character identity are not guaranteed.

Images and their generation settings are saved automatically in `<storage folder>/<model>/images`. With the default storage choice, that means a folder such as `z-image-turbo/images` inside the project. You can choose another output folder for a request. The API also returns the full image and metadata paths.

## Finding your pictures

Open **[the gallery](http://localhost:5732/gallery)** to browse results from all installed models. Search by description or tags, sort the results, and move through pages. Open an image to download it, see its settings, edit its tags or reuse it as a starting point.

![Searchable image gallery with editable tags](docs/screenshots/gallery.png)

The generator, queue and gallery screenshots use demonstration data and a simulated backend. Their timings are not benchmarks. The multiple-options screenshot shows a real two-image Z-Image test.

## Starting, stopping and choosing models

For everyday use, run:

```sh
make serve
```

The menu offers numbered choices for models you've already set up. Other supported models appear below with a command explaining how to install them. Pressing Enter chooses Z-Image Turbo when available, otherwise the first installed model.

To add and use another model:

```sh
make setup MODEL=chroma1-flash
make serve MODEL=chroma1-flash
```

The available IDs are `z-image-turbo`, `chroma1-hd` and `chroma1-flash`. One model runs at a time. Stop the generator before switching; each model's waiting jobs are kept until you start it again.

Press **Ctrl-C** in the serving terminal to stop the generator after its active batch finishes. The separate queue service stays running, so agents can still submit work. To stop that too:

```sh
make stop-queue
```

Useful startup options:

| Command | What it changes |
| --- | --- |
| `make serve NO_LLM=1` | Turns off prompt expansion for this run. |
| `make serve NO_LORAS=1` | Uses the base model with adapters disabled. |
| `make serve PORT=5733` | Uses another browser/API port. |

LoRAs are optional style adapters. Normal setup downloads none. An optional Realism adapter for Z-Image is available with `make download-realism MODEL=z-image-turbo`; installed compatible adapters then appear in the webpage. See [adapter setup](docs/setup.md#optional-adapters) for details.

## Queue work for later

You can leave the lightweight queue service running without keeping an image model loaded:

```sh
make queue
```

Other tools can then submit requests to **[localhost:55302](http://localhost:55302/docs)**, even while the generator is off. When you're ready to use your Mac's compute, run `make serve`. It starts the queue service too if needed, then processes waiting work for the selected model. The webpage is available only while the generator is running.

![Saved generation queue with pause and cancellation controls](docs/screenshots/queue.png)

Ordinary queued work runs in order. The webpage enables **Generate immediately** by default, which puts your request in the next available slot ahead of ordinary work. It still waits for the active image or batch to finish. You can pause ordinary work, cancel waiting entries, clear the waiting list, or retry a failed job. Cancelling or clearing the queue keeps completed images and job history.

Jobs are saved in SQLite inside the project's `state/` folder and survive restarts. The webpage accesses that database through the generator API, so it keeps working if the standalone queue service stops. Run `make queue` again to restore access for outside tools.

## Using the API

The generator's **[interactive API docs](http://localhost:5732/docs)** explain the available settings and responses. Here's a request that saves a job and returns its ID:

```sh
curl --fail-with-body http://localhost:5732/v1/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: fox-example-1' \
  -d '{"prompt":"A watercolor fox in a snowy forest","response_format":"url"}'
```

Read `GET /v1/jobs/<id>` until its status is `succeeded`, `failed` or `cancelled`. A successful response includes each image's URL, full `file_path` and `metadata_path` in `result.data[]`. Acceptance alone doesn't mean the image has finished.

The standalone queue service uses a slightly different format so it can route work to a model that's currently offline:

```sh
curl --fail-with-body http://localhost:55302/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"model":"z-image-turbo","request":{"prompt":"A watercolor fox in a snowy forest","response_format":"url"}}'
```

On the generator API, leaving out `model` uses the currently running model. On the standalone queue API, it always defaults to Z-Image Turbo. Both provide model and LoRA discovery endpoints. See the [full API and queue reference](docs/api.md) for settings, results, editing queued requests, pagination, retries and examples.

## Storage and backups

An external SSD is optional. If you choose one during setup, the project creates links to its model folders and remembers the location. Future starts reuse it. Keep the drive connected while generating; a missing drive produces an error rather than downloading another copy elsewhere.

Weights, generated pictures, references and gallery indexes live in the chosen storage folder. The source code, Python environment and **queue databases stay in the project**. Back up both the data folder and the project's `state/` folder to preserve your pictures, tags and queued work. Local data and downloaded models are excluded from Git.

For scripted setup, custom paths and model download sources, see the [setup reference](docs/setup.md).

## If something goes wrong

- **Interrupted download:** rerun `make setup MODEL=<model-id>`. Existing downloads are reused or resumed.
- **External drive missing:** reconnect it before starting the generator.
- **Missing prompt-expansion model:** run `make download-llm`, or start with `make serve NO_LLM=1`.
- **Port already in use:** choose another port with `make serve PORT=5733`, and use that port in the browser.
- **Slow generation or memory pressure:** try a smaller image, fewer options, and close other apps using the GPU. Chroma can be considerably slower on this Mac.

Run `make health` to check a running generator. `make test` checks the code using simulated downloads and generation. `make smoke` makes a real 512-pixel test image and requires the generator to be running.

For development, read [AGENTS.md](AGENTS.md). An AI tool operating the service can use [the service skill](skill/image-model-server/SKILL.md). Details of the prompt expander's design are in [the prompt-expansion guide](docs/prompt-expansion.md).
