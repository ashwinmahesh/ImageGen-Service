# ImageGen-Service

**A local image generation studio and API for Apple Silicon Macs.** Turn a description or reference photo into images, explore several options, and keep the results in a searchable gallery. After the initial downloads, image generation and prompt expansion run on your Mac without a cloud API.

The project brings **Z-Image Turbo**, **Chroma1-HD** and **Chroma1-Flash** into one app. They share a webpage, API and saved-image library; choose the model when you start the generator. A small local language model helps develop descriptions and explore compositions, while a persistent queue lets people and agents submit work whenever they need it.

Use the webpage for hands-on experimentation, or call the API from scripts and other tools. You can leave the lightweight queue running all day and start the image generator only when you want to use your Mac's compute.

![Image generator with prompt and reference-photo controls](docs/screenshots/generator.png)

For installation and a guided walkthrough of both interfaces, follow **[QUICKSTART.md](QUICKSTART.md)**.

## Techniques for better images

The service combines prompt development, creative alternatives and reference-based iteration. Each control is optional, so you can compare results and decide what helps your particular image.

### Develop the description while preserving the original

A local **Qwen3.5-2B model with 4-bit weights** can expand a short description with concrete details about light, palette, atmosphere, materials and composition. Your original text is kept verbatim, with added detail after it. The LLM is instructed to preserve your subjects, counts, actions, colors, viewpoint, quoted text and exclusions.

The expansion is guided by the requested medium and aesthetic. A watercolor prompt can gain translucent washes and soft edge variation; a cel-animation prompt can gain clean ink contours, flat color shapes and limited shadow tones. The goal is to develop the look you asked for through visible rendering choices.

Guidance also follows the selected image model. For Z-Image, expansions are checked against its actual tokenizer budget before rendering. You can preview the expanded description, edit and adopt it, or turn expansion off. Original and expanded prompts are saved with the image so you can see what was used. The expander reads text only; it does not inspect reference photos.

### Explore multiple options

Request **1–5 images** from the same description and compare them together. Each option uses a different seed. With **Different compositions with LLM** enabled, each also receives its own expanded prompt, exploring open choices such as a wide environmental view, closer framing, diagonal flow or layered depth.

The LLM is instructed to keep framing and other details you've explicitly specified, and explore choices you've left open. This gives you another way to seek variety when seed changes produce similar layouts, although tightly constrained prompts or strongly preserved references can still yield similar images.

![Multiple generated options with selectable previews](docs/screenshots/image-options.png)

### Guide the result with a reference or style adapter

A reference photo provides a visual starting point. **Keep the original** controls how closely the result follows it: lower values allow larger changes. The service preserves the reference's aspect ratio automatically, which helps avoid stretching the composition. You can use a generated image as the next reference to keep developing an idea. This is image-to-image variation; precise localized edits and exact identity retention are not guaranteed.

Optional **LoRAs** are small adapters that influence the model's learned appearance or style. Choose a compatible installed adapter and adjust its strength; configured trigger words are inserted automatically. Normal setup downloads no adapters. **Realism for Z-Image Turbo** is the only bundled optional preset; see [adapter setup](docs/setup.md#optional-adapters).

### Make iteration easier to compare and repeat

Every image is saved with its prompt, expansion when used, seed, model, dimensions, steps and reference/adapter settings. Reuse a seed to compare setting changes, reuse a description, or make a variation from the result you like best. Saved settings give you a record of the choices behind each result.

The effect of prompt expansion and composition planning depends on the scene and model. Preview and compare results to decide which choices work best. The [prompt-expansion design and evidence](docs/prompt-expansion.md) explains the research, implementation choices and limits of the local evaluation.

## Keep a searchable image library

The gallery brings together results from all supported models, including images saved to custom output folders. Search by prompt or tags, sort and browse pages, and open an image to see its settings, download it, edit its tags or use it as a starting point for another generation.

![Searchable image gallery with editable tags](docs/screenshots/gallery.png)

PNG images and JSON generation records are saved automatically in `<storage folder>/<model>/images`, unless you choose another output location. Tag edits are stored separately so the original image and generation record remain intact.

## Queue work around your schedule

The app has two services that share a persistent SQLite queue:

| Service | What it does |
| --- | --- |
| **Generator and webpage**, [localhost:5732](http://localhost:5732/) | Runs the selected image model and optional prompt expander, serves the UI/gallery, and processes that model's queue. |
| **Lightweight queue API**, [localhost:55302/docs](http://localhost:55302/docs) | Accepts and manages jobs while the generator is off, without loading image models or the LLM. |

Ordinary jobs run in order. The webpage defaults to **Generate immediately**, which puts your request in the next free slot ahead of ordinary queued work. Active batches finish first. You can pause ordinary work, cancel waiting jobs, clear the waiting list and retry failed jobs.

![Saved generation queue with pause and cancellation controls](docs/screenshots/queue.png)

Jobs survive restarts. The generator accesses SQLite directly, so the webpage and generation continue if the standalone queue API stops. Start the generator with another model when you want to process that model's waiting jobs.

The **Queue** panel lets you drag waiting jobs into a different order or use **Run Now** to put one next, after the active batch finishes. Run Now also works while ordinary work is paused. Saved order survives restarts, and both APIs expose the same controls.

## Built for local hardware and storage

Quantized model weights and Apple's Metal GPU help make local inference practical. Image options render sequentially, and the prompt-expansion process releases its memory before image rendering. Resolution, image count and model choice affect memory use and generation time; Chroma can be substantially slower on this Mac.

Model downloads and generated media can live inside the project or on an external SSD. Setup remembers the location and creates links when needed. The source code, Python environment and SQLite queue databases stay in the project. Back up both your model/media store and the project's `state/` folder to preserve images, tags and waiting work. Downloaded weights and personal media are excluded from Git.

The generator, gallery and queue screenshots use demonstration data and a simulated backend; their timings are not benchmarks. The multiple-options screenshot shows a real two-image Z-Image test.

## Quick start

Requires an **Apple Silicon Mac**, Git, Make and **Python 3.14**. See the [quick-start guide](QUICKSTART.md#1-check-the-prerequisites) for prerequisites and Chroma runtime compatibility.

```sh
git clone https://github.com/ashwinmahesh/ImageGen-Service.git
cd ImageGen-Service
make setup
make serve
```

During first setup, Enter chooses Z-Image Turbo and project-local storage; you can select another model or an external folder. Serving offers installed models and enables prompt expansion by default. Open **[http://localhost:5732](http://localhost:5732/)** once startup finishes.

`make serve` also starts the background queue API if needed. Ctrl-C stops the generator after its active batch finishes; `make stop-queue` stops the queue API separately. Use `make serve NO_LLM=1` to disable prompt expansion or `make serve NO_LORAS=1` to disable adapters.

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: step-by-step setup, UI and API walkthroughs, settings and shutdown.
- **[Complete API reference](docs/api.md)**: every endpoint on both services, request/response fields, examples, errors, queue management and media downloads.
- **[Setup reference](docs/setup.md)**: model sources, storage choices, startup options and runtime details.
- **[Prompt-expansion design](docs/prompt-expansion.md)**: techniques, research and evaluation limits.

While running, the generator publishes [interactive API docs](http://localhost:5732/docs). The standalone queue API has [its own docs](http://localhost:55302/docs) and a request envelope for routing work to offline models. The [quick-start API walkthrough](QUICKSTART.md#5-generate-through-the-api) explains the difference.

For code changes, read [AGENTS.md](AGENTS.md). An AI tool operating the service can use [the service skill](skill/image-model-server/SKILL.md).

## If something goes wrong

- **Interrupted download:** rerun `make setup MODEL=<model-id>` to reuse or resume it.
- **External drive missing:** reconnect it; the generator won't silently download to another location.
- **Missing prompt LLM:** run `make download-llm`, or start with `make serve NO_LLM=1`.
- **Port conflict:** use `make serve PORT=5733` and open that port in the browser.
- **Slow generation or memory pressure:** try smaller images or fewer options and close other GPU-heavy apps.

`make health` checks a running generator. `make test` uses simulated downloads and generation; `make smoke` generates a real test image on a running server. Further details are in the [setup reference](docs/setup.md#verification-and-troubleshooting).
