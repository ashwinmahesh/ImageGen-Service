# ImageGen-Service

Generate and edit images on your Apple Silicon Mac, using a webpage or API. Your models and images stay local, with a gallery to find your favourites and a queue for work you want to run later.

![ImageGen generator following a running job at 63%, with its live preview](docs/screenshots/generator.png)

## Get started

You need an **Apple Silicon Mac**, Git and Make. Setup handles Python and model downloads.

```sh
git clone https://github.com/ashwinmahesh/ImageGen-Service.git
cd ImageGen-Service
make setup
make serve
```

During setup, press Enter to choose **Z-Image Turbo**, or select another model. You can store model downloads and images in the project or on an external drive.

Open **[localhost:5742](http://localhost:5742/)** to generate your first image. The [Quick Start](QUICKSTART.md) walks through setup, the webpage and API access.

## What you can do

- **Generate options:** make 1–5 images from a description and compare the results.
- **Watch it render:** a percentage and time left show under the preview, in the header and in the queue. Switch on **Live preview** to watch the image form, or choose **Stop generating** to cancel it.
- **Develop your prompt:** an optional local language model adds detail about lighting, style and composition. Preview its suggestions or turn it off.
- **Edit photos:** use Klein or Qwen to change an existing image. With Qwen, paint over the area to change; the model is also told where your selection is.
- **Try styles:** add compatible LoRA adapters with [`make setup-lora`](MANUAL.md#add-your-own-lora).
- **Keep your favourites:** search the gallery by prompt or tags, open an image to view it fitted or full-window, download it, reuse settings or delete it with confirmation.
- **Queue work:** save requests for later, drag jobs into order or choose **Run Now**. **Monitor** opens a job in a new tab with its settings filled in. Switch off **Process queue** to pause; waiting work survives restarts.
- **Save reloads:** the gear icon opens **Settings**. **Keep image weights loaded between generations** keeps FLUX.2 Klein or Qwen-Image in memory for back-to-back images of the same model, then releases it.
- **See resource use:** view CPU, memory, GPU, job logs and model load/unload history in Observability.

Available models: **Z-Image Turbo**, **Chroma1-HD**, **Chroma1-Flash**, **FLUX.2 Klein 4B/9B** and **Qwen-Image 2.1 Turbo**. Set up the ones you want; the generator switches between them as jobs run. See [model details and licences](docs/setup.md#models-and-effective-settings) before commercial use.

## Use it from other tools

The running generator provides [interactive API docs](http://localhost:5742/docs).

To accept jobs while the generator is off, run `make queue` and use the [queue API](http://localhost:55302/docs). Run `make serve` when you are ready to generate; it also starts the queue API if needed.

Ctrl+C stops the generator after its active batch finishes. The queue stays running until you use `make stop-queue`.

## Learn more

- [Quick Start](QUICKSTART.md): setup, your first image and basic UI/API use.
- [User Manual](MANUAL.md): all controls, photo editing, LoRAs, queues and monitoring.
- [API reference](docs/api.md): image requests, queue controls, gallery and metrics.
- [Setup reference](docs/setup.md): storage, models and troubleshooting.
- [Prompt expansion](docs/prompt-expansion.md): how descriptions and composition options are developed.

For AI tools: [operating skill](skill/image-model-server/SKILL.md). For contributors: [AGENTS.md](AGENTS.md).

<details>
<summary>More screenshots</summary>

![Compare three image options](docs/screenshots/image-options.png)

![Search the gallery](docs/screenshots/gallery.png)

![Queue window with a running job's progress and waiting jobs](docs/screenshots/queue.png)

All screenshots use simulated demo data and fake rendering; percentages and timings are not benchmarks.

</details>
