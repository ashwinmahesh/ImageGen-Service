# ImageGen-Service

Generate and edit images on your Apple Silicon Mac, using a webpage or API. Your models and images stay local, with a gallery to find your favourites and a queue for work you want to run later.

![ImageGen webpage](docs/screenshots/generator.png)

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
- **Develop your prompt:** an optional local language model adds detail about lighting, style and composition. Preview its suggestions or turn it off.
- **Edit photos:** use Klein or Qwen to change an existing image. Qwen also supports selecting an area to edit.
- **Try styles:** add compatible LoRA adapters with [`make setup-lora`](MANUAL.md#add-your-own-lora).
- **Keep your favourites:** search the gallery by prompt or tags, download images, reuse settings or delete unwanted results with confirmation.
- **Queue work:** save requests for later, drag jobs into order or choose **Run Now**. Waiting work survives restarts.
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

![Compare image options](docs/screenshots/image-options.png)

![Search the gallery](docs/screenshots/gallery.png)

![Manage the queue](docs/screenshots/queue.png)

Generator, gallery and queue screenshots use simulated demo data. The image-options screenshot is from a real Z-Image test; timings are not benchmarks.

</details>
