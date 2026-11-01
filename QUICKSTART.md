# Quick Start

This guide takes you from a fresh download to your first image. Use the webpage for everyday work, or the API to generate images from another tool. For detailed controls, see the [User Manual](MANUAL.md).

## 1. Install and set up

You need an **Apple Silicon Mac**, Git and Make. Setup handles Python for you. If Git or Make is missing, run `xcode-select --install`, finish the installation, then continue.

Open **Terminal** and run these commands in order:

```sh
git clone https://github.com/ashwinmahesh/ImageGen-Service.git
cd ImageGen-Service
make setup
```

Answer the setup questions:

1. **Model:** press Enter for **Z-Image Turbo**, the recommended starting point.
2. **Storage:** press Enter to keep models and images inside the project, or enter an external folder such as `/Volumes/MySSD/models`. Keep that drive connected when using the tool.
3. Wait for setup to finish downloading the image model and the small prompt-expansion model. The first download can take a while.

Setup is safe to rerun. Already installed models show `[SETUP COMPLETE]`; unchanged weights are reused. To cancel, choose `q` at the model menu or press **Ctrl+C**; rerun setup later to continue.

## 2. Start the app

In the same project folder, run:

```sh
make serve
```

Choose your installed default model. At **Enable local LLM prompt expansion? [Y/n]**, press Enter: this lets a small local text model add visual detail to your descriptions. You can switch expansion off for individual images.

Keep this terminal open, wait for startup to finish, then open **[http://localhost:5742](http://localhost:5742/)**.

## 3. Generate your first image

1. In **Generate**, enter: **A watercolor fox in a snowy forest, soft morning light**.
2. Choose **Quick draft · 512 × 512**, keep **Number of images** at **1**, and leave the other settings at their defaults.
3. Click **Generate image**. A percentage and time left appear under the preview and in the header. Switch on **Live preview** before generating to watch the image form, or choose **Stop generating** to cancel. When it finishes, view or download the result; it's also saved automatically.

Grey question-mark icons explain the controls. These are the main choices to try next:

| Control | What it does |
| --- | --- |
| **Model** | Chooses an installed model; switching happens automatically between jobs. |
| **Resolution** | Larger images take longer and use more memory. |
| **Number of images** | Makes 1–5 options with different seeds. |
| **Prompt expansion** | Adds descriptive detail. Use **Preview expansion** to review it first. |
| **Different compositions with LLM** | Explores different compositions for a multi-image batch with expansion enabled. |
| **Live preview** | Shows the image as it renders. Off by default because each preview adds decoding time. |
| **Generate immediately** | Runs next after the active batch; switch off to join the ordinary queue. |

Open **Queue** in the header to see each job's progress, drag waiting jobs into order or choose **Run Now**; an active batch always finishes first. **Monitor** opens a job in a new tab with its prompt and settings filled in. Switch off **Process queue** in the header to pause waiting work. The gear icon opens **Settings**; see the [manual](MANUAL.md#settings) before turning on **Keep image weights loaded between generations**.

## 4. Find images and edit photos

Open **Gallery** to search by prompt or tags. Click an image or its caption to open it, then download it, reuse its settings or expand it to fill the window. **Delete image** asks for confirmation. The generator footer shows the default save folder; use **Save location** to choose another folder for a request.

To edit a photo, stop the app with Ctrl+C, add an editing model with `make setup MODEL=flux2-klein-4b`, then run `make serve` again. Choose **Edit photo**, select the editor, upload a photo and describe the change. Leave resolution on **Auto** to follow the source size. You can also choose **Edit photo** from a gallery image. See [photo editing](MANUAL.md#edit-an-existing-photo) for Qwen selected-area edits and preservation controls.

## 5. Use the API

Keep the app running. The easiest way to try requests is **[localhost:5742/docs](http://localhost:5742/docs)**: open an endpoint, choose **Try it out**, then **Execute**.

Or open another terminal and submit:

```sh
curl --fail-with-body http://localhost:5742/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A watercolor fox in a snowy forest","size":"512x512","generate_immediately":true,"response_format":"url"}'
```

The response includes a job `id`. **202 means accepted, not finished.** Omitted `model` uses your saved default. Replace the ID below, then repeat this command occasionally to check progress:

```sh
IMAGEGEN_JOB_ID="paste-the-returned-id-here"
curl --fail-with-body "http://localhost:5742/v1/jobs/$IMAGEGEN_JOB_ID"
```

While it runs, `progress.percent` shows how far it has got. When `status` is `succeeded`, open a `result.data[]` image's `url`, or find its saved `file_path`. On failure, read `error`. To cancel a waiting or running job:

```sh
curl --fail-with-body -X DELETE "http://localhost:5742/v1/jobs/$IMAGEGEN_JOB_ID"
```

See the [API reference](docs/api.md) for all settings and endpoints.

## 6. Stop and come back later

Press **Ctrl+C** in the app terminal. It finishes the active batch before stopping; waiting jobs and saved images remain. Next time, open Terminal in the project folder and run `make serve`.

The lightweight queue API stays running separately. Use `make stop-queue` to stop it, or `make queue` to accept jobs while the generator is off. The [manual](MANUAL.md#queue-images-while-the-generator-is-off) shows how other tools submit to that API.

For more options, follow the [User Manual](MANUAL.md). If startup or downloads fail, see [troubleshooting](MANUAL.md#if-something-goes-wrong).
