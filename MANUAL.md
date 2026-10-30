# User Manual

This guide explains how to use ImageGen-Service and its controls. For a first run, follow [Quick Start](QUICKSTART.md); for exact request fields and responses, use the [API reference](docs/api.md).

## Contents

- [Choose a model](#choose-a-model)
- [Install and set up](#install-and-set-up)
- [Start the generator](#start-the-generator)
- [Techniques for better images](#techniques-for-better-images)
- [Generate from the webpage](#generate-from-the-webpage)
- [Settings for each request](#settings-for-each-request)
- [Edit an existing photo](#edit-an-existing-photo)
- [Add your own LoRA](#add-your-own-lora)
- [Search and manage saved images](#search-and-manage-saved-images)
- [Manage the queue](#manage-the-queue)
- [Monitor resource use](#monitor-resource-use)
- [Generate through the API](#generate-through-the-api)
- [Queue images while the generator is off](#queue-images-while-the-generator-is-off)
- [Local hardware, storage and backups](#local-hardware-storage-and-backups)
- [Stop and return later](#stop-and-return-later)
- [If something goes wrong](#if-something-goes-wrong)

## Choose a model

| Model ID | Checkpoint | Default steps | Reference workflow |
| --- | --- | ---: | --- |
| `z-image-turbo` | MLX 4-bit | 9 | Photo variations |
| `chroma1-hd` | GGUF Q4_K_M | 40 | Photo variations |
| `chroma1-flash` | GGUF Q4_K_M | 8 | Photo variations |
| `flux2-klein-4b` | MLX 8-bit | 4 | Instruction editing |
| `flux2-klein-9b` | MLX 4-bit | 4 | Instruction editing |
| `qwen-image-2.1-turbo` | MLX 4-bit, including vision encoder | 8 | Whole-photo/masked editing and RGBA output |

Add a model with `make setup MODEL=<model-id>`, then choose it in the webpage or an API request. `make serve MODEL=<model-id>` chooses the default for omitted request models. The interactive setup list marks models whose local assets are ready with `[SETUP COMPLETE]`. Setup downloads weights; serving uses local assets only. Start Qwen Turbo around 1024px for detail; peak memory depends on the reference and output size. Performance depends on your Mac, resolution and reference photo.

Klein 4B uses Apache 2.0. Commercial model use of [Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) and [Qwen Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE) requires the respective owner's separate license. The service shows their model-specific notices when each backend loads and in the webpage/API.

## Install and set up

### Check the prerequisites

You need an **Apple Silicon Mac**, Git and Make. Setup reuses a compatible Python environment or installed interpreter, and downloads Python into the project only when needed. No system Python installation or shell changes are required. The provided setup uses Apple's Metal GPU; it does not support Windows or NVIDIA/CUDA. The bundled Chroma runtime targets macOS 26.6.2 on Apple Silicon; other versions may need a compatible build.

If Git or Make is missing, install Apple's command line tools:

```sh
xcode-select --install
```

### Download the project and set up a model

Open Terminal and run:

```sh
git clone https://github.com/ashwinmahesh/ImageGen-Service.git
cd ImageGen-Service
make setup
```

If you already have the project, open Terminal in that folder and run `make setup`.

Follow the setup questions:

1. **Choose a model.** Press Enter for **Z-Image Turbo**, or select Chroma1-HD, Chroma1-Flash, FLUX.2 Klein 4B/9B or Qwen-Image 2.1 Turbo. Models whose required local assets are ready show `[SETUP COMPLETE]` beside their name; you can still select them to rerun setup.
2. **Choose a storage folder.** Press Enter to keep model downloads and pictures inside the project. To use an external SSD, enter an absolute path such as `/Volumes/MySSD/models`.
3. **Wait for setup to finish.** It selects Python, installs pinned dependencies into `.venv`, and downloads the chosen image model plus a shared Qwen prompt-expansion model. Setup does not start either server.

To cancel, enter `q` at the model menu or press **Ctrl+C** at any point during setup. Existing files and cached partial downloads are kept; rerun `make setup` to continue. Selecting an already downloaded model is safe: unchanged weights need no redownload, and missing or changed pinned files are updated in place. Setup still checks dependencies and shared assets.

The external folder is a parent store: setup creates model folders inside it and links them into the project. Your choice is remembered, so later setup runs reuse it. Keep the drive connected when generating. Queue databases always stay inside the project's `state/` folder.

Setup supports standard Apple Silicon CPython **3.12, 3.13 and 3.14**, except 3.14.1. It checks your target environment first, then installed interpreters, and downloads 3.14.9 only if none are compatible. Downloaded Python and its helper/cache stay in ignored `.python` and `.bootstrap` folders inside the project; `DATA_DIR` still controls models and pictures.

An incompatible or broken environment is preserved with an explanation. Choose a fresh environment, for example `make setup VENV="$(pwd)/.venv-new"`, or move the old one aside yourself before retrying. Keep using the same VENV override when serving.

If you prefer choosing Python yourself, provide an absolute executable path (optional):

```sh
make setup BOOTSTRAP_PYTHON=/absolute/path/to/python3.12
```

Serving and model-only downloads use the environment already set up. They do not download Python or install packages; `BOOTSTRAP_PYTHON` is a setup-only override. See [Python selection](docs/setup.md#python-selection) for details.

## Start the generator

First complete `make setup` for at least one model. Serving exits with setup instructions if its environment is missing or no model passes readiness checks; it does not start the queue/generator, prompt for the LLM, or download anything. `MODEL`, `NO_LLM` and `NO_LORAS` do not bypass this requirement.

From the project folder, run:

```sh
make serve
```

1. Choose a **default model** from the installed models. The generator can process requests for every setup model; the default applies when a request omits its model. Klein names show `(edits enabled)`; Qwen shows `(edit selected area enabled)`. Enter reuses a ready saved default, then prefers Z-Image Turbo or the first ready model; models you haven't set up appear separately with installation commands.
2. At **Enable local LLM prompt expansion? [Y/n]**, press Enter to keep expansion available. A small AI text model on your Mac can add visual detail to your image descriptions before generation; you can turn this off for each image.
3. Wait for startup to finish. Keep this terminal open while using the generator.

You now have two local services:

| Address | Purpose |
| --- | --- |
| [localhost:5742](http://localhost:5742/) | Generator, webpage, gallery and image API. Processes all setup models, one at a time. |
| [localhost:55302/docs](http://localhost:55302/docs) | Lightweight queue API. Keeps accepting jobs when the generator is off. |

`make serve` starts the queue service automatically if needed. To check generator readiness from another terminal in the project folder, run `make health`; look for `"status":"ready"`. Model loading happens when a job starts; the browser/API stays available during loading and switching.

### Startup choices

These settings are chosen before the generator starts:

| Choice | Example | What it does |
| --- | --- | --- |
| Model | `make serve MODEL=z-image-turbo` | Sets the saved default and skips the model menu. Other IDs: `chroma1-hd`, `chroma1-flash`, `flux2-klein-4b`, `flux2-klein-9b`, `qwen-image-2.1-turbo`. |
| Prompt expansion | `make serve NO_LLM=1` | Disables the text LLM for this run. Otherwise it defaults on. |
| LoRAs | `make serve NO_LORAS=1` | Disables optional style adapters and their UI selector. |
| Generator port | `make serve PORT=5733` | Changes the webpage/API address to `localhost:5733`. |
| Queue port | `make serve QUEUE_PORT=55303` | Uses another standalone queue port. Use the same override with `make queue` and `make stop-queue`. |

Options can be combined, for example `make serve MODEL=z-image-turbo NO_LLM=1 NO_LORAS=1`. The examples below use the default ports.

To add an editing model later, run one of these setup commands, then select it in the webpage or request:

```sh
make setup MODEL=flux2-klein-4b
make setup MODEL=flux2-klein-9b
make setup MODEL=qwen-image-2.1-turbo
make serve MODEL=flux2-klein-9b
```

Each setup downloads only its selected pre-quantized checkpoint and the shared prompt LLM. All three support text-only generation and single-photo instruction editing. Klein defaults to 4 steps; Qwen Turbo to 8. All models share one waiting queue; one model runs at a time. Commercial model use of Klein 9B and Qwen Turbo requires a separate license; see the [model notes](docs/setup.md#models-and-effective-settings).

Choose a setup model for each request in the webpage. The generator switches automatically when that job reaches the front of the queue, after finishing the current batch. Only one image backend is active at a time; switching may add loading time. Consecutive Z-Image jobs reuse its resident model, while Chroma/Klein/Qwen retain their per-image low-memory runtime behavior.

## Techniques for better images

The service combines prompt development, creative alternatives and reference-based iteration. Each control is optional, so you can compare results and decide what helps your particular image.

### Develop the description while preserving the original

A local **Qwen3.5-2B model with 4-bit weights** can expand a short description with concrete details about light, palette, atmosphere, materials and composition. Your original text is kept verbatim, with added detail after it. The LLM is instructed to preserve your subjects, counts, actions, colors, viewpoint, quoted text and exclusions.

The expansion is guided by the requested medium and aesthetic. A watercolor prompt can gain translucent washes and soft edge variation; a cel-animation prompt can gain clean ink contours, flat color shapes and limited shadow tones. The goal is to develop the look you asked for through visible rendering choices.

Guidance also follows the selected image model. For Z-Image, expansions are checked against its actual tokenizer budget before rendering. You can preview the expanded description, edit and adopt it, or turn expansion off. Original and expanded prompts are saved with the image so you can see what was used. The expander reads text only; it does not inspect reference photos. An unusable structured response gets one automatic formatting retry. If that still fails, the request reports the reason before image rendering.

### Explore multiple options

Request **1–5 images** from the same description and compare them together. Each option uses a different seed. With **Different compositions with LLM** enabled, each also receives its own expanded prompt, exploring open choices such as a wide environmental view, closer framing, diagonal flow or layered depth.

The LLM is instructed to keep framing and other details you've explicitly specified, and explore choices you've left open. This gives you another way to seek variety when seed changes produce similar layouts, although tightly constrained prompts or strongly preserved references can still yield similar images.

![Multiple generated options with selectable previews](docs/screenshots/image-options.png)

### Guide the result with a reference or style adapter

A reference photo provides a visual starting point. **FLUX.2 Klein and Qwen-Image 2.1 Turbo** follow editing instructions such as “change the jacket to green.” With **Z-Image and Chroma**, the photo instead guides a variation, and **Keep the original** controls how much is preserved. The service keeps the photo's aspect ratio and lets you use generated images as the next reference. Use the dedicated **Edit photo** view to describe a change, specify what should stay the same, and optionally refine the instruction with a preservation-focused text LLM. With **Qwen**, paint or drag rectangles on the source preview to switch automatically to a selected-area edit; the service copies pixels outside the mask from the resized source. Klein supports whole-photo edits. Whole-photo editing does not guarantee exact identity or unchanged pixels.

Optional **LoRAs** are small adapters that influence the model's learned appearance or style. Choose a compatible installed adapter and adjust its strength; configured trigger words are inserted automatically. Normal setup downloads no adapters. **Realism for Z-Image Turbo** is the only bundled optional preset; see [adapter setup](docs/setup.md#optional-adapters). To download or import your own compatible LoRA, run `make setup-lora`; its weights and personal catalog stay outside tracked code, so you can keep updating with `git pull`. Follow [Add your own LoRA](#add-your-own-lora).

### Make iteration easier to compare and repeat

Every image is saved with its prompt, expansion when used, seed, model, dimensions, steps and reference/adapter settings. Reuse a seed to compare setting changes, reuse a description, or make a variation from the result you like best. Saved settings give you a record of the choices behind each result.

The effect of prompt expansion and composition planning depends on the scene and model. Preview and compare results to decide which choices work best. The [prompt-expansion design and evidence](docs/prompt-expansion.md) explains the research, implementation choices and limits of the local evaluation.

## Generate from the webpage

Generator, Gallery and Observability share the same header. Command-click a navigation link on Mac, Ctrl-click or middle-click to open it in another tab. The current page is highlighted in teal. Open **Queue** from any page; **Monitor**, **Open image** and **Run Now** open the relevant job in Generator when needed.

1. Open **[http://localhost:5742](http://localhost:5742/)**. Grey question-mark icons beside settings explain them on hover, keyboard focus or tap; press Escape or click elsewhere to dismiss the help.
2. Enter a prompt, such as **A watercolor fox in a snowy forest, soft morning light**.
3. For a first image, choose **Quick draft · 512 × 512**, keep **Number of images** at **1**, and leave the default steps alone.
4. Keep **Prompt expansion** switched on if you want added style, lighting and composition details. **Preview expansion** lets you review the description first. If you adopt it with **Use this description**, automatic expansion turns off to avoid expanding twice. Malformed LLM responses get one automatic formatting retry; if it still fails, preview again or turn Prompt expansion off to use your original description.
5. Click **Generate image**. **Generate immediately** is switched on by default: your request gets the next free slot ahead of ordinary queued work, after the active batch finishes.
6. View or download the result. With multiple options, select a thumbnail to choose the image you want.

For a photo variation, click **Choose photo** or expand **Or use a file path**. **Keep the original** defaults to 0.6; lower values allow larger changes. The output follows the photo's aspect ratio. This creates variations, with no guarantee of precise edits or exact identity preservation.


## Settings for each request

The API names below map to the webpage controls.

| Setting | API field | What to know |
| --- | --- | --- |
| Description | `prompt` | Required. Describe subject, medium/style, setting and light. Maximum 4000 characters; Z-Image processes 512 tokens including formatting, so put essential details first. |
| Resolution | `size` | Without a reference: `1024x1024` by default. With a reference: Auto detects its oriented dimensions without upscaling, capped at a 2048px long edge. API clients omit `size` or pass `"auto"`; an explicit `WIDTHxHEIGHT` overrides the long edge (multiples of 64 from 256–2048). The photo sets the ratio. |
| Number of options | `n` | 1–5, default 1. Options render in turn with different seeds. More options take longer. |
| Prompt expansion | `expand_prompt` | Uses the text LLM when enabled at startup. Switched on by default in the UI; API default is false. It does not inspect reference photos. |
| Different compositions | `vary_prompt` | Requires `expand_prompt:true` and `n>=2`. Gives each option its own expanded composition. The UI enables it for eligible batches; API default is false. |
| Steps | `steps` | Defaults: Z-Image 9, Chroma HD 40, Flash 8. More steps cost time and do not always improve results. Keep the default to start. |
| Seed | `seed` | Leave blank/omit for random. Reuse a returned seed to compare setting changes; a seed alone does not guarantee the same image across models/settings. |
| Reference photo | `reference_image` | One absolute path on the server's Mac. The UI also uploads photos for you. Upload limits: 20 MiB and 40 megapixels. |
| Keep the original | `image_strength` | Z-Image/Chroma variations only: [0,1), default 0.6; higher preserves more. Omit for Klein/Qwen instruction editing; the webpage hides it. All references require at least 2 steps. |
| Style adapter | `lora`, `lora_strength` | Optional compatible installed LoRA; strength 0–2. Choose an ID from the API catalog. Configured trigger words are added automatically. |
| Tags | `tags` | Optional labels for gallery search. UI: comma-separated text. API: an array such as `["Forest","Watercolor"]`. Editable later. |
| Save location | `output_dir` | Absolute folder or `~/...` on the server's Mac. Omit to use the model's images folder. |
| Priority | `generate_immediately` | True runs next without interrupting active work. UI default true; async API default false to join the end of the queue. |
| API result format | `response_format` | Use `"url"` for easy previews/downloads. Default is `"b64_json"`. Both save the PNG and metadata on disk. |

With Klein or Qwen selected, upload a photo and write an instruction such as “Change the jacket to green; keep the face, pose and background.” The service follows the photo's aspect ratio. Reference resolution starts on Auto; Qwen also supports a selected-area mask; multiple reference photos are not exposed. Turn off description expansion when you want to send your instruction exactly as written.

## Edit an existing photo

1. Set up an editing model, for example `make setup MODEL=qwen-image-2.1-turbo`, then run `make serve`.
2. Select **Edit photo**, then choose an editing model. The tab is available whenever at least one setup model supports editing, even if the Generate tab currently uses Z-Image or Chroma. On the first visit it opens with an empty model selector; choose a setup Klein or Qwen model before submitting. Later visits restore your editing model. If none are setup, the tab stays disabled and its hover/focus tooltip names suitable models to set up.
3. Choose the source photo and describe only what to change. Add optional requirements under **What should stay the same?**
4. For Qwen, paint over the parts of the source preview you want to change; **Selected area** is chosen automatically. Paint several separate areas, use **Rectangle** to drag boxes, switch to **Erase** to trim, adjust **Brush size** (or press `[` / `]`), and use **Undo** (⌘Z / Ctrl+Z) or **Clear selection** to start over. **Whole photo** remains available; Klein supports whole-photo editing only. With Qwen, the service also tells the model where your selection is, as described below.
5. Leave reference resolution on **Auto** to detect the photo size (up to a 2048px long edge), or choose a different long edge. Keep the model's default steps initially. Refinement uses a separate editing policy; **Preview refinement** shows the exact instruction. It cannot inspect the source photo.
6. Select **Edit photo** to submit. Several options repeat the same requested edit with different seeds; composition variation is unavailable in this mode. Clear the selection to choose a new area.

To edit a saved result, open **Gallery**, select the image and choose **Edit photo** at the top of its details. The photo opens directly in the Edit preview with a blank edit instruction and **Auto** resolution. This works even if Z-Image or Chroma created it: the service selects an eligible original/default editor or the sole setup editor; otherwise choose an editing model. Describe your change and submit when ready. Nothing is submitted automatically. With no setup editor, the disabled action's help names models you can install.

The Edit preview uses a single box: upload or select a gallery photo to show it centered at the same display scale as generated images, then the completed edit replaces that preview. Removing the photo returns the box to its empty state. The source file stays unchanged.

### How a selected area steers the edit (Qwen)

A mask alone only decides which pixels are kept; the model never sees it. So for every masked Qwen edit the service automatically adds one line to the instruction that names where the selection is, as `{"bbox_2d": [x1, y1, x2, y2]}` boxes on the 0–1000 scale of the image that Qwen's text encoder was trained to read. There is no setting for this: choosing **Selected area** (or sending `mask_image`) is the signal. Whole-photo edits, Klein and ordinary generation send exactly the prompts they always did.

- Each separately painted area becomes its own box, largest first, up to **eight**. A long thin stroke is split into two to four boxes along its length so the hint follows it instead of boxing everything inside a curve. Any painted pixel counts, including a minimum-size dot, a one-pixel stroke or a faint grey area. Smaller areas beyond the eighth are still edited but are not named.
- Boxes are measured on a coarse 64×64 grid of the image Qwen actually receives (after resizing and padding), so they are approximate, a little larger than the painted area, and never exact outlines.
- For inference the mask is expanded to the 16×16-pixel patches the model works in, keeping soft grey values, so thin strokes are not blended back to the source. Your exact painted mask is still used at the end to copy every unselected pixel from the source, so pixels outside the selection are unchanged.
- The location line counts toward the **4000-character limit on the complete prompt** (your instruction, preservation requirements, the service's editing guidance and the boxes; the boxes take up to about 450 characters). Prompt refinement leaves room for them. The service never shortens your instruction, drops preservation text or leaves out boxes to fit: an over-long masked edit fails before any image is generated, with a message asking you to shorten the text or turn refinement off.

Treat this as a nudge, not a guarantee. Boxes tell the model *where*, not *what makes sense there*: asking for leaves across a forehead can still produce no leaves, and describing the object in scene terms (“a crown of leaves on top of her head”) works better. Improvement has only been judged on a handful of comparison images, so keep an eye on results and describe the location in words when a hint isn't enough.

Use the underlined **Generate** / **Edit photo** tabs to change workflows. With a tab focused, ← / → or Home / End switches between available tabs; the active tab is teal.

Generate and Edit keep separate drafts and previews while the page is open, including model/settings, source photo, selected area, expansion text and selected image option. Switching tabs never brings the other tab's photo across. Background jobs update their own tab, and both tabs can follow a job at the same time through the shared queue. Use **Edit image** or a gallery reuse action when you want to transfer a saved image intentionally. Reloading the page clears form drafts, but accepted jobs and images remain saved.

The **Generate** view continues to offer Klein and Qwen for creating images from text, without requiring a source photo. Selected areas also work through both APIs: pass `mask_image` with `task:"edit"` in the generator's flat request, or inside the standalone queue's `request` envelope with `model:"qwen-image-2.1-turbo"`. White mask pixels edit; black pixels preserve. See the [photo-editing examples](docs/api.md#selected-area-qwen-edits).

## Add your own LoRA

For an optional Realism LoRA on Z-Image, run `make download-realism MODEL=z-image-turbo` before starting the generator. Normal setup downloads no LoRAs. Different seeds or composition prompts can help variety, but strongly constrained scenes and preserved references may still look similar.

You can download a LoRA from Hugging Face or import one you already have. Your adapters and their settings are saved separately from project code, so `git pull` keeps them intact.

1. Run `make setup` for the base model first. The custom importer currently supports **Z-Image Turbo, Chroma HD and Chroma Flash**; Klein/Qwen adapter formats are not supported by this installer.
2. Find a **LoRA `.safetensors` file trained for that model**. Check the creator's license, trigger words and recommended strength. A LoRA for a different model family will not work just because it has the same file extension.
3. In the project folder, run:

   ```sh
   make setup-lora
   ```

4. Choose the model (Enter defaults to Z-Image Turbo). Paste a local file path, a Hugging Face repository such as `owner/repository`, or the LoRA's Hugging Face file-page/download URL. If the repository has several weight files, choose the adapter file from the list.
5. Enter a display name, a short ID, optional trigger words and a default strength (Enter defaults to `0.8`; allowed range `0–2`). The command adds `custom-` to the ID: `ink` becomes `custom-ink`.
6. Reload the webpage, select that model and choose the new **Style adapter**. The service loads the weights and inserts configured trigger words automatically. For the API, discover it with `GET /v1/loras?model=z-image-turbo`, then use `"lora":"custom-ink"` and optionally `"lora_strength":0.8`. Do not put `<lora:...>` tags in the prompt.

For example, to import a downloaded file without questions:

```sh
make setup-lora MODEL=z-image-turbo \
  LORA_SOURCE="$HOME/Downloads/style.safetensors" \
  LORA_ID=ink LORA_NAME="Ink illustration" \
  LORA_TRIGGER="creator-trigger" LORA_STRENGTH=0.8 NONINTERACTIVE=1
```

Replace the example path and trigger with the creator's actual values; omit `LORA_TRIGGER` if none is required. See [download options](docs/setup.md#custom-lora-downloads-and-imports) for Hugging Face filenames, revisions, checksums and private repositories.

Weights and `custom.json` live in the selected model's ignored `loras` folder, on your configured external drive or inside the project. The original source file is kept. Chroma HD/Flash share one adapter folder, with each adapter listed only for models you explicitly selected during installation. Rerunning the same ID, weights and settings verifies them and makes no changes; use a new ID for a different version or settings. Cancel with `q` at a question or Ctrl+C; existing registrations and resumable downloads are kept.

The importer checks file format and model-family hints; it cannot guarantee exact compatibility or image quality. Installing an adapter does not set up its base model or start a server. An already running server normally needs only a page reload; if it was started with `NO_LORAS=1`, restart it with adapters enabled when convenient.

## Search and manage saved images

The gallery brings together results from all supported models, including images saved to custom output folders. Search by prompt or tags, sort and browse pages, and open an image to see its settings, download it, edit its tags or use it as a starting point for another generation. Choose **Edit photo** to open the selected image directly in the Edit preview, even when a different model created it. The action is available whenever a setup model supports editing. Choose an editor if prompted, describe the change and submit; your original file stays unchanged. **Make a variation** and **Reuse settings** remain separate actions.

Open **[Gallery](http://localhost:5742/gallery)** to search saved images by prompt or tags, browse pages, edit tags and reuse settings. Images and JSON settings are saved automatically in `<storage folder>/<model>/images`. The webpage footer shows the default save folder; **Save location** lets you override it for a request.

To delete a result, open it in **Gallery**, choose **Delete image** below its file path, then review the thumbnail and prompt in the confirmation popup. **Cancel** or Escape keeps it; **Delete image** permanently removes the PNG, its JSON settings, tags and cached previews, including files in custom save folders. There is no undo. Uploaded references, other images and job history stay saved. If a waiting or running job uses the image, finish or cancel that job first. The gallery refreshes after deletion and moves back a page if necessary.

![Searchable image gallery](docs/screenshots/gallery.png)

This screenshot uses simulated demonstration data.

## Manage the queue

Open **Queue** in any page's header to see active and waiting jobs, saved history, pause/clear controls and pagination. **Monitor** follows a job in the preview without changing its place. Drag a waiting job by its left-hand grip to reorder it, or focus the grip and press ↑ / ↓. **Run Now** puts it in the next free slot, even when ordinary work is paused. An active batch always finishes first. Close the modal with **Close**, Escape or the backdrop.

You can cancel waiting jobs, clear the waiting list and retry failed jobs. These actions keep generated images and history. Order and accepted jobs survive restarts. The generator reads/writes SQLite directly, so it remains usable if the standalone queue API stops.

Jobs keep their requested model. If its files become unavailable after admission, the job stays waiting with an explanation while other eligible jobs continue. New requests for models that need setup are rejected; run setup first.

![Queue modal](docs/screenshots/queue.png)

This screenshot uses simulated demonstration data; its timings are not benchmarks.

## Monitor resource use

Choose **Observability** in the header, or open [http://localhost:5742/observability](http://localhost:5742/observability). Select 5m, 30m, 1h, 6h, 24h, 3 days or 7 days, and optionally filter by model. Hover a graph—or focus it and use the arrow keys—to inspect values and model names. Solid lines show sample means; dashed lines preserve sampled peaks in longer windows.

Choose **View job** in the measured-jobs table, or **Metrics** beside an attempted queue entry. Job view offers **Entire job**, **Last minute**, **Last 5 minutes** and **Last 15 minutes**, ending at completion for finished jobs. Retries/recovery attempts are identified separately. The table has Previous/Next pagination. The job’s **Generation logs** panel shows available stages, runtime output and errors; expand a source to read its latest text. New attempts have separate logs, while native image logs show their latest contents. Older jobs can have no logs. Live updates refresh the webpage every five seconds; switching them off does not stop server-side recording.

The generator records every second during jobs and every five seconds while idle. No samples are backfilled for old jobs or periods when it was stopped. Raw samples are retained for eight days; finished attempt summaries remain afterward. History is stored at `state/observability/metrics.sqlite3` inside the project, separate from the queue and external model store.

Service CPU is a percentage of the machine’s total CPU capacity. Service RAM sums resident memory (RSS) across the generator and its children; shared pages may be counted twice. GPU utilization and driver memory are **whole-device** readings, so other apps can affect them. GPU memory and process RAM overlap in unified memory and should not be added together. Missing counters stay blank; sampled peaks can miss brief spikes. No administrator password is needed.

Below the graphs, **Model load & unload** shows individual events rather than a continuous usage curve. Its **Event history** menu is independent of the graph window: choose 24 hours, 7 days, **30 days** (default), 90 days or **All recorded history**. The model filter applies to both; **Operation** selects load/unload. The summary cards average only successful operations across every matching event, while the timeline and Previous/Next table show the current page. Open **Triggered by** to inspect the job that caused a switch. In job view, events span the entire job and can include unloading a previous model, even when the resource graph shows only the last minute.

**Runtime start/stop** measures child process startup/imports and confirmed process-group shutdown/cleanup. **Weights load/release** measures the resident Z-Image constructor and adapter replacement/cache release. They can overlap; do not add their durations. Chroma, Klein and Qwen load weights inside per-image CLIs, so their exact weight phases are not separately reported. First-inference compilation is not a weight-load timer. Failed operations keep their measured failure duration; interrupted operations without a known finish show a dash. Warm reuse does not count as a new load. Events stay in project `state/observability/metrics.sqlite3` until locally removed; they start recording after this upgrade and are not backfilled. Restart a running generator to enable the new instrumentation.

## Generate through the API

Keep the generator running. You can try requests directly in **[the interactive API docs](http://localhost:5742/docs)**: expand an endpoint, click **Try it out**, edit its JSON and click **Execute**. Or use the terminal steps below.

### Submit a request

Open another terminal and run:

```sh
curl --fail-with-body http://localhost:5742/v1/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: manual-fox-1' \
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

The response returns a job `id` with status `queued` or `running`. **HTTP 202 means saved, not finished.** Leaving out `model` uses the shared saved default; specifying another setup model loads it automatically when scheduled. If you started with `NO_LLM=1`, set `expand_prompt` to false.

The idempotency key prevents duplicate work if you retry this same request after losing its response. Use a **new key for each intentional new request**; reusing a key with different settings returns 409.

### Check progress and get the picture

Copy the returned ID into the variable below:

```sh
IMAGEGEN_JOB_ID="paste-the-returned-id-here"
curl --fail-with-body "http://localhost:5742/v1/jobs/$IMAGEGEN_JOB_ID"
```

Run the second command again occasionally until the status is `succeeded`, `failed` or `cancelled`. On success, **`result.data[]`** contains every generated image, with its `url`, full `file_path`, `metadata_path` and seed. Open an image's `url` in your browser, or use the file already saved on disk. On failure, read the job's `error`. Closing your browser or timing out a request does not cancel a saved job.

### Move a waiting job or run it next

Read `queue_position` from a job/list response. It starts at 1 across all waiting jobs, independent of pagination. Send the desired position to move it up/down:

```sh
curl --fail-with-body --request POST "http://localhost:5742/v1/jobs/$IMAGEGEN_JOB_ID/move" \
  -H 'Content-Type: application/json' -d '{"position":1}'
curl --fail-with-body --request POST "http://localhost:5742/v1/jobs/$IMAGEGEN_JOB_ID/run-now"
```

Run Now marks that same job immediate and moves it first; it doesn't interrupt active work. These controls also exist on port 55302, using the job ID without a model query, even when the generator is off. Running/finished jobs return 409. If another client changes the queue, refresh positions before trying again.

### Discover models, adapters and saved images

```sh
curl --fail-with-body http://localhost:5742/v1/models
curl --fail-with-body http://localhost:5742/v1/loras
curl --fail-with-body 'http://localhost:5742/v1/images?q=forest&limit=24&offset=0'
```

Use a LoRA entry with `available:true`, setting `lora` to its `id`. Adapter discovery reports disabled/empty when the generator starts with `NO_LORAS=1`. For API reference photos, pass `reference_image` as a server-local absolute path; see [uploading references](docs/api.md#reference-photos-and-variations) if you prefer sending image bytes.

## Queue images while the generator is off

After setup, run **only** the lightweight queue service:

```sh
make queue
```

It runs in the background and returns control to your terminal. It doesn't load image models or the LLM and has no webpage. Its **[API docs](http://localhost:55302/docs)** remain available while the generator is off.

Submit the image settings inside a `request` object:

```sh
curl --fail-with-body http://localhost:55302/v1/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: manual-queued-fox-1' \
  -d '{
    "model": "z-image-turbo",
    "request": {
      "prompt": "A watercolor fox in a snowy forest",
      "size": "512x512",
      "response_format": "url"
    }
  }'
```

On this standalone API, omitted `model` uses the **shared saved default**, just like the generator API. A fresh setup prefers ready Z-Image Turbo, otherwise the first ready model; the resolved model is saved with the job. Ordinary work joins the end of the queue; you can reorder it later. The queue accepts settings for later validation; a saved job may still fail when the generator processes it.

List or inspect queued work with:

```sh
curl --fail-with-body 'http://localhost:55302/v1/jobs?model=z-image-turbo&status=pending&limit=24&offset=0'
IMAGEGEN_JOB_ID="paste-the-queue-job-id-here"
curl --fail-with-body "http://localhost:55302/v1/jobs/$IMAGEGEN_JOB_ID"
```

When you're ready to generate, run `make serve MODEL=z-image-turbo`. It processes saved jobs across all setup models; this command selects Z-Image as the default for new requests. The queue API returns raw saved metadata and local paths, with batch files under `result.images[]`; it doesn't serve image download URLs. Use the generator's job endpoint for `result.data[]` with URLs, or browse the gallery.

Queue editing, cancelling, clearing, retrying and pausing are described in the [API reference](docs/api.md#independent-queue-api-generator-may-be-off). The UI continues working through the generator API if the standalone queue service stops; `make queue` restores access for outside tools.

## Local hardware, storage and backups

Quantized model weights and Apple's Metal GPU help make local inference practical. Image options render sequentially, and the prompt-expansion process releases its memory before image rendering. FLUX.2 Klein and Qwen use pre-quantized MLX checkpoints and low-memory child processes, releasing their weights after each image. Resolution, image count and model choice affect memory use and generation time; Chroma can take substantially longer; compare models at the same resolution on your hardware.

Model downloads and generated media can live inside the project or on an external SSD. Setup remembers the location and creates links when needed. The source code, Python environment and SQLite queue database stay in the project. Back up both your model/media store and the project's `state/` folder to preserve images, tags and waiting work. Downloaded weights and personal media are excluded from Git.

## Stop and return later

1. Press **Ctrl-C** in the generator terminal. Its active batch finishes before shutdown; waiting work stays saved.
2. The queue API stays up. To stop it too, run `make stop-queue` from the project folder.
3. Next time, run `make serve` again. No setup is needed unless you're adding a model or repairing an incomplete download.

Pictures and settings live in your chosen storage folder; queue databases live in project `state/`. Back up both to retain images, tags and pending work. Downloaded models and generated media are excluded from Git.

## If something goes wrong

- **Interrupted download:** rerun `make setup MODEL=<model-id>` to reuse or resume it.
- **External drive missing:** reconnect it; the generator won't silently download to another location.
- **Missing prompt LLM:** run `make download-llm`, or start with `make serve NO_LLM=1`.
- **Port conflict:** use `make serve PORT=5733` and open that port in the browser.
- **Slow generation or memory pressure:** try smaller images or fewer options and close other GPU-heavy apps.

`make health` checks a running generator. `make test` uses simulated downloads and generation; `make smoke` generates a real test image on a running server. Further details are in the [setup reference](docs/setup.md#verification-and-troubleshooting).

See the [setup reference](docs/setup.md) for pinned model sources, runtime compatibility and all startup settings, or the [API reference](docs/api.md) for the complete HTTP contract. For AI tools, use the [operating skill](skill/image-model-server/SKILL.md); contributors should read [AGENTS.md](AGENTS.md).
