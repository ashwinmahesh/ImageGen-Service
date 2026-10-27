"""One cross-model durable queue, one managed runtime, and a responsive local UI."""
import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import hashlib
import fcntl
import sqlite3
import io
import logging
import os
from pathlib import Path
import secrets
import threading
import time
import uuid
from typing import Literal

# Direct CLI startup must choose the model before importing configuration.
if __name__ == '__main__':
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--model')
    selected,_=pre.parse_known_args()
    if selected.model:
        os.environ['IMAGE_MODEL']=selected.model

from model_profiles import PROFILES, model_details
from photo_workflows import supported_tasks, edit_prompt, describe_mask_region
from config import QUEUE_DATABASE, GPU_LOCK_FILE, PROFILE, MODEL_ROOTS, RUNTIME_DIR, API_MODEL, IMAGE_DIR, MODEL_DIR, MODEL_ID, MODEL_REVISION, ROOT, PROJECT_ROOT
from storage_paths import require_storage
from lora_catalog import list_loras, resolve_lora, effective_prompt
from reference_geometry import prepare_geometry, prepare_mask, read_mask, composite_edit, ReferenceError
import image_catalog
from job_queue import JobStore, QueueConflict, fingerprint, TERMINAL
from generation_worker import QueueWorker
from observability import ResourceMonitor
from observability_api import create_router as observability_router
from queue_storage import initialize_shared_queue,queue_key
from model_assets import available_models,check_model_assets
from model_catalog import default_model,require_ready,configure_schema
from model_process import ModelProcess
from adapter_catalog import model_loras
from queue_client import QueueClient
from queue_api import DEFAULT_QUEUE_PORT, MoveJob
import prompt_expander
from prompt_policy import public_guidance, public_edit_guidance
from enhancer_profile import check_assets as check_enhancer_assets

# Downloads are an explicit separate step. Serving never needs the network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from ui_shell import render_page
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

log = logging.getLogger("image-model-server")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-inference")
busy = threading.Lock()
model = None
job_store = None
job_worker = None
observability_monitor = None
observability_error = None
queue_client = QueueClient(int(os.environ.get("IMAGEGEN_QUEUE_PORT",DEFAULT_QUEUE_PORT)),project=PROJECT_ROOT)
preview_gate = threading.Lock()


def load_model():
    global model
    require_storage(ROOT)
    if not (MODEL_DIR / "download.json").exists():
        raise RuntimeError(f"Model is missing or incomplete. Run `make download MODEL={API_MODEL}` first.")
    notice=PROFILE.get('license',{}).get('notice')
    if notice:
        log.warning('%s: %s %s',PROFILE['name'],notice,PROFILE['license']['url'])
    if PROFILE.get('cli_module'):
        from mflux_engine import MfluxEngine
        model=MfluxEngine()
    elif PROFILE['backend']=='mflux':
        import mlx.core as mx
        from engine import ZImageEngine
        if not mx.metal.is_available():
            raise RuntimeError('Apple Metal GPU is unavailable in this process.')
        mx.set_cache_limit(2 * 1024**3)
        model=ZImageEngine()
    else:
        from chroma_engine import ChromaEngine
        model=ChromaEngine()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Model ready")


def ready_models():
    return available_models(PROJECT_ROOT)


def model_root(identifier):
    return ROOT if identifier==API_MODEL else MODEL_ROOTS[identifier]


def selected_default():
    return default_model(PROJECT_ROOT) or API_MODEL


def ready_or_error(identifier):
    try:require_ready(PROJECT_ROOT,identifier)
    except ValueError as error:raise HTTPException(422,str(error)) from None


def resolve_for_model(payload):
    return resolve_lora(payload.lora,payload.lora_strength,**({'model':payload.model} if payload.model!=API_MODEL else {}))


def initialize_service():
    global job_store,job_worker,model
    if app.state.llm_enabled:check_enhancer_assets(PROJECT_ROOT)
    check_model_assets(PROJECT_ROOT,API_MODEL)
    job_store=JobStore(QUEUE_DATABASE)
    model=ModelProcess(PROJECT_ROOT,loras_enabled=app.state.loras_enabled,llm_enabled=app.state.llm_enabled,
                       stage=lambda value:setattr(job_worker,'stage',value))
    job_worker=QueueWorker(job_store,execute_job,loras_enabled=lambda:app.state.loras_enabled,llm_enabled=lambda:app.state.llm_enabled,
        busy=busy,gpu_lock_file=GPU_LOCK_FILE,observer=observability_monitor,eligible_models=ready_models)
    job_worker.acquire(prepare=lambda:initialize_shared_queue(PROJECT_ROOT))
    job_store.set_setting('default_model',API_MODEL)
    if observability_monitor:observability_monitor.recover_jobs()
    for identifier in ready_models():
        license=PROFILES[identifier].get('license',{})
        if license.get('notice'):log.warning('%s: %s %s',PROFILES[identifier]['name'],license['notice'],license['url'])
    log.info('Generator ready for all setup models; default %s. Models load on demand.',API_MODEL)


@asynccontextmanager
async def lifespan(app):
    global observability_monitor,observability_error
    try:
        observability_monitor=ResourceMonitor(PROJECT_ROOT,API_MODEL,stage=lambda:job_worker.stage if job_worker else 'starting')
        observability_monitor.start()
        observability_error=None
    except Exception:
        log.warning('Resource sampling could not start; generation remains available',exc_info=True)
        observability_monitor=None
        observability_error='Resource sampling could not start; check the project state directory.'
    try:
        await asyncio.get_running_loop().run_in_executor(executor,initialize_service)
        job_worker.start(executor)
        yield
    finally:
        try:
            if job_worker:await asyncio.to_thread(job_worker.stop)
            executor.shutdown(wait=True)
        finally:
            try:
                if isinstance(model,ModelProcess):await asyncio.to_thread(model.close)
            finally:
                if observability_monitor:await asyncio.to_thread(observability_monitor.stop)


app = FastAPI(
    title=f"Image Model Server — {PROFILE['name']}", version="4.0.0", lifespan=lifespan,
    description=(
        "Local image generation on Apple Silicon. Images and JSON metadata are saved to "
        f"`{IMAGE_DIR}` by default. Set `output_dir` in a generation request to choose another "
        "absolute directory on this Mac. Every response includes `file_path` and `metadata_path`. "
        "These are server-local filesystem paths; `url` is the browser-accessible PNG. "
        "Use `reference_image` for one local reference photo. FLUX.2 Klein and Qwen 2.1 Turbo "
        "apply editing instructions; Z-Image and Chroma make variations controlled by `image_strength`. The webpage can also upload a photo "
        "to `/v1/reference-images` and use its returned local path. One image can generate at a time. Interactive webpage: `/`. Browse saved generations at `/gallery` or GET `/v1/images`. "
        "The generator API reads/writes project-local SQLite directly. "
        f"The independent queue API at `{queue_client.url}` accepts agent requests while the generator is stopped. "
        "This generator processes one shared queue across setup models, switching runtimes between batches."
    ),
)
app.state.loras_enabled = True
app.state.llm_enabled = False
app.include_router(observability_router(lambda:observability_monitor,PROJECT_ROOT))
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])


@app.exception_handler(QueueConflict)
async def queue_conflict(request,error):
    return JSONResponse({'detail':str(error)},status_code=409)


@app.exception_handler(sqlite3.Error)
async def queue_storage_error(request,error):
    return JSONResponse({'detail':'Project-local queue storage is unavailable; check its path and permissions, then retry with the same Idempotency-Key for unconfirmed submissions.','code':'queue_storage_unavailable'},status_code=503,headers={'Retry-After':'2'})


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal[tuple(PROFILES)] = API_MODEL
    task: Literal['generate','edit'] = Field(default='generate',description='generate retains the existing generation/reference workflow. edit requires a source photo and a setup editing-capable model; prompt is an edit instruction, with preservation-focused expansion and no composition variation.')
    preserve: str | None = Field(default=None,max_length=1000,description='Edit task only: optional explicit requirements for what must stay unchanged, such as face, pose or background. These are prompt constraints, not a pixel mask or a guarantee.')
    mask_image: str | None = Field(default=None,max_length=4096,description='Qwen photo edits only: absolute local path to a mask matching the oriented source photo. White pixels are edited, black pixels preserve the resized source. Upload mask PNG bytes through /v1/reference-images if needed.')

    @model_validator(mode='before')
    @classmethod
    def model_defaults(cls,value):
        if not isinstance(value,dict):return value
        value=dict(value);selected=value.setdefault('model',selected_default())
        value.setdefault('size','auto' if value.get('reference_image') else '1024x1024')
        if selected in PROFILES:
            value.setdefault('steps',PROFILES[selected]['steps'])
            value.setdefault('output_dir',str(model_root(selected)/'images'))
        return value

    prompt: str = Field(min_length=1, max_length=4000, description="Describe the image to generate. API limit: 4000 characters. " + PROFILE["prompt_note"] + " Put essential details first.")
    size: str = Field(default="1024x1024", description="Omit with reference_image, or use auto, to detect the oriented photo dimensions without upscaling, capped at a 2048px long edge. Without a reference, omitted size is 1024x1024 and auto is invalid. An explicit WIDTHxHEIGHT overrides resolution: each dimension must be a multiple of 64 from 256 to 2048; with a reference, only the larger dimension sets the long edge and output follows its aspect ratio. Padding for the model grid is trimmed from the saved image; detected dimensions need not be multiples of 64. Above 1024, VAE tiling reduces memory use. Larger images take longer and use more memory.")
    tags: list[str] = Field(default_factory=list, max_length=32, description="Optional image tags: at most 32 text labels, each 1–64 characters after whitespace normalization. Commas/control characters are disallowed. Duplicates are removed case-insensitively. Editable later with PATCH /v1/images/{id}/tags.")
    expand_prompt: bool = Field(default=False, strict=True, description="Requires LLM-enabled serving (default; NO_LLM=1 disables it). Keep the description verbatim and append visual detail with the local text LLM before rendering. Original and expanded prompts are persisted. Over-budget expansions fail without compression/truncation. Does not inspect reference photos.")
    vary_prompt: bool = Field(default=False,strict=True,description="Requires expand_prompt:true and n>=2. Plan a distinct LLM expansion/composition per option, preserving explicit original constraints and style. All prompts are persisted before rendering and reused after restart. Default false shares one expansion; the UI enables this for multi-image batches with expansion. Improves creative variety but cannot guarantee visibly different results for constrained prompts or strongly preserved references.")
    generate_immediately: bool = Field(default=False,strict=True,description="false adds a durable shared-queue job. true runs next ahead of ordinary queued work, FIFO among immediate jobs. It never interrupts inference. The UI enables this by default.")
    n: int = Field(default=1,ge=1,le=5,strict=True,description="Number of image options, 1–5 (default 1). One durable batch rendered sequentially with distinct seeds. seed is the first seed; later options increment it modulo 2^32. expand_prompt shares one expansion unless vary_prompt:true requests distinct composition expansions.")
    steps: int = Field(default=PROFILE["steps"], ge=1, le=max(item["max_steps"] for item in PROFILES.values()), strict=True, description="Sampling steps; omitted uses the requested model default. GET /v1/models provides each model’s default_steps and max_steps; values beyond that model’s limit are rejected.")
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1, strict=True, description="Omit or use null for a random seed. The actual seed is returned per image. With n>1 this is the first seed; later options increment it modulo 2^32.")
    response_format: Literal["b64_json", "url"] = "b64_json"
    lora: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,79}$", description="Optional installed adapter ID from GET /v1/loras. Requires /health loras_enabled=true; base-only servers reject non-null adapters. Omit or use null for no LoRA. One adapter per request; model families are not interchangeable.", examples=["realism"])
    lora_strength: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False, description="Adapter influence, 0–2. Requires lora. Omit/null to use the catalog default (Realism: 0.8). This is separate from reference-photo preservation strength.")
    output_dir: str = Field(
        default=str(IMAGE_DIR), min_length=1, max_length=4096,
        description=("Directory on the server's Mac in which to save the PNG and JSON metadata. "
                     "Must be absolute (~/ is supported). Missing directories are created if writable. "
                     "Filenames are generated uniquely; existing images are not overwritten."),
        examples=[str(IMAGE_DIR), "/absolute/path/to/my-project/images"],
    )
    reference_image: str | None = Field(
        default=None, max_length=4096,
        description="Optional absolute local image path (~/ supported). One reference only; a path string, not HTTP URLs or raw image bytes. FLUX.2 Klein and Qwen 2.1 Turbo follow prompt editing instructions; Z-Image and Chroma make image-to-image variations. Output preserves the oriented photo aspect ratio; the longer dimension of size sets its long edge. Uploads are supported via /v1/reference-images.",
    )
    image_strength: float = Field(
        default=0.6, ge=0, lt=1,
        description="Variation models only (Z-Image and Chroma): higher preserves more of the reference and runs fewer denoising steps. Range [0,1), default 0.6. Omit for FLUX.2 Klein and Qwen 2.1 Turbo instruction editing; explicitly supplying it for those models is rejected.",
    )

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, value):
        return image_catalog.normalize_tags(value)

    @field_validator("reference_image")
    @classmethod
    def validate_reference_image(cls, value):
        if value is None:
            return None
        return cls.validate_output_dir(value)

    @field_validator('mask_image')
    @classmethod
    def validate_mask_path(cls,value):
        return cls.validate_output_dir(value) if value is not None else None

    @model_validator(mode="after")
    def validate_reference_settings(self):
        profile=PROFILES[self.model]
        if self.size=='auto' and self.reference_image is None:
            raise ValueError('Auto resolution requires reference_image; use WIDTHxHEIGHT for text-to-image generation')
        if self.steps>profile['max_steps']:raise ValueError(f"steps must be at most {profile['max_steps']} for {profile['name']}")
        if self.task=='edit':
            if 'edit' not in supported_tasks(profile):
                raise ValueError('Photo editing requires FLUX.2 Klein or Qwen-Image 2.1 Turbo; choose a setup editing model in the request')
            if self.reference_image is None:
                raise ValueError('Edit photo requires reference_image')
            if self.vary_prompt:
                raise ValueError('Composition variation is unavailable for photo edits; use n for alternate attempts at the same edit')
            if len(edit_prompt(self.prompt,self.preserve))>4000:
                raise ValueError('Edit instruction and preservation requirements must fit within 4000 characters including editing guidance')
        elif self.preserve is not None:
            raise ValueError('preserve requires task:edit')
        if self.mask_image is not None and (self.task!='edit' or not profile.get('mask_editing')):
            raise ValueError('mask_image requires task:edit with Qwen-Image 2.1 Turbo')
        if profile.get('reference_mode')=='edit' and 'image_strength' in self.model_fields_set:
            raise ValueError('image_strength is unsupported for instruction editing; omit it and describe the desired change in prompt')
        if self.reference_image is not None and self.steps < 2:
            raise ValueError("Reference-image generation requires at least 2 steps")
        if self.reference_image is None and "image_strength" in self.model_fields_set:
            raise ValueError("image_strength requires reference_image")
        if self.lora is None and self.lora_strength is not None:
            raise ValueError("lora_strength requires lora")
        if self.vary_prompt and (not self.expand_prompt or self.n<2):
            raise ValueError('vary_prompt requires expand_prompt:true and n>=2')
        return self

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, value):
        if "\x00" in value:
            raise ValueError("Path cannot contain a null character")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("Path must be absolute or start with ~/")
        return str(path.resolve())

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value):
        if not value.strip():
            raise ValueError("Prompt must contain text")
        if "<lora:" in value.lower():
            raise ValueError("Select LoRAs with the lora and lora_strength fields, not prompt tags")
        return value

    @field_validator("size")
    @classmethod
    def validate_size(cls, value):
        if value=='auto':
            return value
        try:
            width, height = map(int, value.split("x"))
        except ValueError:
            raise ValueError("Use auto with a reference photo, or WIDTHxHEIGHT such as 1024x1024") from None
        if any(d < 256 or d > 2048 or d % 64 for d in (width, height)):
            raise ValueError("Each dimension must be a multiple of 64 between 256 and 2048")
        return f"{width}x{height}"


class PromptExpansion(BaseModel):
    original_prompt: str
    expanded_prompt: str
    mode: Literal['expand']
    model: str
    revision: str
    policy_version: int
    task: Literal['generate','edit'] | None = None
    seconds: float
    original_image_tokens: int | None = None
    expanded_image_tokens: int | None = None
    image_token_limit: int | None = None
    style_anchor: str | None = None
    image_model: str | None = None
    guidance_profile: str | None = None
    guidance_version: int | None = None
    details: dict[str,str] | None = None
    sampling: dict | None = None
    policy_sha256: str | None = None
    structured_output_attempts: int | None = Field(default=None,ge=1,le=2,description="Generation JSON attempts: one normally, two after a formatting retry. Null for older records or edit refinement.")
    structured_output_retry_reason: Literal['invalid_json','invalid_fields','invalid_values','empty_details'] | None = Field(default=None,description="Reason for the one generation-format retry, without raw LLM output.")
    variation_index: int | None = None
    variation_count: int | None = None
    variation_goal: str | None = None
    peak_mlx_memory_gb: float | None = Field(default=None,description="Peak MLX allocator memory in decimal GB in the short-lived LLM child; overlaps process RSS, do not add them.")
    peak_process_rss_gb: float | None = Field(default=None,description="Peak LLM-child resident memory in decimal GB; total service memory is higher because the image model may remain loaded.")


class ImageResult(BaseModel):
    model: str
    task: Literal['generate','edit'] = 'generate'
    candidate_index: int = Field(default=1,description="One-based option position in the batch.")
    candidate_count: int = Field(default=1,description="Requested image count for this batch.")
    prompt_expansion: PromptExpansion | None = None
    tags: list[str] = Field(default_factory=list)
    seed: int
    size: str = Field(description="Actual saved PNG dimensions, WIDTHxHEIGHT. With a reference, these follow its oriented aspect ratio and may differ from the request size.")
    width: int = Field(description="Actual saved PNG width in pixels.")
    height: int = Field(description="Actual saved PNG height in pixels.")
    generation_size: str = Field(description="Internal model canvas dimensions, padded to multiples of 64. Extra padding is removed from reference outputs.")
    file_path: str = Field(description="Absolute path to the saved PNG on this Mac.")
    metadata_path: str = Field(description="Absolute path to the JSON sidecar containing the prompt, seed, size, steps, timing, and paths.")
    url: str | None = Field(default=None, description="PNG URL when response_format is url. Works while this server is running, including after restart if the file remains available.")
    b64_json: str | None = Field(default=None, description="Base64-encoded PNG when response_format is b64_json. The PNG is also saved to disk.")


class GenerateResponse(BaseModel):
    created: int = Field(description="Unix timestamp in seconds.")
    data: list[ImageResult]
    generation_seconds: float


class LocalFileError(Exception):
    http_status=400


MAX_REFERENCE_BYTES = 20 * 1024**2
MAX_REFERENCE_PIXELS = 40_000_000


class ReferenceResult(BaseModel):
    file_path: str = Field(description="Absolute path to the uploaded reference, stored as PNG on the SSD. Pass this as reference_image in a generation request.")
    url: str = Field(description="Local URL for previewing the uploaded reference.")
    width: int
    height: int


def save_reference(data: bytes):
    try:
        require_storage(ROOT)
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_REFERENCE_PIXELS:
                raise HTTPException(413, "Reference photo must be at most 40 megapixels.")
            # Normalize orientation and strip metadata; save a separate copy of the source.
            photo = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        raise HTTPException(400, "Cannot decode this photo. Choose a PNG, JPEG, or WebP image.") from None
    folder = ROOT / "references"
    path = folder / f"{uuid.uuid4().hex}.png"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as target:
            photo.save(target, format="PNG")
    except OSError:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "Cannot save the reference photo. Check that the SSD is connected and writable.") from None
    return path, photo.size


@app.post(
    "/v1/reference-images", response_model=ReferenceResult, status_code=201,
    summary="Upload a reference photo",
    description="Send raw image bytes (not multipart or JSON). Saves an oriented PNG copy in references/ on the SSD. Pass the returned file_path as reference_image in a generation request. Uploads remain on disk for reuse.",
    openapi_extra={"requestBody": {"required": True, "content": {
        "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
        "image/png": {"schema": {"type": "string", "format": "binary"}},
        "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
        "image/webp": {"schema": {"type": "string", "format": "binary"}},
    }}},
    responses={400: {"description": "Photo cannot be decoded or saved."},
               413: {"description": "File exceeds 20 MiB or image exceeds 40 megapixels."}},
)
async def upload_reference(request: Request):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_REFERENCE_BYTES:
            raise HTTPException(413, "Reference photo must be at most 20 MiB.")
        data.extend(chunk)
    path, (width, height) = await asyncio.to_thread(save_reference, bytes(data))
    return {"file_path": str(path), "url": str(request.url_for("reference_photo", identifier=path.stem)),
            "width": width, "height": height}


@app.get("/references/{identifier}.png", name="reference_photo", include_in_schema=False)
def reference_photo(identifier: str):
    if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
        raise HTTPException(404)
    path = next((folder/"references"/f"{identifier}.png" for folder in catalog_roots().values() if (folder/"references"/f"{identifier}.png").is_file()), ROOT/"references"/f"{identifier}.png")
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


def generate(payload, job=None):
    if model is None:load_model()
    try:
        require_storage(ROOT)
        require_storage(payload.output_dir)
    except RuntimeError as error:
        raise LocalFileError(str(error)) from error
    try:
        with prepare_geometry(payload.reference_image, payload.size, ROOT / "cache" / "scratch") as geometry:
            with prepare_mask(payload.mask_image,geometry,ROOT/'cache'/'scratch') as mask_path:
                return generate_with_geometry(payload, geometry, job, mask_path=mask_path)
    except ReferenceError as exc:
        raise LocalFileError(str(exc)) from exc


def generate_with_geometry(payload, geometry, job=None, *, mask_path=None):
    try:
        if payload.lora is not None and not app.state.loras_enabled:
            raise ValueError("LoRAs are disabled for this server; restart without --no-loras to enable them")
        selection = resolve_lora(payload.lora, payload.lora_strength) if app.state.loras_enabled else None
        if job and selection and selection["sha256"]!=job.get("_adapter_sha"):
            raise ValueError("The queued adapter changed; re-submit with the intended adapter")
    except (ValueError, OSError) as exc:
        raise LocalFileError(str(exc)) from exc
    seed = payload.seed if payload.seed is not None else secrets.randbelow(2**32)
    width, height = geometry.model_size
    output_width, output_height = geometry.output_size
    identifier = job['id'] if job else uuid.uuid4().hex
    output_dir = Path(payload.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Check writability before spending GPU time; retain the reserved unique filename.
        final_path = output_dir / f"{identifier}.png"
        if final_path.exists():
            raise LocalFileError('An output already occupies this job ID; refusing to overwrite it')
        path=output_dir/f'.{identifier}.working.png' if job else final_path
        # Only the same interrupted job may replace its own private working file.
        with path.open('wb' if job else 'xb'):
            pass
    except OSError as exc:
        raise LocalFileError(f"Cannot write to output_dir {output_dir}: {exc.strerror}") from exc
    started = time.monotonic()
    try:
        child_backend=PROFILE['backend']=='sd-cpp' or bool(PROFILE.get('cli_module'))
        instruction_edit=PROFILE.get('reference_mode')=='edit'
        tiled = PROFILE.get('vae_tiling',False) if PROFILE.get('cli_module') else PROFILE['backend']=='sd-cpp' or max(width,height)>1024
        if not child_backend:
            if tiled:
                from mflux.models.common.vae.tiling_config import TilingConfig
                model.tiling_config=TilingConfig()
            else:
                model.tiling_config=None
        else:
            model.active_log=ROOT/'logs'/(identifier+'.log')
        region=describe_mask_region(read_mask(payload.mask_image,geometry.reference_size)) if mask_path else None
        rendering_prompt=edit_prompt(payload.prompt,payload.preserve,region) if payload.task=='edit' else payload.prompt
        result = model.generate_image(
            seed=seed, prompt=rendering_prompt, width=width, height=height,
            num_inference_steps=payload.steps, lora=selection,
            **({'inherited_fds':job.get('_inherited_fds',()),'job_id':identifier} if job and child_backend else {}),
            image_path=geometry.image_path,
            image_strength=payload.image_strength if payload.reference_image is not None and not instruction_edit else None,
            **({'mask_path':mask_path} if mask_path else {}),
        )
        result.save(str(path), overwrite=True)
        if mask_path:
            composite_edit(path,geometry.image_path,mask_path)
        geometry.finish(path)
        elapsed = round(time.monotonic() - started, 2)
        metadata = {
            "backend": PROFILE["backend"], **PROFILE.get("applied", {}),
            "model": API_MODEL, "source": MODEL_ID, "revision": MODEL_REVISION,
            "created": int(time.time()), "tags": payload.tags,
            "prompt": job['request']['prompt'] if job else payload.prompt, "seed": seed, "size": f"{output_width}x{output_height}",
            "requested_size": payload.size, "generation_size": f"{width}x{height}",
            "reference_size": list(geometry.reference_size) if geometry.reference_size else None,
            "steps": payload.steps, "seconds": elapsed,
            "file_path": str(final_path), "metadata_path": str(final_path.with_suffix(".json")),
            "reference_image": payload.reference_image,
            "image_strength": payload.image_strength if payload.reference_image is not None and not instruction_edit else None,
            "vae_tiling": tiled, "lora": selection,
            "effective_prompt": effective_prompt(rendering_prompt, selection),
            "loras_enabled": app.state.loras_enabled,
        }
        if payload.task=='edit':
            metadata.update(task='edit',preserve=payload.preserve,mask_image=payload.mask_image,
                            edit_scope='mask' if payload.mask_image else 'whole_photo')
        if PROFILE.get('cli_module'):
            metadata[API_MODEL]={**model_details(API_MODEL),'low_ram':True,
                                 'reference_mode':'edit' if payload.reference_image else 'none'}
        if job:
            metadata.update(job_id=job.get('_batch_id',identifier),request_fingerprint=fingerprint(job['request']))
            if job.get('_batch_id'):
                metadata.update(batch_id=job['_batch_id'],candidate_index=job['_candidate_index'],candidate_count=job['request']['n'])
            if job.get('prompt_expansion'):
                metadata.update(prompt_expansion=job['prompt_expansion'],expanded_prompt=payload.prompt)
                metadata.setdefault(API_MODEL,{})['prompt_expansion_guidance']=public_edit_guidance(API_MODEL) if payload.task=='edit' else public_guidance(API_MODEL)
        temporary=output_dir/f'.{identifier}.metadata.tmp'
        temporary.write_text(json.dumps(metadata,indent=2)+'\n')
        temporary.replace(final_path.with_suffix('.json'))
        if job:
            path.replace(final_path)
        path=final_path
        # Persist custom output locations so their image URLs survive a server restart.
        image_catalog.write_index(ROOT, identifier, path, payload.tags)
        return path, metadata
    except Exception:
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
        raise
    finally:
        if PROFILE['backend']=='mflux' and not PROFILE.get('cli_module'):
            import mlx.core as mx
            mx.clear_cache()
        else:
            model.active_log=None


@app.get("/", include_in_schema=False)
def root():
    return render_page(PROJECT_ROOT,"index.html")


@app.get("/assets/help.css", include_in_schema=False)
def ui_help_styles():
    return FileResponse(PROJECT_ROOT / "ui" / "assets" / "help.css", media_type="text/css", headers={"Cache-Control": "no-store"})


@app.get("/assets/help.js", include_in_schema=False)
def ui_help_script():
    return FileResponse(PROJECT_ROOT / "ui" / "assets" / "help.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})


@app.get("/assets/shell.css", include_in_schema=False)
def ui_shell_styles():
    return FileResponse(PROJECT_ROOT / "ui/assets/shell.css", media_type="text/css", headers={"Cache-Control":"no-store"})


@app.get("/assets/shell.js", include_in_schema=False)
def ui_shell_script():
    return FileResponse(PROJECT_ROOT / "ui/assets/shell.js", media_type="text/javascript", headers={"Cache-Control":"no-store"})


@app.get("/health")
def health():
    state = {"status": "ready" if model is not None else "loading", 'service':'imagegen-generator','version':'4.0.0','runtime_model_switching':True,'queue_scope':'shared','default_model':default_model(PROJECT_ROOT),'active_model':model.active_model if isinstance(model,ModelProcess) else API_MODEL if model else None,'model_loaded':model.loaded if isinstance(model,ModelProcess) else model is not None,  "busy": busy.locked() or preview_gate.locked(), "prompt_expansion": prompt_expander.expansion_capability(app.state.llm_enabled,selected_default()), "project_dir": str(PROJECT_ROOT), "data_dir": str(model_root(selected_default())), "loras_enabled": app.state.loras_enabled, "capabilities": capabilities()}
    if observability_monitor:state['observability']=observability_monitor.status()
    elif observability_error:state['observability']={'recording':False,'error':observability_error}
    if state['busy'] and hasattr(model,'progress'):
        state['progress']=model.progress()
    if job_store:
        try:
            state['queue']=job_store.summary()
            state['busy']=state['busy'] or bool(state['queue']['running'])
            state['queue']['stage']=job_worker.stage if job_worker else 'idle'
            state['queue']['blocked_reason']='Next job requires LoRAs; restart with adapters enabled or cancel it.' if state['queue']['next_requires_lora'] and not app.state.loras_enabled else None
            if state['queue'].get('next_requires_llm') and not app.state.llm_enabled:
                state['queue']['blocked_reason']='Next job requires prompt expansion; restart with make serve and choose Yes for LLM expansion or cancel it.'
            if state['queue'].get('next_model') not in ready_models() and state['queue'].get('next_model'):
                state['queue']['blocked_reason']='A waiting model is not set up or its storage is unavailable; other eligible jobs can continue.'
            state['queue']['error']=job_worker.error if job_worker else None
            if state['queue']['error']: state['status']='error'
        except (OSError,RuntimeError,sqlite3.Error):
            state.update(status='error',queue={'error':'Queue storage unavailable'})
    state['queue_api']=queue_client.status()
    return state


def capabilities(identifier=None,*,ready=None):
    identifier=identifier or selected_default();profile=PROFILES[identifier]
    ready=ready_models() if ready is None else ready
    return {'model':identifier,'name':profile['name'],'backend':profile['backend'],
        'default_steps':profile['steps'],'max_steps':profile['max_steps'],
        identifier:{**model_details(identifier),'prompt_expansion_guidance':public_guidance(identifier)},
        'reference_mode':profile.get('reference_mode','variation'),'image_strength_supported':profile.get('reference_mode')!='edit',
        'tasks':supported_tasks(profile),'mask_editing':bool(profile.get('mask_editing')),
        'photo_editing':{'supported':'edit' in supported_tasks(profile),'models':[key for key in ready if 'edit' in supported_tasks(PROFILES[key])],
            'setup_hint':'Set up '+', '.join(profile['name'] for profile in PROFILES.values() if 'edit' in supported_tasks(profile))+'. Use make setup to add one.',
            'requires_reference':True,'default_reference_size':'auto','default_long_edge':None,'max_long_edge':2048,'composition_variation':False,'mask_supported':bool(profile.get('mask_editing')),
            'prompt_expansion_guidance':public_edit_guidance(identifier) if 'edit' in supported_tasks(profile) else None},
        'default_size':'1024x1024','prompt_note':profile['prompt_note'],'max_prompt_characters':4000,'min_dimension':256,'max_dimension':2048,
        'loras_enabled':app.state.loras_enabled,'prompt_expansion':prompt_expander.expansion_capability(app.state.llm_enabled,identifier),
        'queue_api_url':queue_client.url,'persistent_queue':True,'immediate_generation':True,'default_image_count':1,'max_image_count':5}


@app.get('/v1/loras',summary='List compatible local LoRAs for a setup model')
def loras(model:Literal[tuple(PROFILES)]|None=None):
    identifier=model or selected_default();ready_or_error(identifier)
    entries=(list_loras() if identifier==API_MODEL else model_loras(PROJECT_ROOT,identifier)[1]) if app.state.loras_enabled else []
    return {'object':'list','model':identifier,'enabled':app.state.loras_enabled,'data':entries if app.state.loras_enabled else []}


@app.get('/v1/models',summary='List setup models available for runtime selection')
def models():
    ready=ready_models()
    return {'object':'list','default_model':default_model(PROJECT_ROOT,ready),'active_model':model.active_model if isinstance(model,ModelProcess) else API_MODEL if model else None,'model_loaded':model.loaded if isinstance(model,ModelProcess) else model is not None,
        'data':[{'id':identifier,'object':'model','owned_by':'local','active':isinstance(model,ModelProcess) and model.active_model==identifier,
                 'downloaded':True,'output_dir':str(model_root(identifier)/'images'),**capabilities(identifier,ready=ready)} for identifier in ready]}


class ExpandPromptRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model:Literal[tuple(PROFILES)]=Field(default_factory=selected_default)
    prompt: str = Field(min_length=1,max_length=4000)
    lora: str | None = None
    lora_strength: float | None = Field(default=None,ge=0,le=2,allow_inf_nan=False)
    task: Literal['generate','edit'] = 'generate'
    preserve: str | None = Field(default=None,max_length=1000)

    @field_validator('prompt')
    @classmethod
    def prompt_text(cls,value):
        return GenerateRequest.validate_prompt(value)

    @model_validator(mode='after')
    def strength_requires_adapter(self):
        if self.lora is None and self.lora_strength is not None:
            raise ValueError('lora_strength requires lora')
        if self.task=='edit' and 'edit' not in supported_tasks(PROFILES[self.model]):
            raise ValueError('Photo-edit instruction refinement requires an editing model')
        if self.task!='edit' and self.preserve is not None:
            raise ValueError('preserve requires task:edit')
        if self.task=='edit' and len(edit_prompt(self.prompt,self.preserve))>4000:
            raise ValueError('Edit instruction and preservation requirements exceed the prompt budget')
        return self


@app.post('/v1/prompts/expand',response_model=PromptExpansion,summary='Preview local prompt expansion',
          description='Requires LLM-enabled serving (the default). Text only; does not inspect a reference image or create an image job. Keeps original description verbatim. Returns 409 if inference/the shared accelerator is occupied; preview is not queued. Expansion failures return 422 and never silently fall back. Long image prompts are not compressed.')
def expand_prompt(payload:ExpandPromptRequest):
    if not app.state.llm_enabled:
        raise HTTPException(422,'Prompt expansion is disabled; restart with make serve and choose Yes for LLM expansion.')
    queue_ready()
    ready_or_error(payload.model)
    if not preview_gate.acquire(blocking=False):
        raise HTTPException(409,'Another prompt preview is running; retry after it completes.')
    gpu=None
    inference_owned=False
    try:
        if payload.lora and not app.state.loras_enabled:
            raise HTTPException(422,'LoRAs are disabled for this server.')
        selection=resolve_for_model(payload)
        # Same order as the image worker: accelerator, then inference lock.
        require_storage(GPU_LOCK_FILE)
        GPU_LOCK_FILE.parent.mkdir(parents=True,exist_ok=True)
        gpu=GPU_LOCK_FILE.open('a+')
        try:
            fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise HTTPException(409,'The shared GPU is busy. Wait for inference to finish, then preview again.') from None
        inference_owned=busy.acquire(blocking=False)
        if not inference_owned:
            raise HTTPException(409,'Image inference is busy; preview again after it finishes.')
        if isinstance(model,ModelProcess):model.close()
        return prompt_expander.expand(payload.prompt,project=PROJECT_ROOT,image_model=payload.model,
               image_weights=model_root(payload.model)/'weights',selection=selection,
               **({'task':'edit','preserve':payload.preserve} if payload.task=='edit' else {}),
               inherited_fds=tuple(handle.fileno() for handle in [job_worker.lease,gpu] if handle))
    except prompt_expander.ExpansionError as error:
        raise HTTPException(422,str(error)) from error
    except ValueError as error:
        raise HTTPException(400,str(error)) from error
    except (OSError,RuntimeError) as error:
        raise HTTPException(503,'Prompt assets or GPU-lock storage unavailable; check setup and storage.') from error
    finally:
        if inference_owned: busy.release()
        if gpu: gpu.close()
        preview_gate.release()


class SyncGenerateRequest(GenerateRequest):
    generate_immediately: bool=Field(default=True,strict=True,description='Synchronous route defaults to immediate priority (next free slot); false waits in ordinary queue order.')


class EditRequest(SyncGenerateRequest):
    task: Literal['edit'] = 'edit'
    reference_image: str = Field(max_length=4096,description='Required absolute local path to the photo to edit. Upload photo bytes through /v1/reference-images first if needed.')


def normalized_request(payload):
    data=payload.model_dump()
    if data['task']=='generate':
        data.pop('task')
    if data['preserve'] is None:
        data.pop('preserve')
    if data['mask_image'] is None:
        data.pop('mask_image')
    # Keep legacy request fingerprints stable for the default disabled feature.
    if not data['expand_prompt']:
        data.pop('expand_prompt')
    if not data['vary_prompt']:
        data.pop('vary_prompt')
    if data['reference_image'] is None or PROFILES[payload.model].get('reference_mode')=='edit':
        data.pop('image_strength',None)
    return data


def queue_ready(require_worker=True):
    if job_store is None:
        raise HTTPException(503,'Queue storage is not ready; check health')
    if require_worker and (job_worker is None or job_worker.error):
        raise HTTPException(503,'Queue worker is not ready; check health and server logs')


def submit_job(payload,key=None,retry_of=None):
    queue_ready()
    data=normalized_request(payload)
    request_hash=fingerprint(data)
    key=queue_key(payload.model,key)
    try:
        existing=job_store.existing(key,request_hash)
        if existing: return existing
        if payload.lora is not None and not app.state.loras_enabled:
            raise HTTPException(422,'LoRAs are disabled for this server; restart without --no-loras to enable them.')
        if payload.expand_prompt and not app.state.llm_enabled:
            raise HTTPException(422,'Prompt expansion is disabled; restart with make serve and choose Yes for LLM expansion or set expand_prompt:false.')
        ready_or_error(payload.model)
        require_storage(model_root(payload.model)); require_storage(payload.output_dir)
        # Check input headers without resizing/re-encoding every waiting request.
        if payload.reference_image:
            from PIL import Image
            try:
                with Image.open(payload.reference_image) as photo:
                    if photo.width*photo.height>MAX_REFERENCE_PIXELS:
                        raise ValueError('Reference photo must be at most 40 megapixels')
                    photo.verify()
            except (OSError,ValueError,Image.DecompressionBombError) as error:
                raise ReferenceError(f'Cannot read reference_image: {error}') from error
        if payload.mask_image:
            from PIL import ImageOps
            with Image.open(payload.reference_image) as photo:
                read_mask(payload.mask_image,ImageOps.exif_transpose(photo).size)
        selection=resolve_for_model(payload)
        record=job_store.enqueue(data,request_hash,key=key,
            adapter_sha=selection['sha256'] if selection else None,retry_of=retry_of)
        job_worker.wake.set()
        return record
    except QueueConflict as error:
        raise HTTPException(409,str(error)) from None
    except (ReferenceError,ValueError) as error:
        raise HTTPException(400,str(error)) from None
    except (OSError,RuntimeError) as error:
        raise HTTPException(503,'Cannot persist the job. Check storage before retrying.') from error


def result_images(metadata):
    return metadata.get('images',[metadata]) if metadata else []


def batch_result(images):
    return {'images':list(images),'created':images[0]['created'],
            'seconds':round(sum(image['seconds'] for image in images),2)}


def candidate_identifier(identifier,index,count):
    return identifier if count==1 else uuid.uuid5(uuid.UUID(identifier),f'image-{index+1}').hex


def committed_candidate(payload,job,identifier,index):
    path=Path(payload.output_dir)/(identifier+'.png')
    metadata=image_catalog.read_metadata(path.with_suffix('.json'))
    if not (path.is_file() and metadata):
        return None
    if metadata.get('job_id')!=job['id'] or metadata.get('request_fingerprint')!=fingerprint(job['request']):
        raise LocalFileError('Existing output does not match this queued job; refusing to overwrite it')
    if payload.n>1 and (metadata.get('candidate_index')!=index+1 or metadata.get('candidate_count')!=payload.n or metadata.get('seed')!=(payload.seed+index)%2**32):
        raise LocalFileError('Saved output does not match this batch option; refusing to overwrite it')
    if payload.vary_prompt:
        saved=job.get('prompt_expansion') or {}
        candidates=saved.get('candidates',[])
        if saved.get('mode')!='vary' or len(candidates)!=payload.n or metadata.get('prompt_expansion')!=candidates[index]:
            raise LocalFileError('Saved output does not match its planned composition; it was left untouched')
    from PIL import Image
    with Image.open(path) as image:
        if image.format!='PNG' or list(image.size)!=list(map(int,metadata['size'].split('x'))):
            raise LocalFileError('Saved output is invalid; it was left untouched')
        image.verify()
    if not (ROOT/'image-index'/(identifier+'.json')).is_file():
        image_catalog.write_index(ROOT,identifier,path,metadata['tags'])
    return metadata


def prepare_job_prompt(payload,job):
    selection=resolve_for_model(payload) if payload.lora else None
    if selection and selection['sha256']!=job.get('_adapter_sha'):
        raise LocalFileError('The queued adapter changed; re-submit with the intended adapter')
    if payload.expand_prompt:
        if not app.state.llm_enabled:
            raise LocalFileError('Prompt expansion is disabled; restart with make serve and choose Yes for LLM expansion')
        if not job.get('prompt_expansion'):
            job_worker.stage='expanding_prompt'
            expansion=prompt_expander.expand(payload.prompt,project=PROJECT_ROOT,image_model=API_MODEL,
                  **({'task':'edit','preserve':payload.preserve} if payload.task=='edit' else {}),
                  image_weights=MODEL_DIR,selection=selection,inherited_fds=job.get('_inherited_fds',()))
            job['prompt_expansion']=job_store.save_expansion(job['id'],expansion)
        prompt_expander.validate_expansion(payload.prompt,job['prompt_expansion'])
        payload=payload.model_copy(update={'prompt':job['prompt_expansion']['expanded_prompt']})
    return payload


def prepare_candidate_prompts(payload,job):
    if not payload.vary_prompt:
        prepared=prepare_job_prompt(payload,job)
        return [prepared]*payload.n,[job.get('prompt_expansion')]*payload.n
    selection=resolve_for_model(payload) if payload.lora else None
    if selection and selection['sha256']!=job.get('_adapter_sha'):
        raise LocalFileError('The queued adapter changed; re-submit with the intended adapter')
    if not app.state.llm_enabled:
        raise LocalFileError('Prompt expansion is disabled; restart with make serve and choose Yes for LLM expansion')
    if not job.get('prompt_expansion'):
        job_worker.stage='planning_variations'
        candidates=prompt_expander.expand_batch(payload.prompt,count=payload.n,project=PROJECT_ROOT,
                image_model=API_MODEL,image_weights=MODEL_DIR,selection=selection,
                inherited_fds=job.get('_inherited_fds',()))
        prompt_expander.validate_variations(payload.prompt,candidates,payload.n)
        job['prompt_expansion']=job_store.save_expansion(job['id'],{'mode':'vary','candidates':candidates})
    saved=job['prompt_expansion']
    if saved.get('mode')!='vary':
        raise LocalFileError('Saved expansion does not match the varied batch')
    candidates=saved['candidates']
    prompt_expander.validate_variations(payload.prompt,candidates,payload.n)
    return [payload.model_copy(update={'prompt':item['expanded_prompt']}) for item in candidates],candidates


def execute_job(job):
    if isinstance(model,ModelProcess):return model.execute(job)
    payload=GenerateRequest.model_validate(job['request'])
    if not job.get('_prepared'):
        data=normalized_request(payload)
        data['seed']=payload.seed if payload.seed is not None else secrets.randbelow(2**32)
        selection=resolve_for_model(payload)
        if selection and job.get('_adapter_sha') and selection['sha256']!=job['_adapter_sha']:
            raise LocalFileError('The queued adapter changed; re-submit with the intended adapter')
        if selection: data['lora_strength']=selection['strength']
        frozen=job_store.prepare(job['id'],data,selection['sha256'] if selection else None)
        job.update(frozen)
        payload=GenerateRequest.model_validate(job['request'])
    rendered=[]
    prepared=None
    expansions=None
    for index in range(payload.n):
        identifier=candidate_identifier(job['id'],index,payload.n)
        metadata=committed_candidate(payload,job,identifier,index)
        if metadata is None:
            if prepared is None:
                prepared,expansions=prepare_candidate_prompts(payload,job)
            job_worker.stage=f'generating_option_{index+1}_of_{payload.n}' if payload.n>1 else 'generating'
            candidate=prepared[index].model_copy(update={'n':1,'seed':(payload.seed+index)%2**32})
            child={**job,'id':identifier,'prompt_expansion':expansions[index]}
            if payload.n>1: child.update(_batch_id=job['id'],_candidate_index=index+1)
            _,metadata=generate(candidate,child)
        rendered.append(metadata)
        if payload.n>1:
            job_store.save_progress(job['id'],batch_result(rendered))
    return rendered[0] if payload.n==1 else batch_result(rendered)


def generation_response(path,metadata,request,response_format='url'):
    items=[]
    for saved in result_images(metadata):
        image_path=Path(saved['file_path'])
        width,height=map(int,saved['size'].split('x'))
        item={'model':saved['model'],'tags':saved['tags'],'seed':saved['seed'],'file_path':str(image_path),'metadata_path':str(image_path.with_suffix('.json')),
              'task':saved.get('task','generate'),
              'size':saved['size'],'width':width,'height':height,'generation_size':saved['generation_size'],
              'candidate_index':saved.get('candidate_index',1),'candidate_count':saved.get('candidate_count',1)}
        if saved.get('prompt_expansion'): item['prompt_expansion']=saved['prompt_expansion']
        if response_format=='b64_json': item['b64_json']=base64.b64encode(image_path.read_bytes()).decode('ascii')
        else: item['url']=str(request.url_for('image',identifier=image_path.stem))
        items.append(item)
    return {'created':metadata['created'],'data':items,'generation_seconds':metadata['seconds']}


class JobResponse(BaseModel):
    id: str
    model: str
    status: Literal['queued','running','succeeded','failed','cancelled']
    created: float
    started: float | None=None
    finished: float | None=None
    attempts: int
    generate_immediately: bool
    queue_position: int | None=Field(default=None,description="One-based position among all waiting jobs; absent once running or finished.")
    request: dict
    error: str | None=None
    retry_of: str | None=None
    prompt_expansion: PromptExpansion | None=None
    candidate_expansions: list[PromptExpansion] | None=None
    completed_images: int = 0
    total_images: int = 1
    blocked_reason: str | None=None
    partial_result: GenerateResponse | None = None
    result: GenerateResponse | None=None


class JobList(BaseModel):
    data: list[JobResponse]
    total: int
    limit: int
    offset: int
    has_more: bool
    next_offset: int | None=None
    has_previous: bool
    previous_offset: int | None=None


def job_response(job,request,*,include_result=True,ready=None):
    item={**job,'model':job['request']['model']}
    if job['status']=='queued':
        if item['model'] not in (ready_models() if ready is None else ready):item['blocked_reason']='Model not set up or storage unavailable; run make setup MODEL='+item['model']
        elif job['request'].get('expand_prompt') and not app.state.llm_enabled:item['blocked_reason']='Prompt expansion is disabled for this run'
        elif job['request'].get('lora') and not app.state.loras_enabled:item['blocked_reason']='LoRAs are disabled for this run'
    if job.get('prompt_expansion') and job['prompt_expansion'].get('mode')=='vary':
        item['candidate_expansions']=job['prompt_expansion']['candidates']
        item['prompt_expansion']=None
    metadata=job['result']
    count=job['request'].get('n',1)
    item['total_images']=count if type(count) is int and 1<=count<=5 else 1
    item['completed_images']=len(result_images(metadata))
    response=generation_response(None,metadata,request,job['request'].get('response_format','url')) if metadata and include_result else None
    item['result']=response if job['status']=='succeeded' else None
    item['partial_result']=response if job['status']!='succeeded' else None
    return item


@app.post('/v1/jobs',response_model=JobResponse,status_code=202,response_model_exclude_none=True,
          summary='Persist an image job',description='Accepts generation or task:"edit" with reference_image and optional preserve. Qwen selected-area edits also accept mask_image: white edits, black preserves at output resolution. Default generate_immediately=false joins the end of the queue. true runs next ahead of ordinary waiting jobs without interrupting current work. A 202 is returned only after the SQLite commit. Same Idempotency-Key and settings return the original job; reusing it with different settings returns 409. Poll the job after a disconnect or restart. Uses shared SQLite directly and works while the standalone queue API is down. Storage outages return503/code=queue_storage_unavailable with Retry-After; if a reply was lost, retry the same Idempotency-Key and settings. Any setup model is accepted; omitted model uses the shared default.')
def enqueue_job(payload:GenerateRequest,request:Request,
                idempotency_key:str|None=Header(default=None,max_length=128,pattern=r'^[A-Za-z0-9._:-]+$')):
    return job_response(submit_job(payload,idempotency_key),request)


@app.get('/v1/jobs',response_model=JobList,response_model_exclude_none=True,summary='List persisted jobs',
         description='Shared cross-model queue. Running first, then waiting jobs in execution order, then recent finished jobs. Full results are returned by the detail endpoint. Pagination uses limit/offset.')
def list_jobs(request:Request,limit:int=Query(24,ge=1,le=100),offset:int=Query(0,ge=0),
              status:Literal['pending','queued','running','succeeded','failed','cancelled']|None=None,model:Literal[tuple(PROFILES)]|None=None):
    queue_ready(require_worker=False)
    records,total=job_store.listing(limit,offset,status,model=model)
    ready=ready_models()
    following=offset+len(records)
    return {'data':[job_response(job,request,include_result=False,ready=ready) for job in records],'total':total,'limit':limit,'offset':offset,
            'has_more':following<total,'next_offset':following if following<total else None,
            'has_previous':offset>0,'previous_offset':max(0,offset-limit) if offset else None}


def find_job(identifier):
    queue_ready(require_worker=False)
    if not image_catalog.valid_identifier(identifier): raise HTTPException(404,'Unknown job ID')
    try:
        return job_store.get(identifier)
    except FileNotFoundError:
        raise HTTPException(404,'Unknown job ID') from None


@app.get('/v1/jobs/{identifier}',response_model=JobResponse,response_model_exclude_none=True,summary='Read job status and result')
def get_job(identifier:str,request:Request):
    return job_response(find_job(identifier),request)


@app.patch('/v1/jobs/{identifier}',response_model=JobResponse,response_model_exclude_none=True,
           summary='Replace settings of a never-started waiting job')
def update_job(identifier:str,payload:GenerateRequest,request:Request):
    previous=find_job(identifier)
    if previous['request']['model']!=payload.model:raise HTTPException(422,'A queued job model cannot be changed; submit a new job')
    ready_or_error(payload.model)
    item=job_store.update(identifier,normalized_request(payload))
    job_worker.wake.set()
    return job_response(item,request)


@app.post('/v1/jobs/{identifier}/move',response_model=JobResponse,response_model_exclude_none=True,
          summary='Move a waiting job to a saved queue position',description='position is one-based among all waiting jobs, independent of pagination and priority. Only queued jobs may move; active/finished jobs and stale positions beyond the queue return 409. Request settings and checkpoints are preserved. Uses SQLite directly, including when the queue API is offline.')
def move_job(identifier:str,payload:MoveJob,request:Request):
    find_job(identifier)
    item=job_store.move(identifier,payload.position)
    job_worker.wake.set()
    return job_response(item,request)


@app.post('/v1/jobs/{identifier}/run-now',response_model=JobResponse,response_model_exclude_none=True,
          summary='Run a waiting job at the next free slot',description='Moves the same job to position 1 and enables immediate scheduling priority, even while paused. Never interrupts an active image batch. Only queued jobs qualify (409 otherwise). Preserves the original request, frozen seed, idempotency comparison and progress. Uses SQLite directly.')
def run_job_now(identifier:str,request:Request):
    find_job(identifier)
    item=job_store.run_now(identifier)
    job_worker.wake.set()
    return job_response(item,request)


@app.delete('/v1/jobs/{identifier}',response_model=JobResponse,response_model_exclude_none=True,summary='Cancel one waiting job',
            description='Retains its cancelled record. Running inference cannot be cancelled and returns 409.')
def cancel_job(identifier:str,request:Request):
    find_job(identifier)
    try:
        item=job_store.cancel(identifier)
    except QueueConflict as error:
        raise HTTPException(409,str(error)) from None
    job_worker.wake.set()
    return job_response(item,request)


@app.delete('/v1/jobs',summary='Clear waiting jobs',description='Cancels all waiting jobs across models, or just an explicit model filter. Running work and completed records/files are preserved.')
def clear_jobs(model:Literal[tuple(PROFILES)]|None=None):
    queue_ready(require_worker=False)
    count=job_store.clear(model=model); job_worker.wake.set()
    return {'cancelled':count}


@app.post('/v1/jobs/{identifier}/retry',response_model=JobResponse,status_code=202,response_model_exclude_none=True,
          summary='Retry a failed or cancelled job',description='Creates a new ordinary job with the saved settings/seed and a new output identity.')
def retry_job(identifier:str,request:Request):
    item=find_job(identifier)
    if item['status'] not in ['failed','cancelled']: raise HTTPException(409,'Only failed or cancelled jobs can be retried')
    # A deferred queue request may contain invalid generation fields.
    try:
        payload=GenerateRequest.model_validate({**item['request'],'generate_immediately':False})
    except ValidationError as error:
        raise HTTPException(422,str(error)) from error
    if payload.reference_image and not Path(payload.reference_image).is_file():
        raise HTTPException(400,'Cannot read reference_image; restore it before retrying')
    return job_response(submit_job(payload,retry_of=identifier),request)


class QueueSettings(BaseModel):
    model_config=ConfigDict(extra='forbid')
    paused:bool=Field(strict=True)


@app.patch('/v1/queue',summary='Pause or resume ordinary queued work',description='Persisted across restarts. An active image finishes. Explicit immediate requests may still run while ordinary work is paused.')
def queue_settings(payload:QueueSettings):
    queue_ready(require_worker=False)
    state=job_store.pause(payload.paused); job_worker.wake.set()
    return state


@app.post('/v1/images/generations',response_model=GenerateResponse,response_model_exclude_none=True,
          summary='Generate and wait for the saved image',description='Compatibility synchronous endpoint backed by the same persistent worker. Defaults to generate_immediately=true (next free slot ahead of ordinary queued jobs). false waits for its queue turn. Use POST /v1/jobs for a promptly returned durable job ID and restart-safe polling.')
async def generations(payload:SyncGenerateRequest,request:Request,
                      idempotency_key:str|None=Header(default=None,max_length=128,pattern=r'^[A-Za-z0-9._:-]+$')):
    record=await asyncio.to_thread(submit_job,payload,idempotency_key)
    while record['status'] not in TERMINAL:
        await asyncio.sleep(.1)
        record=await asyncio.to_thread(job_store.get,record['id'])
        if job_worker.error: raise HTTPException(503,job_worker.error)
    if record['status']!='succeeded':
        raise HTTPException(400 if record['status']=='cancelled' else (record.get('error_status') or 500),record['error'] or 'Job was cancelled')
    metadata=record['result']
    return generation_response(None,metadata,request,payload.response_format)


@app.post('/v1/images/edits',response_model=GenerateResponse,response_model_exclude_none=True,
          summary='Edit an existing photo and wait for the result',
          description='JSON request, not multipart. Requires reference_image and a setup editing-capable model. Uses the same durable worker as generation. For asynchronous editing, POST /v1/jobs with task:"edit". Qwen supports optional mask_image: white edits, black preserves the resized source. Klein supports whole-photo instruction editing. Unmasked edits preserve unrequested content through prompt guidance rather than a pixel guarantee.')
async def photo_edits(payload:EditRequest,request:Request,
                      idempotency_key:str|None=Header(default=None,max_length=128,pattern=r'^[A-Za-z0-9._:-]+$')):
    return await generations(payload,request,idempotency_key)


def catalog_roots():
    # ROOT may be redirected by embedding/tests; do not scan unrelated real data then.
    if ROOT != MODEL_ROOTS[API_MODEL]:
        return {API_MODEL:ROOT}
    return MODEL_ROOTS


def image_location(identifier):
    for identifier_model,folder in catalog_roots().items():
        images=IMAGE_DIR if folder==ROOT else folder/'images'
        try:
            path=image_catalog.resolve_image(folder,images,identifier)
            return folder,images,path
        except (OSError,ValueError):
            continue
    raise FileNotFoundError('Saved image is unavailable')


@app.get("/images/{identifier}.png", name="image")
def image(identifier: str):
    try:
        _, _, path = image_location(identifier)
    except (OSError, ValueError):
        raise HTTPException(404, "Saved image is unavailable") from None
    return FileResponse(path, media_type="image/png")


class CatalogImage(BaseModel):
    tags: list[str] = Field(default_factory=list, description="Current editable image tags, stored in the local image index.")
    id: str
    created: int = Field(description="Saved timestamp in Unix seconds; older files use their modification time.")
    url: str
    thumbnail_url: str
    metadata_url: str | None = None
    file_path: str
    metadata_path: str | None = None
    metadata_available: bool
    prompt: str | None = None
    model: str | None = None
    size: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    steps: int | None = None
    generation_seconds: float | None = None
    lora: dict | None = None
    file_size: int


class CatalogDetail(CatalogImage):
    metadata: dict | None = Field(default=None, description="Saved generation settings from the JSON sidecar, when readable.")


class CatalogResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[CatalogImage]
    total: int
    limit: int
    offset: int
    has_previous: bool
    previous_offset: int | None = None
    has_more: bool
    next_offset: int | None = None


def catalog_urls(record, request):
    identifier = record['id']
    return {**record, 'url': str(request.url_for('image', identifier=identifier)),
            'thumbnail_url': str(request.url_for('image_thumbnail', identifier=identifier)) + '?v=' + record['_version'],
            'metadata_url': str(request.url_for('image_metadata', identifier=identifier)) if record['metadata_available'] else None}


@app.get("/gallery", include_in_schema=False)
def gallery():
    return render_page(PROJECT_ROOT,"gallery.html")


@app.get("/v1/images", response_model=CatalogResponse, response_model_exclude_none=True,
         summary="Browse saved generations",
         description="Lists saved PNGs in the default images folder and custom locations recorded in image-index/. Reference uploads are excluded. Search matches prompts, tags, seeds, dimensions, models and adapter labels. Repeated tag parameters require all exact labels (case-insensitive). Pagination uses limit/offset and returns totals and previous/next offsets. Missing files are skipped; images without readable metadata remain browsable. No GPU inference is performed.")
def saved_images(request: Request, response: Response,
                 limit: int = Query(default=24, ge=1, le=100),
                 offset: int = Query(default=0, ge=0),
                 q: str = Query(default="", max_length=4000),
                 sort: Literal["newest", "oldest"] = "newest",
                 tag: list[str] | None = Query(default=None, max_length=32, description="Optional exact tag filters; repeat tag to require all labels. Combined with q.")):
    try:
        records=[]
        require_storage(ROOT)
        for origin,folder in catalog_roots().items():
            if not folder.is_dir():
                if folder==ROOT:
                    raise OSError('Active storage is unavailable')
                continue
            images=IMAGE_DIR if folder==ROOT else folder/'images'
            items,_=image_catalog.list_images(folder,images,q,sort,2**63-1,0,tag)
            records.extend({**item,'model':item['model'] or origin} for item in items)
        records.sort(key=lambda item:(item['created'],item['id']),reverse=sort=='newest')
        total=len(records)
        records=records[offset:offset+limit]
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    except (OSError, RuntimeError):
        raise HTTPException(503, "Cannot read the image library. Check the project storage.") from None
    response.headers['Cache-Control'] = 'no-store'
    next_offset = offset + len(records)
    return {'object': 'list', 'data': [catalog_urls(item, request) for item in records],
            'total': total, 'limit': limit, 'offset': offset,
            'has_previous': offset > 0, 'previous_offset': max(0, offset - limit) if offset > 0 else None,
            'has_more': next_offset < total,
            'next_offset': next_offset if next_offset < total else None}


@app.get("/v1/images/{identifier}", response_model=CatalogDetail, response_model_exclude_none=True,
         summary="Read a saved generation", description="Returns saved settings, actual file paths, and local preview/download URLs for a gallery image.")
def saved_image(identifier: str, request: Request, response: Response):
    try:
        folder,images,_=image_location(identifier)
        record = image_catalog.image_record(folder,images,identifier)
    except (OSError, ValueError):
        raise HTTPException(404, "Saved image is unavailable") from None
    response.headers['Cache-Control'] = 'no-store'
    return catalog_urls(record, request)


class ImageTagsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tags: list[str] = Field(max_length=32, description="Replace all current tags. Send [] to clear. At most 32 labels, each 1–64 characters; commas and control characters are disallowed. Case-insensitive duplicates are removed.")

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, value):
        return image_catalog.normalize_tags(value)


@app.patch("/v1/images/{identifier}/tags", response_model=CatalogDetail, response_model_exclude_none=True,
           summary="Replace a saved image's tags",
           description="Stores editable tags atomically in the local image index. Original PNG and generation sidecar remain unchanged. Works for older default-folder images and indexed custom destinations. No inference is performed.")
def set_image_tags(identifier: str, payload: ImageTagsRequest, request: Request, response: Response):
    try:
        folder,images,_=image_location(identifier)
        record = image_catalog.update_tags(folder,images,identifier,payload.tags)
    except FileNotFoundError:
        raise HTTPException(404, "Saved image is unavailable") from None
    except (OSError, RuntimeError):
        raise HTTPException(503, "Cannot save image tags. Check the project storage.") from None
    response.headers['Cache-Control'] = 'no-store'
    return catalog_urls(record, request)


@app.get("/thumbnails/{identifier}.webp", name="image_thumbnail", include_in_schema=False)
def image_thumbnail(identifier: str):
    try:
        folder,images,_=image_location(identifier)
        path = image_catalog.thumbnail(folder,images,identifier)
    except (OSError, ValueError):
        raise HTTPException(404, "Image preview is unavailable") from None
    return FileResponse(path, media_type="image/webp", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/images/{identifier}.json", name="image_metadata", include_in_schema=False)
def image_metadata(identifier: str):
    try:
        _,_,original=image_location(identifier)
        path=original.with_suffix('.json')
        if image_catalog.read_metadata(path) is None:
            raise FileNotFoundError()
    except (OSError, ValueError):
        raise HTTPException(404, "Saved metadata is unavailable") from None
    return FileResponse(path, media_type="application/json", filename=path.name)


configure_schema(app,PROJECT_ROOT,['GenerateRequest','SyncGenerateRequest','EditRequest','ExpandPromptRequest'],selected_default)

def parse_server_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5742)
    parser.add_argument("--no-loras", action="store_true", help="Serve only the base model; disable the LoRA UI and API")
    parser.add_argument('--no-llm',action='store_true',help='Disable local prompt expansion (default on)')
    parser.add_argument("--model", choices=PROFILES, default=API_MODEL)
    return parser.parse_args(argv)


if __name__ == "__main__":
    import uvicorn
    args=parse_server_args()
    app.state.llm_enabled = not args.no_llm
    log.warning("Local prompt expansion: %s", "enabled (Qwen3.5-2B 4-bit, text only)" if app.state.llm_enabled else "disabled explicitly; restart with make serve to enable")
    app.state.loras_enabled = not args.no_loras
    if args.no_loras:
        app.description += " This server is running in base-only mode: LoRAs are disabled."
    from queue_launch import ensure_queue
    queue_port=int(os.environ.get('IMAGEGEN_QUEUE_PORT',DEFAULT_QUEUE_PORT))
    if args.port==queue_port:
        raise SystemExit('Generator and queue API ports must be different')
    ensure_queue(queue_port)
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)
