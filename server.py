"""Single-model, single-worker localhost image API for Apple Silicon."""
import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
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

from model_profiles import PROFILES
from config import PROFILE, MODEL_ROOTS, RUNTIME_DIR, API_MODEL, IMAGE_DIR, MODEL_DIR, MODEL_ID, MODEL_REVISION, ROOT, PROJECT_ROOT
from storage_paths import require_storage
from lora_catalog import list_loras, resolve_lora, effective_prompt
from reference_geometry import prepare_geometry, ReferenceError
import image_catalog

# Downloads are an explicit separate step. Serving never needs the network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

log = logging.getLogger("image-model-server")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-inference")
busy = threading.Lock()
model = None


def load_model():
    global model
    require_storage(ROOT)
    if not (MODEL_DIR / "download.json").exists():
        raise RuntimeError(f"Model is missing or incomplete. Run `make download MODEL={API_MODEL}` first.")
    if PROFILE['backend']=='mflux':
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


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_running_loop().run_in_executor(executor, load_model)
    yield
    executor.shutdown(wait=True)


app = FastAPI(
    title=f"Image Model Server — {PROFILE['name']}", version="2.0.0", lifespan=lifespan,
    description=(
        "Local image generation on Apple Silicon. Images and JSON metadata are saved to "
        f"`{IMAGE_DIR}` by default. Set `output_dir` in a generation request to choose another "
        "absolute directory on this Mac. Every response includes `file_path` and `metadata_path`. "
        "These are server-local filesystem paths; `url` is the browser-accessible PNG. "
        "Use `reference_image` for one local reference photo (image-to-image); `image_strength` "
        "controls how strongly the starting image is preserved. The webpage can also upload a photo "
        "to `/v1/reference-images` and use its returned local path. One image can generate at a time. Interactive webpage: `/`. Browse saved generations at `/gallery` or GET `/v1/images`."
    ),
)
app.state.loras_enabled = True
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal['z-image-turbo','chroma1-hd','chroma1-flash'] = API_MODEL

    @field_validator('model')
    @classmethod
    def validate_model(cls,value):
        if value != API_MODEL:
            raise ValueError(f'This server runs {API_MODEL}; restart with make serve MODEL={value} to switch')
        return value

    prompt: str = Field(min_length=1, max_length=4000, description="Describe the image to generate. API limit: 4000 characters. " + PROFILE["prompt_note"] + " Put essential details first.")
    size: str = Field(default="1024x1024", description="WIDTHxHEIGHT; each dimension must be a multiple of 64 from 256 to 2048. With reference_image, only the larger dimension is used: output automatically follows the oriented reference aspect ratio, rounded to whole pixels. Padding for the model grid is trimmed from the saved image; its short edge may be below 256. Above 1024, VAE tiling reduces memory use. Larger images take longer; 2048x2048 may require closing other GPU-heavy apps on this 24 GB Mac.")
    tags: list[str] = Field(default_factory=list, max_length=32, description="Optional image tags: at most 32 text labels, each 1–64 characters after whitespace normalization. Commas/control characters are disallowed. Duplicates are removed case-insensitively. Editable later with PATCH /v1/images/{id}/tags.")
    n: Literal[1] = 1
    steps: int = Field(default=PROFILE["steps"], ge=1, le=PROFILE["max_steps"], strict=True, description=f"Sampling steps for {PROFILE['name']}; recommended {PROFILE['steps']} (range 1–{PROFILE['max_steps']}).")
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1, strict=True, description="Omit or use null for a random seed. The actual seed is returned.")
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
        description="Optional absolute local image path (~/ supported) for image-to-image generation. One reference only; a path string, not HTTP URLs or raw image bytes. PNG, JPEG, WebP and other Pillow-readable images. Output automatically preserves the oriented photo aspect ratio without stretching. The longer dimension of size sets the output long edge; its width/height shape is ignored. Uploaded paths are supported via /v1/reference-images.",
    )
    image_strength: float = Field(
        default=0.6, ge=0, lt=1,
        description="Reference preservation strength, following MFLUX semantics: higher values retain more of the original and run fewer denoising steps. 0 starts from full noise; 0.6 is a useful starting point; must be less than 1. Only used with reference_image.",
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

    @model_validator(mode="after")
    def validate_reference_settings(self):
        if self.reference_image is not None and self.steps < 2:
            raise ValueError("Reference-image generation requires at least 2 steps")
        if self.reference_image is None and "image_strength" in self.model_fields_set:
            raise ValueError("image_strength requires reference_image")
        if self.lora is None and self.lora_strength is not None:
            raise ValueError("lora_strength requires lora")
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
        try:
            width, height = map(int, value.split("x"))
        except ValueError:
            raise ValueError("Use WIDTHxHEIGHT, such as 1024x1024") from None
        if any(d < 256 or d > 2048 or d % 64 for d in (width, height)):
            raise ValueError("Each dimension must be a multiple of 64 between 256 and 2048")
        return f"{width}x{height}"


class ImageResult(BaseModel):
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
    pass


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


def generate(payload):
    try:
        require_storage(ROOT)
        require_storage(payload.output_dir)
    except RuntimeError as error:
        raise LocalFileError(str(error)) from error
    try:
        with prepare_geometry(payload.reference_image, payload.size, ROOT / "cache" / "scratch") as geometry:
            return generate_with_geometry(payload, geometry)
    except ReferenceError as exc:
        raise LocalFileError(str(exc)) from exc


def generate_with_geometry(payload, geometry):
    try:
        if payload.lora is not None and not app.state.loras_enabled:
            raise ValueError("LoRAs are disabled for this server; restart without --no-loras to enable them")
        selection = resolve_lora(payload.lora, payload.lora_strength) if app.state.loras_enabled else None
    except (ValueError, OSError) as exc:
        raise LocalFileError(str(exc)) from exc
    seed = payload.seed if payload.seed is not None else secrets.randbelow(2**32)
    width, height = geometry.model_size
    output_width, output_height = geometry.output_size
    identifier = uuid.uuid4().hex
    output_dir = Path(payload.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Check writability before spending GPU time; retain the reserved unique filename.
        path = output_dir / f"{identifier}.png"
        with path.open("xb"):
            pass
    except OSError as exc:
        raise LocalFileError(f"Cannot write to output_dir {output_dir}: {exc.strerror}") from exc
    started = time.monotonic()
    try:
        tiled = PROFILE['backend']=='sd-cpp' or max(width,height)>1024
        if PROFILE['backend']=='mflux':
            if tiled:
                from mflux.models.common.vae.tiling_config import TilingConfig
                model.tiling_config=TilingConfig()
            else:
                model.tiling_config=None
        else:
            model.active_log=ROOT/'logs'/(identifier+'.log')
        result = model.generate_image(
            seed=seed, prompt=payload.prompt, width=width, height=height,
            num_inference_steps=payload.steps, lora=selection,
            image_path=geometry.image_path,
            image_strength=payload.image_strength if payload.reference_image is not None else None,
        )
        result.save(str(path), overwrite=True)
        geometry.finish(path)
        elapsed = round(time.monotonic() - started, 2)
        metadata = {
            "backend": PROFILE["backend"], **PROFILE.get("applied", {}),
            "model": API_MODEL, "source": MODEL_ID, "revision": MODEL_REVISION,
            "created": int(time.time()), "tags": payload.tags,
            "prompt": payload.prompt, "seed": seed, "size": f"{output_width}x{output_height}",
            "requested_size": payload.size, "generation_size": f"{width}x{height}",
            "reference_size": list(geometry.reference_size) if geometry.reference_size else None,
            "steps": payload.steps, "seconds": elapsed,
            "file_path": str(path), "metadata_path": str(path.with_suffix(".json")),
            "reference_image": payload.reference_image,
            "image_strength": payload.image_strength if payload.reference_image is not None else None,
            "vae_tiling": tiled, "lora": selection,
            "effective_prompt": effective_prompt(payload.prompt, selection),
            "loras_enabled": app.state.loras_enabled,
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        # Persist custom output locations so their image URLs survive a server restart.
        image_catalog.write_index(ROOT, identifier, path, payload.tags)
        return path, metadata
    except Exception:
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
        raise
    finally:
        if PROFILE['backend']=='mflux':
            import mlx.core as mx
            mx.clear_cache()
        else:
            model.active_log=None


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(PROJECT_ROOT / "ui" / "index.html", media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    state = {"status": "ready" if model is not None else "loading", "model": API_MODEL, "busy": busy.locked(), "project_dir": str(PROJECT_ROOT), "data_dir": str(ROOT), "loras_enabled": app.state.loras_enabled, "capabilities": capabilities()}
    if state['busy'] and hasattr(model,'progress'):
        state['progress']=model.progress()
    return state


def capabilities():
    return {'model':API_MODEL,'name':PROFILE['name'],'backend':PROFILE['backend'],
            'default_steps':PROFILE['steps'],'max_steps':PROFILE['max_steps'],
            'default_size':'1024x1024','prompt_note':PROFILE['prompt_note'],
            'max_prompt_characters':4000,'min_dimension':256,'max_dimension':2048,
            'loras_enabled':app.state.loras_enabled}


@app.get("/v1/loras", summary="List compatible local LoRAs", description="The UI uses this catalog. enabled=false and an empty data list indicate a base-only server started with --no-loras. Only available=true entries can generate. IDs are specific to the active model; null disables LoRA. Entries include trigger text, default strength, source, and installed checksum.")
def loras():
    return {"object": "list", "enabled": app.state.loras_enabled,
            "data": list_loras() if app.state.loras_enabled else []}


@app.get("/v1/models")
def models():
    return {'object':'list','active_model':API_MODEL,'data':[
        {'id':identifier,'object':'model','owned_by':'local','name':profile['name'],
         'active':identifier==API_MODEL,'downloaded':(MODEL_ROOTS[identifier]/'weights'/'download.json').is_file(),
         'default_steps':profile['steps'],'max_steps':profile['max_steps']}
        for identifier,profile in PROFILES.items()]}


@app.post(
    "/v1/images/generations", response_model=GenerateResponse, response_model_exclude_none=True,
    summary="Generate and save an image",
    description="Always saves a PNG plus JSON metadata. Returns their absolute local paths together with a URL or base64 image. output_dir changes only this request's destination. With a reference, its oriented aspect ratio is preserved automatically; size sets the long edge. Returned size/width/height are the saved PNG dimensions.",
    responses={400: {"description": "Output/reference path is invalid, or the selected LoRA is unknown, unavailable, or has changed."},
               429: {"description": "Another image is generating. Retry when /health busy is false."}},
)
async def generations(payload: GenerateRequest, request: Request):
    if payload.lora is not None and not app.state.loras_enabled:
        raise HTTPException(422, "LoRAs are disabled for this server; restart without --no-loras to enable them.")
    if not busy.acquire(blocking=False):
        raise HTTPException(429, "One image is already generating. Try again when /health reports busy=false.", headers={"Retry-After": "10"})
    future = executor.submit(generate, payload)
    # A disconnected client must not unlock GPU access while its image is still running.
    future.add_done_callback(lambda _: busy.release())
    try:
        path, metadata = await asyncio.shield(asyncio.wrap_future(future))
    except LocalFileError as exc:
        raise HTTPException(400, str(exc)) from None
    except Exception:
        log.exception("Generation failed")
        raise HTTPException(500, "Generation failed; see the server log. Try a smaller image.") from None
    output_width, output_height = map(int, metadata["size"].split("x"))
    item = {"tags": metadata["tags"], "seed": metadata["seed"], "file_path": str(path), "metadata_path": str(path.with_suffix(".json")),
            "size": metadata["size"], "width": output_width, "height": output_height,
            "generation_size": metadata["generation_size"]}
    if payload.response_format == "b64_json":
        item["b64_json"] = base64.b64encode(path.read_bytes()).decode("ascii")
    else:
        item["url"] = str(request.url_for("image", identifier=path.stem))
    return {"created": int(time.time()), "data": [item], "generation_seconds": metadata["seconds"]}


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
    return FileResponse(PROJECT_ROOT / "ui" / "gallery.html", media_type="text/html", headers={"Cache-Control": "no-store"})


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


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-loras", action="store_true", help="Serve only the base model; disable the LoRA UI and API")
    parser.add_argument("--model", choices=PROFILES, default=API_MODEL)
    args = parser.parse_args()
    app.state.loras_enabled = not args.no_loras
    if args.no_loras:
        app.description += " This server is running in base-only mode: LoRAs are disabled."
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)
