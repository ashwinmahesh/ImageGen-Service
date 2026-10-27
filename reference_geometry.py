"""Preserve reference composition while keeping inference on its 64-pixel grid."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_REFERENCE_EDGE = 2048


class ReferenceError(ValueError):
    pass


def output_dimensions(reference_size, longest_edge):
    """Fit the oriented photo to the requested long edge, rounded to whole pixels."""
    width, height = reference_size
    if width >= height:
        return longest_edge, max(1, (height * longest_edge + width // 2) // width)
    return max(1, (width * longest_edge + height // 2) // height), longest_edge


@dataclass(frozen=True)
class Geometry:
    image_path: str | None
    model_size: tuple[int, int]
    output_size: tuple[int, int]
    reference_size: tuple[int, int] | None = None

    @property
    def crop_box(self):
        width, height = self.output_size
        left = (self.model_size[0] - width) // 2
        top = (self.model_size[1] - height) // 2
        return left, top, left + width, top + height

    def finish(self, path):
        if self.reference_size is None:
            return
        with Image.open(path) as generated:
            if generated.size != self.model_size:
                raise RuntimeError(f"Unexpected generated dimensions: {generated.size}")
            cropped = generated.crop(self.crop_box)
        if self.output_size != self.model_size:
            cropped.save(path, format="PNG")


def edge_pad(photo, canvas_size):
    """Extend edge pixels into padding without stretching or cropping the photo."""
    width, height = photo.size
    canvas = Image.new("RGB", canvas_size)
    left = (canvas.width - width) // 2
    top = (canvas.height - height) // 2
    canvas.paste(photo, (left, top))
    if left:
        canvas.paste(photo.crop((0, 0, 1, height)).resize((left, height)), (0, top))
    right = canvas.width - width - left
    if right:
        canvas.paste(photo.crop((width - 1, 0, width, height)).resize((right, height)), (left + width, top))
    if top:
        canvas.paste(canvas.crop((0, top, canvas.width, top + 1)).resize((canvas.width, top)), (0, 0))
    bottom = canvas.height - height - top
    if bottom:
        canvas.paste(canvas.crop((0, top + height - 1, canvas.width, top + height)).resize((canvas.width, bottom)), (0, top + height))
    return canvas


@contextmanager
def prepare_geometry(reference_path, requested_size, scratch_root):
    automatic = requested_size == 'auto'
    requested = None if automatic else tuple(map(int, requested_size.split("x")))
    if reference_path is None:
        if automatic:
            raise ReferenceError('Auto resolution requires a reference photo')
        yield Geometry(None, requested, requested)
        return
    try:
        if not Path(reference_path).is_file():
            raise ReferenceError("File does not exist or is not a regular file")
        with Image.open(reference_path) as source:
            if source.width * source.height > 40_000_000:
                raise ReferenceError("Reference photo must be at most 40 megapixels")
            orientation = source.getexif().get(274, 1)
            photo = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ReferenceError(f"Cannot read reference_image {reference_path}: {exc}") from exc
    long_edge = min(max(photo.size), MAX_REFERENCE_EDGE) if automatic else max(requested)
    output_size = output_dimensions(photo.size, long_edge)
    model_size = tuple(max(256, ((dimension + 63) // 64) * 64) for dimension in output_size)
    if photo.size == output_size == model_size and orientation == 1:
        yield Geometry(reference_path, model_size, output_size, photo.size)
        return
    try:
        scratch_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="reference-", dir=scratch_root)
    except OSError as exc:
        raise ReferenceError(f"Cannot prepare reference photo: {exc}") from exc
    with temporary:
        prepared = Path(temporary.name) / "reference.png"
        try:
            scaled = photo.resize(output_size, Image.Resampling.LANCZOS)
            edge_pad(scaled, model_size).save(prepared, format="PNG")
        except OSError as exc:
            raise ReferenceError(f"Cannot prepare reference photo: {exc}") from exc
        yield Geometry(str(prepared), model_size, output_size, photo.size)


def read_mask(mask_path, reference_size):
    try:
        with Image.open(mask_path) as source:
            if source.width*source.height>40_000_000:
                raise ValueError('Edit mask must be at most 40 megapixels')
            mask=ImageOps.exif_transpose(source).convert('L')
        if mask.size!=reference_size:
            raise ValueError('Edit mask dimensions must match the oriented source photo')
        if mask.getextrema()[1]==0:
            raise ValueError('Edit mask must select at least one pixel (white edits, black preserves)')
        return mask
    except (OSError,ValueError,Image.DecompressionBombError) as error:
        raise ReferenceError(f'Cannot use mask_image: {error}') from error


@contextmanager
def prepare_mask(mask_path, geometry, scratch_root):
    if mask_path is None:
        yield None
        return
    mask=read_mask(mask_path,geometry.reference_size)
    mask=mask.resize(geometry.output_size,Image.Resampling.NEAREST)
    canvas=Image.new('L',geometry.model_size,0)
    canvas.paste(mask,geometry.crop_box[:2])
    scratch_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='edit-mask-',dir=scratch_root) as temporary:
        path=Path(temporary)/'mask.png'
        canvas.save(path)
        yield str(path)


def widen_mask_to_grid(mask_path, grid=16):
    """Select every model patch the mask touches, so thin strokes are not blended back to the source.

    Qwen edits in 16x16 latent patches and pins each one toward the source in proportion
    to how much of it is unselected. composite_edit still trims to the exact painted mask.
    """
    with Image.open(mask_path) as image:
        mask=image.convert('L').point(lambda value:255 if value>127 else 0)
    columns,rows=-(-mask.width//grid),-(-mask.height//grid)
    patches=mask.resize((columns,rows),Image.Resampling.BOX).point(lambda value:255 if value else 0)
    widened=patches.resize((columns*grid,rows*grid),Image.Resampling.NEAREST).crop((0,0,mask.width,mask.height))
    path=Path(mask_path).with_name('model-mask.png')
    widened.save(path)
    return str(path)


def composite_edit(path, source_path, mask_path):
    """Copy unselected source pixels exactly at the prepared output resolution."""
    with Image.open(path) as image:
        generated=image.convert('RGBA')
    with Image.open(source_path) as image:
        source=image.convert('RGBA')
    with Image.open(mask_path) as image:
        mask=image.convert('L')
    if generated.size!=source.size or mask.size!=source.size:
        raise ReferenceError('Masked edit output does not match the prepared source and mask')
    # Qwen already blends soft mask values. Copy only fully unselected pixels
    # here, rather than applying fractional alpha a second time.
    editable=mask.point(lambda value:255 if value else 0)
    Image.composite(generated,source,editable).save(path,format='PNG')
