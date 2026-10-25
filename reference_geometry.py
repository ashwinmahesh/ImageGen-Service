"""Preserve reference composition while keeping inference on its 64-pixel grid."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile

from PIL import Image, ImageOps, UnidentifiedImageError


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
    requested = tuple(map(int, requested_size.split("x")))
    if reference_path is None:
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
    output_size = output_dimensions(photo.size, max(requested))
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
