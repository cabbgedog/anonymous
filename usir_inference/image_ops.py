"""Input discovery and aspect-ratio-preserving image preparation."""

from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def discover_images(input_path: Path) -> list[tuple[Path, Path]]:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported input image: {input_path}")
        return [(input_path, Path(input_path.name))]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    return [
        (path, path.relative_to(input_path))
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def resize_long_edge(image: Image.Image, max_long_edge: int) -> Image.Image:
    if max_long_edge <= 0:
        raise ValueError("max_long_edge must be positive")
    width, height = image.size
    if max(width, height) <= max_long_edge:
        return image.copy()
    scale = max_long_edge / max(width, height)
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
