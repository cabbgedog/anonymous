from pathlib import Path

from PIL import Image

from usir_inference.image_ops import discover_images, resize_long_edge


def test_resize_long_edge_preserves_aspect_ratio():
    image = Image.new("RGB", (2000, 1000))
    assert resize_long_edge(image, 1024).size == (1024, 512)


def test_discover_images_preserves_relative_paths(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "nested" / "x.png")
    assert discover_images(tmp_path) == [(tmp_path / "nested" / "x.png", Path("nested/x.png"))]
