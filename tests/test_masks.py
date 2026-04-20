"""Tests for border-safe Canny in src/data/masks.py."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

requires_cv2 = pytest.mark.skipif(not _HAS_CV2, reason="opencv-python not installed")


def _image_with_border(size: int = 512, border: int = 2) -> Image.Image:
    """Solid grey image with a white border (simulates JPEG boundary artifacts)."""
    arr = np.full((size, size, 3), 128, dtype=np.uint8)
    arr[:border, :] = 255
    arr[-border:, :] = 255
    arr[:, :border] = 255
    arr[:, -border:] = 255
    return Image.fromarray(arr)


def _checkerboard(size: int = 512, tile: int = 64) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(0, size, tile):
        for c in range(0, size, tile):
            if (r // tile + c // tile) % 2 == 0:
                arr[r : r + tile, c : c + tile] = 255
    return Image.fromarray(arr)


@requires_cv2
@pytest.mark.unit
def test_canny_border_ring_is_zero() -> None:
    """The 16-px border ring must be all zeros — no bounding-box artifact."""
    from src.data.masks import image_to_canny

    edges = image_to_canny(_image_with_border())
    arr = np.array(edges.convert("L"))
    border_px = 16

    assert arr[:border_px, :].sum() == 0, "Top border has edges"
    assert arr[-border_px:, :].sum() == 0, "Bottom border has edges"
    assert arr[:, :border_px].sum() == 0, "Left border has edges"
    assert arr[:, -border_px:].sum() == 0, "Right border has edges"


@requires_cv2
@pytest.mark.unit
def test_canny_preserves_interior_edges() -> None:
    """Interior checkerboard edges must survive the border-safe processing."""
    from src.data.masks import image_to_canny

    edges = image_to_canny(_checkerboard())
    arr = np.array(edges.convert("L"))
    interior = arr[32:480, 32:480]
    assert interior.sum() > 0, "No interior edges detected in checkerboard"


@requires_cv2
@pytest.mark.unit
def test_canny_output_is_512x512_rgb() -> None:
    """Output must always be 512×512 RGB regardless of input size."""
    from src.data.masks import image_to_canny

    for size in [256, 512, 800]:
        img = Image.new("RGB", (size, size), color=(100, 150, 200))
        result = image_to_canny(img)
        assert result.size == (512, 512), f"Expected 512x512, got {result.size} for input {size}"
        assert result.mode == "RGB", f"Expected RGB, got {result.mode}"


@pytest.mark.unit
def test_trimap_to_seg_map_shape() -> None:
    """Seg map must have same spatial dimensions as input trimap."""
    from src.data.masks import trimap_to_seg_map

    trimap_arr = np.ones((128, 128), dtype=np.uint8)  # all foreground
    trimap_arr[10:20, 10:20] = 2  # small background patch
    trimap = Image.fromarray(trimap_arr, mode="L")

    seg = trimap_to_seg_map(trimap)
    assert seg.size == (128, 128)
    assert seg.mode == "RGB"
