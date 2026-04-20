"""Trimap → ADE20K seg map  +  border-safe Canny edge helper  +  depth map.

Oxford trimap values:
  1 = pet foreground
  2 = background
  3 = uncertain / boundary

ADE20K class 126 = "animal" → palette color [255, 0, 122]
Background pixels → [0, 0, 0]
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# ADE20K 150-class palette (index → [R, G, B])
_ADE20K_PALETTE: list[list[int]] = [
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
    [102, 255, 0], [92, 0, 255],
]

_ANIMAL_COLOR = np.array(_ADE20K_PALETTE[126], dtype=np.uint8)  # [255, 0, 122]
_BG_COLOR = np.array([0, 0, 0], dtype=np.uint8)


def trimap_to_seg_map(trimap: Image.Image) -> Image.Image:
    """Map Oxford trimap to ADE20K-style RGB seg map for ControlNet-seg.

    Foreground (1) and uncertain (3) → ADE20K animal class [255, 0, 122].
    Background (2) → [0, 0, 0].
    """
    mask = np.array(trimap.convert("L"))
    seg = np.full((*mask.shape, 3), _BG_COLOR, dtype=np.uint8)
    seg[mask == 1] = _ANIMAL_COLOR
    # Treat uncertain pixels as foreground to avoid eroding the animal region
    seg[mask == 3] = _ANIMAL_COLOR
    return Image.fromarray(seg, mode="RGB")


def image_to_canny(
    image: Image.Image,
    low: int | None = None,
    high: int | None = None,
    cfg_path: str = "configs/generation.yaml",
) -> Image.Image:
    """Extract border-safe Canny edges for ControlNet conditioning.

    Applies center-crop, Gaussian pre-blur, and border-ring erasure before
    running Canny to prevent JPEG/photo-boundary rectangles from appearing
    as bounding boxes in generated images. All thresholds are YAML-configurable.
    """
    import yaml
    from PIL import ImageFilter

    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        canny_cfg = cfg.get("canny", {})
    except Exception:
        canny_cfg = {}

    _low = low if low is not None else canny_cfg.get("low", 80)
    _high = high if high is not None else canny_cfg.get("high", 160)
    border_px = int(canny_cfg.get("border_erase_px", 16))
    blur_r = float(canny_cfg.get("preblur_radius", 1.5))
    crop_pct = float(canny_cfg.get("center_crop_pct", 0.90))

    img = image.convert("RGB").resize((512, 512), Image.LANCZOS)

    # Center-crop to strip JPEG borders before edge detection
    if crop_pct < 1.0:
        w, h = img.size
        cw, ch = int(w * crop_pct), int(h * crop_pct)
        left, top = (w - cw) // 2, (h - ch) // 2
        img = img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)

    # Pre-blur suppresses texture/noise edges that Canny would otherwise pick up
    if blur_r > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_r))

    try:
        import cv2
        arr = np.array(img.convert("L"))
        edges = cv2.Canny(arr, _low, _high)
    except ImportError:
        # Fallback when cv2 is not available (local dev without opencv-python)
        from controlnet_aux import CannyDetector
        detector = CannyDetector()
        result = detector(img, low_threshold=_low, high_threshold=_high)
        edges = np.array(result.convert("L"))

    # Zero out border ring — eliminates any residual outer-rectangle artifact
    if border_px > 0:
        edges[:border_px, :] = 0
        edges[-border_px:, :] = 0
        edges[:, :border_px] = 0
        edges[:, -border_px:] = 0

    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_rgb)


def image_to_depth(image: Image.Image) -> Image.Image:
    """Extract MiDaS depth map for ControlNet-depth conditioning.

    No border-rectangle artifact since depth maps are smooth continuous fields.
    """
    try:
        from controlnet_aux import MidasDetector
    except ImportError as e:
        raise ImportError("pip install controlnet-aux") from e

    detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
    img_512 = image.convert("RGB").resize((512, 512), Image.LANCZOS)
    return detector(img_512)
