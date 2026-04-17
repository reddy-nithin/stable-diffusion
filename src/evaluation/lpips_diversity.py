"""LPIPS diversity evaluation — perceptual diversity across seeds.

Metric
------
mean_pairwise_lpips
    For each (cell, breed, condition, environment) triple, compute LPIPS
    between every pair of the K seed images.  Average these values.

    Higher LPIPS → more perceptually diverse outputs (same prompt, different
    random seeds produce varied results — good for creative generation).
    Very low LPIPS → the model has collapsed to a single solution regardless
    of seed (bad diversity, possibly mode collapse).

    Target range on T4 with SD 1.5: ~0.35 – 0.65 for balanced diversity.

Backend: torchmetrics LearnedPerceptualImagePatchSimilarity (alex net).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


_LPIPS_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # LPIPS expects [-1,1]
])


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _load_lpips(device: str):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False)
    return metric.to(device)


def _to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    return _LPIPS_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)


@torch.no_grad()
def pairwise_lpips(images: list[Image.Image]) -> float:
    """Mean pairwise LPIPS across K images.

    Parameters
    ----------
    images  ≥ 2 PIL images with the same prompt / triple.

    Returns
    -------
    float; NaN if fewer than 2 images provided.
    """
    if len(images) < 2:
        return float("nan")

    dev    = _device()
    metric = _load_lpips(dev)
    tensors = [_to_tensor(img, dev) for img in images]

    K = len(tensors)
    total, count = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            score = metric(tensors[i], tensors[j]).item()
            total += score
            count += 1

    return total / count if count > 0 else float("nan")


# ---------------------------------------------------------------------------
# Directory-level helper
# ---------------------------------------------------------------------------

def lpips_scores_from_dirs(
    output_root: str | Path,
    cells: list[str],
) -> list[dict]:
    """Compute per-triple pairwise LPIPS for all cells.

    Returns one record per image: includes the triple-level lpips_diversity
    score (same value repeated for all seeds of a triple, for easy joining
    with the CLIP / DINOv2 records).
    """
    import json
    from collections import defaultdict

    out_root = Path(output_root)

    # group images by (cell, breed, condition, environment)
    triple_images:  dict[tuple, list[Image.Image]] = defaultdict(list)
    triple_records: dict[tuple, list[dict]]         = defaultdict(list)
    all_records: list[dict] = []

    for cell in cells:
        cell_dir = out_root / cell
        if not cell_dir.exists():
            continue
        for img_path in sorted(cell_dir.glob("*.png")):
            sidecar = img_path.with_suffix(".json")
            if not sidecar.exists():
                continue
            meta  = json.loads(sidecar.read_text())
            breed = meta.get("breed", "")
            cond  = meta.get("condition", "")
            env   = meta.get("environment", "")
            image = Image.open(img_path).convert("RGB")

            rec = {
                "stem":            img_path.stem,
                "cell":            meta.get("cell", "?"),
                "breed":           breed,
                "condition":       cond,
                "environment":     env,
                "seed":            meta.get("seed", -1),
                "lpips_diversity": float("nan"),  # filled below
            }
            all_records.append(rec)

            key = (cell, breed, cond, env)
            triple_images[key].append(image)
            triple_records[key].append(rec)

    # Back-fill LPIPS per triple
    for key, imgs in triple_images.items():
        score = pairwise_lpips(imgs)
        for rec in triple_records[key]:
            rec["lpips_diversity"] = score

    return all_records
