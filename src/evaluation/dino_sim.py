"""DINOv2 similarity evaluation — breed identity + condition consistency.

Two metrics
-----------
breed_identity
    Cosine similarity between a *generated* image embedding and the mean
    embedding of real Oxford images for the same breed.  Measures how
    recognisable the generated animal is as that breed.

condition_consistency
    Mean pairwise cosine similarity among the K seeds generated for the
    same (breed, condition, environment) triple.  Measures how stable the
    scene is across random seeds — stable conditioning should score high.

Model: facebook/dinov2-base (via torch.hub — no auth required; weights
downloaded once and cached by torch).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


_HUB_REPO  = "facebookresearch/dinov2"
_HUB_MODEL = "dinov2_vitb14"

_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _load_dino(device: str):
    model = torch.hub.load(_HUB_REPO, _HUB_MODEL, pretrained=True)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def embed(images: list[Image.Image]) -> torch.Tensor:
    """Return L2-normalised DINOv2 embeddings, shape (N, D)."""
    dev = _device()
    model = _load_dino(dev)

    batch = torch.stack([_TRANSFORM(img.convert("RGB")) for img in images]).to(dev)
    feats = model(batch)                         # (N, D)
    return F.normalize(feats, dim=-1).cpu()


@torch.no_grad()
def breed_identity_score(
    generated: Image.Image,
    references: list[Image.Image],
) -> float:
    """Cosine similarity between generated image and mean ref embedding.

    Parameters
    ----------
    generated   PIL image produced by the pipeline
    references  Real Oxford images of the same breed (≥1 required)

    Returns
    -------
    float ∈ [-1, 1]; higher = more faithful to breed appearance
    """
    gen_emb  = embed([generated])                # (1, D)
    ref_embs = embed(references)                 # (M, D)
    mean_ref = F.normalize(ref_embs.mean(dim=0, keepdim=True), dim=-1)  # (1, D)
    return float((gen_emb * mean_ref).sum())


@torch.no_grad()
def condition_consistency_score(images: list[Image.Image]) -> float:
    """Mean pairwise cosine similarity across K seed images for one triple.

    Parameters
    ----------
    images  All K generated images for the same (breed, condition, env) triple.
            Needs ≥ 2 images.

    Returns
    -------
    float ∈ [-1, 1]; higher = more consistent scene structure across seeds
    """
    if len(images) < 2:
        return float("nan")

    embs = embed(images)           # (K, D)
    K = embs.shape[0]
    total, count = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            total += float((embs[i] * embs[j]).sum())
            count += 1
    return total / count if count > 0 else float("nan")


# ---------------------------------------------------------------------------
# Directory-level helpers (reads from sidecar JSONs)
# ---------------------------------------------------------------------------

def dino_scores_from_dirs(
    output_root: str | Path,
    cells: list[str],
    dataset,                            # OxfordPetDataset instance
) -> list[dict]:
    """Compute breed_identity and condition_consistency for all cells.

    Scans outputs/<cell>/ for PNGs + sidecars, groups by
    (breed, condition, environment) to compute consistency, and looks up
    Oxford reference images for identity.

    Parameters
    ----------
    output_root  Parent of per-cell output directories.
    cells        E.g. ["A", "B", "C", "D"].
    dataset      OxfordPetDataset — used to pull real reference images per breed.

    Returns
    -------
    List of per-image dicts with breed_identity and condition_consistency fields.
    """
    import json
    from collections import defaultdict

    out_root = Path(output_root)

    # Build breed → reference images index (up to 4 per breed)
    breed_refs: dict[str, list[Image.Image]] = defaultdict(list)
    for i in range(len(dataset)):
        img, _, breed, _ = dataset[i]
        if len(breed_refs[breed]) < 4:
            breed_refs[breed].append(img)

    records: list[dict] = []
    # triple_key → list of PIL images (for consistency)
    triple_images: dict[tuple, list[Image.Image]] = defaultdict(list)
    # triple_key → list of record dicts (to backfill consistency)
    triple_records: dict[tuple, list[dict]] = defaultdict(list)

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

            refs  = breed_refs.get(breed, [])
            identity = breed_identity_score(image, refs) if refs else float("nan")

            rec = {
                "stem":             img_path.stem,
                "cell":             meta.get("cell", "?"),
                "breed":            breed,
                "condition":        cond,
                "environment":      env,
                "seed":             meta.get("seed", -1),
                "breed_identity":   identity,
                "condition_consistency": float("nan"),  # filled below
            }
            records.append(rec)

            triple_key = (cell, breed, cond, env)
            triple_images[triple_key].append(image)
            triple_records[triple_key].append(rec)

    # Back-fill condition_consistency per triple
    for key, imgs in triple_images.items():
        score = condition_consistency_score(imgs)
        for rec in triple_records[key]:
            rec["condition_consistency"] = score

    return records
