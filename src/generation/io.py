"""Save generated images and write JSON sidecar metadata files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def save_image_with_sidecar(
    image: Image.Image,
    output_dir: str | Path,
    filename_stem: str,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Save image as PNG and write a matching .json sidecar.

    Parameters
    ----------
    image        PIL image to save
    output_dir   Destination directory (created if absent)
    filename_stem  Base name without extension, e.g. "A_beagle_cone_collar_42"
    metadata     Arbitrary dict — prompt, seed, model, etc.

    Returns
    -------
    (image_path, sidecar_path)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_path = out / f"{filename_stem}.png"
    image.save(img_path)

    full_meta = {**metadata, "sha256": _sha256(img_path)}
    sidecar_path = out / f"{filename_stem}.json"
    sidecar_path.write_text(json.dumps(full_meta, indent=2))

    return img_path, sidecar_path


def build_metadata(
    *,
    cell: str,
    prompt_positive: str,
    prompt_negative: str,
    seed: int,
    steps: int,
    cfg_scale: float,
    model_id: str,
    source_image_path: str | None = None,
    controlnet_model_id: str | None = None,
    conditioning_scale: float | None = None,
    breed: str,
    species: str,
    condition: str,
    environment: str,
) -> dict[str, Any]:
    """Construct the standard metadata dict for a single generation."""
    meta: dict[str, Any] = {
        "cell": cell,
        "breed": breed,
        "species": species,
        "condition": condition,
        "environment": environment,
        "prompt_positive": prompt_positive,
        "prompt_negative": prompt_negative,
        "seed": seed,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "model_id": model_id,
        "width": 512,
        "height": 512,
    }
    if source_image_path is not None:
        meta["source_image_path"] = source_image_path
    if controlnet_model_id is not None:
        meta["controlnet_model_id"] = controlnet_model_id
        meta["conditioning_scale"] = conditioning_scale
    return meta
