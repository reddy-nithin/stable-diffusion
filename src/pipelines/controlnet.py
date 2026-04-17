"""ControlNet-seg / Canny pipeline wrapper — ablation cells B and D.

The control modality is selected by arg (seg | canny).
The primary is whichever is set in configs/generation.yaml under
controlnet.primary; callers can override by passing controlnet_type directly.
"""
from __future__ import annotations

from typing import Any

import torch
import yaml
from PIL import Image

from src.data.masks import image_to_canny, trimap_to_seg_map
from src.generation.io import build_metadata
from src.generation.seeds import get_generation_params
from src.pipelines.loader import load_controlnet_pipeline
from src.prompts.mapper import PromptPair


def _load_gen_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_control_image(
    source_image: Image.Image,
    trimap: Image.Image | None,
    controlnet_type: str,
) -> Image.Image:
    """Produce the conditioning image for the chosen modality.

    seg  → trimap_to_seg_map(trimap)  — requires a trimap from Oxford dataset
    canny → image_to_canny(source_image) — works with any RGB image
    """
    if controlnet_type == "seg":
        if trimap is None:
            raise ValueError(
                "controlnet_type='seg' requires a trimap PIL image. "
                "Pass trimap= or switch to controlnet_type='canny'."
            )
        return trimap_to_seg_map(trimap)
    elif controlnet_type == "canny":
        return image_to_canny(source_image)
    else:
        raise ValueError(f"Unknown controlnet_type '{controlnet_type}'. Use 'seg' or 'canny'.")


def run_controlnet(
    prompt_pair: PromptPair,
    source_image: Image.Image,
    *,
    seed: int,
    breed: str,
    species: str,
    condition: str,
    environment: str,
    cell: str,
    trimap: Image.Image | None = None,
    controlnet_type: str | None = None,
    gen_config_path: str = "configs/generation.yaml",
) -> tuple[Image.Image, dict[str, Any]]:
    """Generate one image with SD 1.5 + ControlNet (seg or canny).

    Parameters
    ----------
    prompt_pair       PromptPair from mapper (positive + negative + mode)
    source_image      Reference Oxford pet image (RGB PIL); used for Canny and
                      stored in metadata as the source for reproducibility.
    seed              Reproducibility seed — must match the same seed used for
                      the corresponding baseline cell to keep comparisons valid.
    breed             Oxford breed name (for metadata)
    species           "cat" or "dog" (for metadata)
    condition         Taxonomy condition key (for metadata)
    environment       Taxonomy environment key (for metadata)
    cell              Ablation cell label: "B" or "D"
    trimap            Oxford trimap PIL image; required when controlnet_type="seg"
    controlnet_type   "seg" | "canny" | None (reads generation.yaml primary)
    gen_config_path   Path to generation.yaml

    Returns
    -------
    (PIL image, metadata dict) — pass to save_image_with_sidecar
    """
    cfg = _load_gen_config(gen_config_path)
    params = get_generation_params(gen_config_path)
    model_id: str = cfg["models"]["base"]

    # Resolve control modality
    cn_type = controlnet_type or cfg["controlnet"]["primary"]
    cn_model_id: str = cfg["models"][f"controlnet_{cn_type}"]
    conditioning_scale: float = cfg["controlnet"]["conditioning_scale"]

    # Build the control conditioning image
    control_image = _build_control_image(source_image, trimap, cn_type)

    # Load pipeline (cached by cn_type)
    pipe = load_controlnet_pipeline(cn_type, gen_config_path)
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

    result = pipe(
        prompt=prompt_pair.positive,
        negative_prompt=prompt_pair.negative,
        image=control_image,
        num_inference_steps=params["steps"],
        guidance_scale=params["cfg_scale"],
        width=params["width"],
        height=params["height"],
        generator=generator,
        controlnet_conditioning_scale=conditioning_scale,
    )
    image: Image.Image = result.images[0]

    metadata = build_metadata(
        cell=cell,
        prompt_positive=prompt_pair.positive,
        prompt_negative=prompt_pair.negative,
        seed=seed,
        steps=params["steps"],
        cfg_scale=params["cfg_scale"],
        model_id=model_id,
        source_image_path=None,  # caller can fill in after save if desired
        controlnet_model_id=cn_model_id,
        conditioning_scale=conditioning_scale,
        breed=breed,
        species=species,
        condition=condition,
        environment=environment,
    )
    metadata["controlnet_type"] = cn_type
    return image, metadata, control_image  # also return control image for debugging
