"""Baseline SD 1.5 text-to-image pipeline — ablation cells A and C (no ControlNet)."""
from __future__ import annotations

from typing import Any

import torch
import yaml
from PIL import Image

from src.generation.io import build_metadata
from src.generation.seeds import get_generation_params
from src.pipelines.loader import load_baseline_pipeline
from src.prompts.mapper import PromptPair


def _load_gen_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_baseline(
    prompt_pair: PromptPair,
    *,
    seed: int,
    breed: str,
    species: str,
    condition: str,
    environment: str,
    cell: str,
    gen_config_path: str = "configs/generation.yaml",
) -> tuple[Image.Image, dict[str, Any]]:
    """Generate one image with the baseline SD 1.5 pipeline (no ControlNet).

    Parameters
    ----------
    prompt_pair   PromptPair from mapper (positive + negative + mode)
    seed          Reproducibility seed
    breed         Oxford breed name (for metadata)
    species       "cat" or "dog" (for metadata)
    condition     Taxonomy condition key (for metadata)
    environment   Taxonomy environment key (for metadata)
    cell          Ablation cell label: "A" or "C"
    gen_config_path  Path to generation.yaml

    Returns
    -------
    (PIL image, metadata dict)  — caller passes to save_image_with_sidecar
    """
    params = get_generation_params(gen_config_path)
    cfg = _load_gen_config(gen_config_path)
    model_id: str = cfg["models"]["base"]
    pipe = load_baseline_pipeline(gen_config_path)

    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

    result = pipe(
        prompt=prompt_pair.positive,
        negative_prompt=prompt_pair.negative,
        num_inference_steps=params["steps"],
        guidance_scale=params["cfg_scale"],
        width=params["width"],
        height=params["height"],
        generator=generator,
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
        breed=breed,
        species=species,
        condition=condition,
        environment=environment,
    )
    return image, metadata
