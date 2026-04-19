"""SD 1.5 + ControlNet pipeline wrapper (seg or canny conditioning)."""
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
    control_image: Image.Image | None = None,
    gen_config_path: str = "configs/generation.yaml",
) -> tuple[Image.Image, dict[str, Any], Image.Image]:
    """Generate one image with SD 1.5 + ControlNet.

    Pass control_image to bypass internal construction (e.g. when the caller
    has already computed Canny edges from a breed reference photo).

    Returns (generated image, metadata dict, control image used).
    """
    cfg = _load_gen_config(gen_config_path)
    params = get_generation_params(gen_config_path)
    model_id: str = cfg["models"]["base"]

    cn_type = controlnet_type or cfg["controlnet"]["primary"]
    cn_model_id: str = cfg["models"][f"controlnet_{cn_type}"]
    conditioning_scale: float = cfg["controlnet"]["conditioning_scale"]

    if control_image is None:
        control_image = _build_control_image(source_image, trimap, cn_type)

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
        source_image_path=None,
        controlnet_model_id=cn_model_id,
        conditioning_scale=conditioning_scale,
        breed=breed,
        species=species,
        condition=condition,
        environment=environment,
    )
    metadata["controlnet_type"] = cn_type
    return image, metadata, control_image
