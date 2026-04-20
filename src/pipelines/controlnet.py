"""SD 1.5 + ControlNet pipeline wrapper (seg, canny, or depth conditioning)."""
from __future__ import annotations

from typing import Any

import torch
import yaml
from PIL import Image

from src.data.masks import image_to_canny, trimap_to_seg_map
from src.generation.io import build_metadata
from src.generation.seeds import get_generation_params
from src.pipelines.loader import load_controlnet_pipeline, load_img2img_pipeline
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
        raise ValueError(f"Unknown controlnet_type '{controlnet_type}'. Use 'seg', 'canny', or 'depth'.")


def _get_prompt_inputs(pipe, prompt_pair: PromptPair) -> dict[str, Any]:
    """Return prompt kwargs, using compel for long prompts if available."""
    try:
        from compel import Compel
        compel_obj = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
        )
        pos_embeds = compel_obj(prompt_pair.positive)
        neg_embeds = compel_obj(prompt_pair.negative)
        return {
            "prompt_embeds": pos_embeds,
            "negative_prompt_embeds": neg_embeds,
        }
    except Exception:
        return {
            "prompt": prompt_pair.positive,
            "negative_prompt": prompt_pair.negative,
        }


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
    has already computed edges from a breed reference photo).

    Returns (generated image, metadata dict, control image used).
    """
    cfg = _load_gen_config(gen_config_path)
    params = get_generation_params(gen_config_path)
    model_id: str = cfg["models"]["base"]

    cn_type = controlnet_type or cfg["controlnet"]["primary"]
    cn_model_id: str = cfg["models"][f"controlnet_{cn_type}"]
    conditioning_scale: float = cfg["controlnet"]["conditioning_scale"]
    guidance_start: float = cfg["controlnet"].get("control_guidance_start", 0.0)
    guidance_end: float = cfg["controlnet"].get("control_guidance_end", 0.6)

    if control_image is None:
        control_image = _build_control_image(source_image, trimap, cn_type)

    pipe = load_controlnet_pipeline(cn_type, gen_config_path)
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

    prompt_kwargs = _get_prompt_inputs(pipe, prompt_pair)

    result = pipe(
        **prompt_kwargs,
        image=control_image,
        num_inference_steps=params["steps"],
        guidance_scale=params["cfg_scale"],
        width=params["width"],
        height=params["height"],
        generator=generator,
        controlnet_conditioning_scale=conditioning_scale,
        control_guidance_start=guidance_start,
        control_guidance_end=guidance_end,
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
    metadata["control_guidance_start"] = guidance_start
    metadata["control_guidance_end"] = guidance_end
    return image, metadata, control_image


def run_controlnet_img2img(
    prompt_pair: PromptPair,
    init_image: Image.Image,
    control_image: Image.Image,
    *,
    seed: int,
    breed: str,
    species: str,
    condition: str,
    environment: str,
    cell: str,
    controlnet_type: str = "canny",
    gen_config_path: str = "configs/generation.yaml",
) -> tuple[Image.Image, dict[str, Any], Image.Image]:
    """Generate using the user's photo as init image + breed ControlNet conditioning.

    Keeps the user's actual pet structure while applying the veterinary scenario.
    Returns (generated image, metadata dict, control image used).
    """
    cfg = _load_gen_config(gen_config_path)
    params = get_generation_params(gen_config_path)
    model_id: str = cfg["models"]["base"]
    cn_model_id: str = cfg["models"][f"controlnet_{controlnet_type}"]
    conditioning_scale: float = cfg["controlnet"]["conditioning_scale"]
    guidance_start: float = cfg["controlnet"].get("control_guidance_start", 0.0)
    guidance_end: float = cfg["controlnet"].get("control_guidance_end", 0.6)
    strength: float = cfg.get("img2img", {}).get("strength", 0.6)

    pipe = load_img2img_pipeline(controlnet_type, gen_config_path)
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

    prompt_kwargs = _get_prompt_inputs(pipe, prompt_pair)

    result = pipe(
        **prompt_kwargs,
        image=init_image,
        control_image=control_image,
        strength=strength,
        num_inference_steps=params["steps"],
        guidance_scale=params["cfg_scale"],
        generator=generator,
        controlnet_conditioning_scale=conditioning_scale,
        control_guidance_start=guidance_start,
        control_guidance_end=guidance_end,
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
    metadata["controlnet_type"] = controlnet_type
    metadata["img2img_strength"] = strength
    return image, metadata, control_image
