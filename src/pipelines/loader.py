"""Load SD 1.5 and ControlNet pipelines, handling device and dtype automatically."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
import yaml


def _load_gen_config(path: str = "configs/generation.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    # MPS (Apple Silicon) — float32 required for diffusers on MPS
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


@lru_cache(maxsize=1)
def load_baseline_pipeline(gen_config_path: str = "configs/generation.yaml"):
    """Return a StableDiffusionPipeline on the best available device.

    Cached so subsequent calls reuse the loaded weights.
    """
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    cfg = _load_gen_config(gen_config_path)
    model_id = cfg["models"]["base"]
    device, dtype = _device_and_dtype()

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not installed — attention slicing still active

    return pipe


@lru_cache(maxsize=2)
def load_controlnet_pipeline(
    controlnet_type: str = "seg",
    gen_config_path: str = "configs/generation.yaml",
):
    """Return a StableDiffusionControlNetPipeline for seg or canny conditioning.

    Cached per controlnet_type.
    """
    from diffusers import (
        ControlNetModel,
        DPMSolverMultistepScheduler,
        StableDiffusionControlNetPipeline,
    )

    cfg = _load_gen_config(gen_config_path)
    model_id = cfg["models"]["base"]
    cn_key = f"controlnet_{controlnet_type}"
    controlnet_id = cfg["models"][cn_key]
    device, dtype = _device_and_dtype()

    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not installed — attention slicing still active

    return pipe
