"""Seed management — always load from configs/generation.yaml for reproducibility."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def _load_gen_config(path: str = "configs/generation.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def get_seeds(path: str = "configs/generation.yaml") -> list[int]:
    """Return the fixed seed list from generation.yaml."""
    return list(_load_gen_config(path)["seeds"])


def get_generation_params(path: str = "configs/generation.yaml") -> dict[str, Any]:
    """Return steps, cfg_scale, width, height from generation.yaml."""
    cfg = _load_gen_config(path)
    return {
        "steps": cfg["steps"],
        "cfg_scale": cfg["cfg_scale"],
        "width": cfg["width"],
        "height": cfg["height"],
    }
