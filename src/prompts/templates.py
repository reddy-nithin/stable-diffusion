"""Prompt render functions backed by configs/prompts.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def _load_templates(path: str = "configs/prompts.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def render_naive(breed: str, condition: str, path: str = "configs/prompts.yaml") -> str:
    """Fill the naive template: 'a {breed} {condition}'."""
    tmpl = _load_templates(path)["naive"]
    return tmpl.format(breed=breed, condition=condition)


def render_structured(
    *,
    style: str,
    breed: str,
    species: str,
    condition_clause: str,
    environment_clause: str,
    path: str = "configs/prompts.yaml",
) -> str:
    """Fill the structured template with all semantic slots."""
    tmpl = _load_templates(path)["structured"]
    return tmpl.format(
        style=style,
        breed=breed,
        species=species,
        condition_clause=condition_clause,
        environment_clause=environment_clause,
    )


def render_negative(path: str = "configs/prompts.yaml") -> str:
    """Return the shared negative prompt string."""
    return _load_templates(path)["negative"]
