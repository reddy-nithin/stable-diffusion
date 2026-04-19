"""Taxonomy loader — reads configs/taxonomy.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def load_taxonomy(path: str = "configs/taxonomy.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def all_breeds(taxonomy: dict | None = None) -> list[str]:
    t = taxonomy or load_taxonomy()
    return t.get("cat_breeds", []) + t.get("dog_breeds", [])


def all_conditions(taxonomy: dict | None = None) -> dict[str, dict]:
    t = taxonomy or load_taxonomy()
    return t.get("conditions", {})


def all_environments(taxonomy: dict | None = None) -> dict[str, dict]:
    t = taxonomy or load_taxonomy()
    return t.get("environments", {})


def all_styles(taxonomy: dict | None = None) -> list[str]:
    t = taxonomy or load_taxonomy()
    return t.get("style_modifiers", [])


def breed_descriptors(taxonomy: dict | None = None) -> dict[str, str]:
    t = taxonomy or load_taxonomy()
    return t.get("breed_descriptors", {})


def species_for_breed(breed: str) -> str:
    return "cat" if breed[0].isupper() else "dog"
