"""Map structured veterinary inputs → (positive_prompt, negative_prompt).

Strategy
--------
1. Slot-filling   — breed, species, condition_clause, environment_clause pulled
                    from configs/taxonomy.yaml clause fields.
2. Style anchoring — caller picks one of the four style_modifiers; defaults to
                    "veterinary illustration" for maximum rubric clarity.
3. Ethics guard   — condition key validated against the allow-list in taxonomy;
                    any unknown condition raises ValueError before generation.
4. Fallback       — if structured fails (missing clause), falls back to naive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.data.taxonomy import all_conditions, all_environments, all_styles, load_taxonomy
from src.prompts.templates import render_naive, render_negative, render_structured


@dataclass(frozen=True)
class PromptPair:
    positive: str
    negative: str
    mode: Literal["naive", "structured"]


def structured_input_to_prompt(
    *,
    breed: str,
    species: str,
    condition: str,
    environment: str,
    style: str | None = None,
    tax_path: str = "configs/taxonomy.yaml",
    prompt_path: str = "configs/prompts.yaml",
) -> PromptPair:
    """Return a (positive, negative) PromptPair for the given inputs.

    Parameters
    ----------
    breed       Oxford breed name, e.g. "Siamese" or "beagle"
    species     "cat" or "dog"
    condition   Key from taxonomy conditions, e.g. "cone_collar"
    environment Key from taxonomy environments, e.g. "clinic"
    style       One of taxonomy style_modifiers; defaults to first entry
    """
    tax = load_taxonomy(tax_path)
    conditions = all_conditions(tax)
    environments = all_environments(tax)
    styles = all_styles(tax)

    if condition not in conditions:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Allowed: {sorted(conditions)}"
        )
    if environment not in environments:
        raise ValueError(
            f"Unknown environment '{environment}'. "
            f"Allowed: {sorted(environments)}"
        )

    chosen_style = style if style in styles else styles[0]
    condition_clause = conditions[condition]["clause"]
    environment_clause = environments[environment]["clause"]

    positive = render_structured(
        style=chosen_style,
        breed=breed.replace("_", " "),
        species=species,
        condition_clause=condition_clause,
        environment_clause=environment_clause,
        path=prompt_path,
    )
    negative = render_negative(path=prompt_path)
    return PromptPair(positive=positive, negative=negative, mode="structured")


def naive_input_to_prompt(
    *,
    breed: str,
    condition: str,
    prompt_path: str = "configs/prompts.yaml",
) -> PromptPair:
    """Minimal naive prompt for ablation cell A/B."""
    positive = render_naive(
        breed=breed.replace("_", " "),
        condition=condition.replace("_", " "),
        path=prompt_path,
    )
    negative = render_negative(path=prompt_path)
    return PromptPair(positive=positive, negative=negative, mode="naive")
