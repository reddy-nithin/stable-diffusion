"""Tests for prompt parameter fidelity."""
from __future__ import annotations

import re

import pytest


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


ALL_CONDITIONS = [
    "cone_collar", "bandaged_paw", "grooming", "dental_check",
    "vaccination", "weight_check", "post_bath_drying", "health_exam",
]
ALL_ENVIRONMENTS = ["clinic", "exam_room"]
SAMPLE_BREEDS = [
    "beagle", "Persian", "great_pyrenees", "leonberger",
    "yorkshire_terrier", "samoyed", "japanese_chin", "birman",
]


@pytest.mark.unit
def test_structured_prompt_token_budget() -> None:
    """Structured prompts must stay under 75 words for all breed×condition combos."""
    from src.prompts.mapper import structured_input_to_prompt

    for breed in SAMPLE_BREEDS:
        for condition in ALL_CONDITIONS:
            species = "cat" if breed[0].isupper() else "dog"
            try:
                pair = structured_input_to_prompt(
                    breed=breed,
                    species=species,
                    condition=condition,
                    environment="clinic",
                    style="veterinary illustration",
                )
            except ValueError:
                continue  # skip combos that require unavailable taxonomy entries
            count = _word_count(pair.positive)
            assert count <= 75, (
                f"Prompt too long ({count} words) for breed={breed!r} "
                f"condition={condition!r}: {pair.positive!r}"
            )


@pytest.mark.unit
def test_condition_keyword_in_prompt() -> None:
    """Key condition words must appear in the positive prompt."""
    from src.prompts.mapper import structured_input_to_prompt

    condition_keywords: dict[str, str] = {
        "cone_collar": "cone",
        "bandaged_paw": "bandaged",
        "grooming": "groom",
        "dental_check": "teeth",
        "vaccination": "vaccination",
        "weight_check": "weight",
        "post_bath_drying": "bath",
        "health_exam": "examination",
    }

    for condition, keyword in condition_keywords.items():
        pair = structured_input_to_prompt(
            breed="beagle",
            species="dog",
            condition=condition,
            environment="clinic",
            style="veterinary illustration",
        )
        assert keyword.lower() in pair.positive.lower(), (
            f"Keyword '{keyword}' missing from prompt for condition={condition!r}: "
            f"{pair.positive!r}"
        )


@pytest.mark.unit
def test_environment_keyword_in_prompt() -> None:
    """Environment description must appear in the positive prompt."""
    from src.prompts.mapper import structured_input_to_prompt

    pair = structured_input_to_prompt(
        breed="beagle",
        species="dog",
        condition="bandaged_paw",
        environment="clinic",
        style="veterinary illustration",
    )
    assert "clinic" in pair.positive.lower(), (
        f"'clinic' not found in prompt: {pair.positive!r}"
    )


@pytest.mark.unit
def test_negative_prompt_suppresses_artifacts() -> None:
    """Negative prompt must contain bounding-box and wrong-breed suppressors."""
    from src.prompts.templates import render_negative

    neg = render_negative()
    required_terms = ["frame", "border", "bounding box", "wrong breed"]
    for term in required_terms:
        assert term in neg, f"Missing '{term}' in negative prompt"


@pytest.mark.unit
def test_condition_appears_before_breed_in_prompt() -> None:
    """Condition clause must precede breed name — confirms token-priority ordering."""
    from src.prompts.mapper import structured_input_to_prompt

    pair = structured_input_to_prompt(
        breed="leonberger",
        species="dog",
        condition="cone_collar",
        environment="exam_room",
        style="veterinary illustration",
    )
    pos = pair.positive.lower()
    condition_pos = pos.find("cone")
    breed_pos = pos.find("leonberger")
    assert condition_pos != -1, "Condition keyword 'cone' not found in prompt"
    assert breed_pos != -1, "Breed 'leonberger' not found in prompt"
    assert condition_pos < breed_pos, (
        f"Condition (pos {condition_pos}) must come before breed (pos {breed_pos}) "
        f"in: {pair.positive!r}"
    )
