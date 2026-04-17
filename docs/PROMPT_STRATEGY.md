# Prompt Engineering Strategy

## Overview

Two prompt modes are implemented for the 2×2 ablation study:

| Cell | Prompt mode | ControlNet |
|------|-------------|------------|
| A    | Naive       | Off        |
| B    | Naive       | On (seg)   |
| C    | Structured  | Off        |
| D    | Structured  | On (seg)   |

---

## 1. Naive Template

```
a {breed} {condition}
```

Minimal free-text description with no style anchor or spatial guidance.
Used as the ablation baseline to isolate the contribution of (a) structured
prompting and (b) ControlNet conditioning.

---

## 2. Structured Template

```
{style}, a photograph of a {breed} ({species}) {condition_clause},
{environment_clause}, soft studio lighting, shallow depth of field,
high detail, 50mm lens
```

### Slot breakdown

| Slot | Source | Example |
|------|--------|---------|
| `{style}` | `taxonomy.yaml → style_modifiers[0]` | "veterinary illustration" |
| `{breed}` | Oxford breed name (underscores → spaces) | "basset hound" |
| `{species}` | Inferred from breed capitalisation | "dog" |
| `{condition_clause}` | `taxonomy.yaml → conditions.<key>.clause` | "wearing a protective cone collar after surgery" |
| `{environment_clause}` | `taxonomy.yaml → environments.<key>.clause` | "in a bright clean veterinary clinic" |

### Design decisions

**Style anchoring** — Prefixing with a style modifier ("veterinary illustration",
"professional pet photography", etc.) strongly biases the model towards the
desired aesthetic and reduces undesired stylistic variance across seeds.

**Clause library** — Full natural-language clauses (not bare keywords) provide
richer semantic signal. "wearing a protective cone collar after surgery"
communicates pose, prop, and context simultaneously; "cone collar" alone does
not.

**Species parenthetical** — Appending `(dog)` or `(cat)` disambiguates
ambiguous breed names and reinforces the subject identity when the breed name
is uncommon in the model's training corpus.

**Camera idioms** — "soft studio lighting, shallow depth of field, high detail,
50mm lens" are well-represented in LAION training data and reliably increase
photorealism and sharpness without requiring a fine-tuned model.

---

## 3. Negative Prompt

```
cartoon, anime, blurry, lowres, deformed anatomy, extra limbs,
watermark, text, logo, disturbing, graphic injury, blood, gore,
human faces, nsfw, bad proportions, disfigured
```

A single shared negative prompt is used across all four ablation cells so that
any quality difference is attributable to the positive prompt and ControlNet
conditioning, not to negative prompt variation.

Ethics-relevant terms ("graphic injury", "blood", "gore", "human faces",
"nsfw") are included to enforce the project's educational-only scope.

---

## 4. Ethics Guard

`mapper.structured_input_to_prompt` validates `condition` and `environment`
against the allow-list in `configs/taxonomy.yaml` before any generation call.
Unknown keys raise `ValueError` immediately, preventing out-of-scope imagery
from reaching the diffusion model.

---

## 5. YAML-First Principle

All prompt text lives in `configs/prompts.yaml` and `configs/taxonomy.yaml`.
No string literals appear in generation code. This means prompt variants,
clause wording, and style choices can be changed for ablation experiments
without touching Python source.
