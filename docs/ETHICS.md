# Ethics Statement

## AI-Generated Content Disclaimer

All images produced by this pipeline are **AI-generated educational illustrations**.
They are not medical references, clinical diagnoses, or veterinary advice.
Every rendered UI surface in this project displays the disclaimer:

> "AI-generated illustration — not a medical reference."

Users should consult a licensed veterinarian for any health concerns.

---

## Avoided Categories

The following content categories are explicitly out of scope and enforced at multiple layers:

| Category | Enforcement mechanism |
| --- | --- |
| Clinical gore, blood, graphic injury | Negative prompt (`blood, gore, graphic injury, disturbing`) |
| Human faces | Negative prompt (`human faces`) |
| NSFW imagery | Negative prompt (`nsfw`) |
| Non-educational conditions | `mapper.py` validates `condition` against the allow-list in `taxonomy.yaml`; unknown keys raise `ValueError` before any generation call |
| Medical advice | Disclaimer on all UI surfaces; no diagnostic language in any prompt template |

The allow-listed conditions are strictly educational:
`cone_collar`, `bandaged_paw`, `grooming`, `dental_check`, `vaccination`,
`weight_check`, `post_bath_drying`, `health_exam`.

---

## Negative Prompt Ethics Guards

The shared negative prompt applied to all four ablation cells:

```text
cartoon, anime, blurry, lowres, deformed anatomy, extra limbs,
watermark, text, logo, disturbing, graphic injury, blood, gore,
human faces, nsfw, bad proportions, disfigured
```

Ethics-relevant terms are intentionally included alongside quality terms so that
any relaxation of the negative prompt for quality experiments does not
inadvertently remove safety constraints.

---

## Failure Case Audit

Known failure modes observed during generation and their mitigations:

| Failure | Frequency | Root cause | Mitigation |
| --- | --- | --- | --- |
| Anatomical distortion (extra limbs, fused paws) | Occasional (cell A/B) | SD 1.5 anatomy weakness without ControlNet | ControlNet-seg conditioning (cell D) significantly reduces this |
| Over-smooth / painterly fur | Common (all cells) | SD 1.5 tendency toward illustration style | Expected given the "veterinary illustration" style anchor; not a safety issue |
| Breed identity drift (wrong coat colour/markings) | Moderate (cells A/C) | Uncontrolled generation without shape conditioning | ControlNet shape conditioning + DINOv2 breed-identity metric flags these |
| Background bleed (clinic objects as foreground) | Rare | Weak seg-map single-class signal | Canny fallback mode available; logged in TRACKER.md |
| Condition absent from image (cone collar missing) | Rare | Short clause underweighted vs style tokens | Condition clause placed earlier in structured prompt to increase token weight |

No failure cases involving graphic injury, human subjects, or NSFW content were observed.
The negative prompt and taxonomy allow-list guard against these at the input layer.

---

## Dataset License

The Oxford-IIIT Pet Dataset is used under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
Trimap segmentation masks are used solely to derive ControlNet conditioning signals;
no original Oxford images are committed to this repository or included in outputs.

---

## Model Provenance

| Model | Source | License |
| --- | --- | --- |
| Stable Diffusion v1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` on HF Hub | CreativeML Open RAIL-M |
| ControlNet-seg | `lllyasviel/control_v11p_sd15_seg` on HF Hub | Apache 2.0 |
| ControlNet-canny | `lllyasviel/control_v11p_sd15_canny` on HF Hub | Apache 2.0 |
| CLIP (eval) | `openai/clip-vit-base-patch32` on HF Hub | MIT |
| DINOv2 (eval) | `facebookresearch/dinov2` via torch.hub | Apache 2.0 |

All models are accessed via the Hugging Face Hub and are free for research use.
No model weights are committed to this repository.

---

## AI Tools Disclosure

This project was developed with assistance from Claude (Anthropic) for code
generation, prompt strategy design, and documentation drafting. All generated
code was reviewed and tested by the author before use.
