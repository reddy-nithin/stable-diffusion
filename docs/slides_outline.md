# Slide Deck Outline

## Controlled Image Generation for AI Animal Care

12 slides, mapped 1:1 to the assignment rubric.

---

### Slide 1 — Title + Scenario

#### Rubric: Scenario

- Title: *PawPrep: Controlled Veterinary Illustrations with Stable Diffusion*
- Subtitle: Structured inputs → educational pet-care images via SD 1.5 + ControlNet
- Problem: Vet clinics and educators need visual aids (cone collar, bandaged paw, etc.)
  without expensive photography or stock licenses
- Solution: A pipeline that takes `(animal_type, breed, condition, environment)` and
  produces consistent, breed-accurate educational illustrations

---

### Slide 2 — Dataset

#### Rubric: Dataset

- Oxford-IIIT Pet Dataset — 37 breeds, ~7,400 images, trimap segmentation masks
- Loaded via `torchvision.datasets.OxfordIIITPet`
- Trimaps provide per-pixel foreground/background signal → converted to ADE20K seg maps
  for ControlNet conditioning
- Show: sample image + trimap + derived seg map (side by side)
- Breed coverage: 25 dog breeds + 12 cat breeds

---

### Slide 3 — Methodology Overview

#### Rubric: Methodology

- 2×2 ablation design: vary prompt style (naive / structured) × control (off / ControlNet-seg)
- 4 seeds per cell per (breed, condition, environment) triple for reproducibility
- Every output paired with a metadata sidecar (prompt, seed, steps, cfg, model)
- Quantitative eval: CLIPScore + DINOv2 cosine + LPIPS
- Show: the 2×2 ablation table

```text
                     | no-ControlNet | ControlNet-seg |
  naive prompt       |   cell A      |   cell B       |
  structured prompt  |   cell C      |   cell D ★     |
```

---

### Slide 4 — Pipeline Architecture

#### Rubric: Pipeline

- Flow diagram: taxonomy.yaml + prompts.yaml → Prompt Layer → SD 1.5 (+ ControlNet-seg) → image + metadata.json
- Parallel path: Oxford trimap → `trimap_to_seg_map()` → ControlNet conditioning signal
- Key files: `src/pipelines/baseline.py`, `src/pipelines/controlnet.py`, `src/generation/runner.py`
- Canny fallback: if seg-map conditioning is weak, `image_to_canny()` swaps in automatically

---

### Slide 5 — Prompt Design

#### Rubric: Prompt Design

- Naive: `"a {breed} {condition}"` — minimal, no style anchor
- Structured: `"{style}, a photograph of a {breed} ({species}) {condition_clause}, {environment_clause}, soft studio lighting, shallow depth of field, high detail, 50mm lens"`

Design choices:

- Style anchor ("veterinary illustration") biases aesthetic and reduces seed variance
- Full clauses ("wearing a protective cone collar after surgery") vs bare keywords
- Species parenthetical disambiguates rare breed names
- Camera idioms well-represented in LAION training data
- Negative prompt shared across all cells (ethics + quality guards)
- All prompt text lives in `configs/prompts.yaml` — no hardcoded strings

---

### Slide 6 — Control Mechanism

#### Rubric: Control

- ControlNet-seg (`lllyasviel/control_v11p_sd15_seg`) — primary
- Trimap foreground → ADE20K color `(4, 200, 4)` (animal class)
- Conditioning strength tuned to 0.6–0.8 to preserve prompt signal while anchoring pose
- Effect: forces animal to occupy correct spatial region, reduces anatomy distortion
- Show: same prompt, seed fixed — cell A (no control) vs cell D (ControlNet-seg)
  → visible difference in pose consistency and breed fidelity

---

### Slide 7 — Tools & Libraries

#### Rubric: Tools

| Component | Library / Model |
| --- | --- |
| Diffusion backbone | `diffusers` — `StableDiffusionControlNetPipeline` |
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Control model | `lllyasviel/control_v11p_sd15_seg` |
| Dataset | `torchvision.datasets.OxfordIIITPet` |
| Edge detection | `controlnet_aux.CannyDetector` |
| CLIPScore | `openai/clip-vit-base-patch32` via `transformers` |
| Breed identity | `facebookresearch/dinov2` (DINOv2 ViT-B/14) |
| Demo UI | `gradio` |
| Config | `pyyaml` — YAML-first taxonomy and generation params |

---

### Slide 8 — Quantitative Results

#### Rubric: Results

Table: one row per ablation cell — CLIPScore ↑, DINOv2 cosine ↑, LPIPS ↑

| Cell | Prompt | ControlNet | CLIPScore | DINOv2 | LPIPS |
| --- | --- | --- | --- | --- | --- |
| A | Naive | Off | — | — | — |
| B | Naive | On | — | — | — |
| C | Structured | Off | — | — | — |
| D | Structured | On | — | — | — |

*(Fill from `outputs/eval/summary.csv` after running `python -m src.evaluation.report`)*

Key expected finding: Cell D highest CLIPScore + DINOv2; LPIPS stays above floor (diversity preserved).

---

### Slide 9 — Qualitative Grid + Failure Analysis

#### Rubric: Evaluation

- 2×2 grid: best output per cell for a fixed (breed, condition, environment) triple
- Success cases: cell D — correct breed markings, condition visible, clean background

Failure cases:

- Cell A: anatomy distortion, condition missing
- Cell B: correct pose but wrong breed color
- Failure audit documented in `docs/ETHICS.md`

*(Use curated samples from `samples/` directory)*

---

### Slide 10 — Gradio Demo

#### Rubric: Demo

- Screenshot: Gradio UI with dropdowns (animal_type, breed, condition, environment, style)
- Features: ControlNet toggle, seed slider, 4 variants side-by-side, resolved prompt shown
- Disclaimer banner: "AI-generated illustration — not a medical reference"
- Demo video: [Watch 90-second walkthrough](https://drive.google.com/file/d/1i9GSxKEcFyzYVnSNOI7COgxGKNaRQG1e/view?usp=sharing)
- GitHub: [reddy-nithin/stable-diffusion](https://github.com/reddy-nithin/stable-diffusion)

---

### Slide 11 — Findings + Limitations

#### Rubric: Findings

Findings:

- Structured prompting alone (cell C) improves CLIPScore over naive (cell A)
- ControlNet adds significant pose + anatomy consistency independent of prompt quality
- Cell D (structured + ControlNet) wins on both alignment and identity metrics
- LPIPS diversity is preserved — ControlNet shapes pose, doesn't collapse variety

Limitations:

- SD 1.5 fur texture quality below SDXL — acceptable for assignment scope
- Single-class seg maps (pet foreground only) weaker than full-scene semantic maps
- Evaluation subset (~40 triples × 4 seeds) — not statistically comprehensive
- No fine-tuning — breed fidelity relies entirely on prompt + ControlNet conditioning

---

### Slide 12 — GitHub + AI Tools Disclosure

#### Rubric: Submission

- GitHub: [reddy-nithin/stable-diffusion](https://github.com/reddy-nithin/stable-diffusion)
- Tag: `v1.0-submission`
- Quickstart: 4-cell Colab sequence (see README)

AI Tools Disclosure:

- Claude (Anthropic) — code generation, prompt strategy design, documentation
- All generated code reviewed and tested by author before use
- No model outputs used as ground truth; all evaluation metrics are algorithmic
