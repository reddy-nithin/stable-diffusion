# Project Submission — PawPrep: Controlled Veterinary Illustrations

**Course project | Controlled Image Generation with Stable Diffusion**
**GitHub:** https://github.com/reddy-nithin/stable-diffusion
**Submission tag:** `v1.0-submission`

---

## Slide Deck (10 Slides)

[Open presentation](https://docs.google.com/presentation/d/1eTAOF43F6RIJ8Q4Kza-L65NIxGqlOG0wkIfOhxIEcK8/edit)

| Slide | Rubric item covered |
| --- | --- |
| 1 | Scenario description |
| 2 | Dataset |
| 3 | Methodology (2×2 ablation design) |
| 4 | Pipeline (Stable Diffusion + ControlNet) |
| 5 | Prompt design |
| 6 | Control strategy |
| 7 | Tools & technologies |
| 8 | Results + Evaluation (real metric values) |
| 9 | Demo video link + GitHub link |
| 10 | Findings, limitations, AI tools disclosure |

---

## Demo Video (90 seconds)

[Watch demo](https://drive.google.com/file/d/1i9GSxKEcFyzYVnSNOI7COgxGKNaRQG1e/view?usp=sharing)

Covers:
- System overview — PawPrep UI walkthrough
- Input → prompt → output: Shiba Inu + grooming + ControlNet ON
- Comparison: same breed/seed with ControlNet OFF vs ON

---

## GitHub Repository

https://github.com/reddy-nithin/stable-diffusion

| Requirement | Location |
| --- | --- |
| Code | `src/` — pipelines, prompts, evaluation, Gradio app |
| README | `README.md` — quickstart, pipeline diagram, sample grid |
| Dataset description | `README.md` → Dataset section; `configs/taxonomy.yaml` |
| Prompt strategy | `docs/PROMPT_STRATEGY.md` |
| Sample outputs | `samples/` — 8 curated PNGs (7× cell D + 1× cell A for comparison) |
| Tools & libraries | `README.md` → Evaluation table; `requirements.txt` |

---

## Working Pipeline

### Scenario
Veterinary clinics and pet-care educators need visual aids for common care situations
(cone collar, bandaged paw, grooming, dental check, etc.) without expensive photography.
PawPrep takes structured inputs — `animal_type`, `breed`, `condition`, `environment` —
and produces consistent, breed-accurate educational illustrations.

### Dataset
Oxford-IIIT Pet Dataset — 37 breeds (~7,400 images) with trimap segmentation masks.
Loaded via `torchvision.datasets.OxfordIIITPet`.
Trimaps converted to ADE20K seg maps for ControlNet conditioning.

### Pipeline
```
Oxford trimap → seg map / canny edges ──┐
                                         ├── SD 1.5 + ControlNet → image + metadata.json
taxonomy.yaml + prompts.yaml ────────────┘
```

Ablation cells:
- **A** — naive prompt, no ControlNet (baseline)
- **B** — naive prompt + ControlNet
- **C** — structured prompt, no ControlNet
- **D** — structured prompt + ControlNet

Key files: `src/pipelines/baseline.py`, `src/pipelines/controlnet.py`, `src/generation/runner.py`

---

## Controlled Generation

**Control mechanism:** ControlNet-seg (`lllyasviel/control_v11p_sd15_seg`)

Trimap foreground → ADE20K animal class color `(4, 200, 4)` → ControlNet conditioning signal.
Conditioning strength 0.55, guidance end 0.6 — locks pose in early diffusion steps,
releases in later steps so prompt drives breed details.

Canny fallback (`lllyasviel/control_v11p_sd15_canny`) available when seg-map signal is weak.

Full write-up: `docs/PROMPT_STRATEGY.md` → Section 4 (Ethics Guard) and `src/pipelines/controlnet.py`

---

## Evaluation and Analysis

### Metrics

| Metric | Model | Measures |
| --- | --- | --- |
| CLIPScore | `openai/clip-vit-base-patch32` | Prompt–image alignment |
| Breed Identity | `facebookresearch/dinov2` (DINOv2 ViT-B/14) | Visual breed fidelity vs Oxford reference |
| Condition Consistency | DINOv2 cosine across seeds | Semantic stability |
| LPIPS Diversity | `torchmetrics` | Output variety across seeds |

Evaluation code: `src/evaluation/` | Full run: `notebooks/04_evaluation.ipynb`

### Results

| Cell | Prompt | ControlNet | CLIPScore ↑ | Breed Identity ↑ | Consistency ↑ | LPIPS Diversity ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| A | Naive | Off | 0.2960 ± 0.0285 | 0.5044 | 0.5926 | 0.6676 |
| B | Naive | On | 0.2966 ± 0.0202 | **0.6023** | **0.7419** | 0.5294 |
| C | Structured | Off | 0.3016 ± 0.0283 | 0.3080 | 0.5513 | **0.6864** |
| D | Structured | On | **0.3069** ± 0.0220 | 0.4375 | 0.6847 | 0.5444 |

---

## Findings and Insights

### Prompt effectiveness (C vs A)
Structured prompts improve CLIPScore (+0.0056, +1.9%) — detailed slot-filled prompts
with style anchor and condition clauses produce better text-image alignment.
However, structured prompting *alone* reduces breed identity (0.3080 vs 0.5044) —
detailed style/condition clauses dilute the breed signal without shape conditioning to anchor it.

### Control vs diversity trade-off (B vs A, D vs C)
ControlNet dramatically improves breed identity (B: +0.0979 over A) and condition
consistency (B: +0.1493 over A). Trade-off: diversity drops (LPIPS 0.67 → 0.53),
as shape conditioning anchors the pose and reduces seed variance.
All cells remain above LPIPS 0.52 — no mode collapse observed.

### Diffusion limitations
- SD 1.5 fur texture quality below SDXL — over-smooth coat rendering across all cells
- Single-class seg maps (foreground only) weaker than full-scene semantic conditioning
- Rare conditions (dental check, vaccination) underrepresented in SD 1.5 training data — condition detail sometimes absent from output

### Success and failure cases
Success: `samples/07_shiba_grooming_D.png` — correct orange/white Shiba coat, groomer's hand clearly visible.
Failure comparison: `samples/02_samoyed_cone_clinic_A.png` (Cell A) vs `samples/01_samoyed_cone_clinic_D.png` (Cell D) — same seed, A shows no cone and loose anatomy.

Full failure audit: `docs/ETHICS.md` → Failure Case Audit section

---

## AI Tools Disclosure

| Tool | Use |
| --- | --- |
| Claude (Anthropic) | Code generation, prompt strategy design, documentation drafting |
| Stable Diffusion v1.5 | Image generation backbone — CreativeML Open RAIL-M license |
| ControlNet | Shape conditioning — Apache 2.0 license |
| Oxford-IIIT Pet Dataset | Training reference and ControlNet conditioning — CC BY 4.0 |
| DINOv2 | Breed identity evaluation — Apache 2.0 |
| CLIP | Prompt-image alignment evaluation — MIT |

All generated code was reviewed and tested by the author before use.
No model outputs were used as ground truth — all evaluation metrics are algorithmic.

Full statement: `docs/ETHICS.md`

---

## Quick Reference

| Deliverable | Link / Location |
| --- | --- |
| Slide deck | [Google Slides](https://docs.google.com/presentation/d/1eTAOF43F6RIJ8Q4Kza-L65NIxGqlOG0wkIfOhxIEcK8/edit) |
| Demo video | [Google Drive](https://drive.google.com/file/d/1i9GSxKEcFyzYVnSNOI7COgxGKNaRQG1e/view?usp=sharing) |
| GitHub repo | [reddy-nithin/stable-diffusion](https://github.com/reddy-nithin/stable-diffusion) |
| Sample outputs | `samples/` in repo |
| Evaluation notebook | `notebooks/04_evaluation.ipynb` |
| Prompt strategy | `docs/PROMPT_STRATEGY.md` |
| Ethics statement | `docs/ETHICS.md` |
| Colab quickstart | `README.md` → Quickstart section |
