# Controlled Image Generation — AI Animal Care

## Project one-liner
Turn structured veterinary inputs (animal_type, breed, condition, environment) into educational illustrations using SD 1.5 + ControlNet-seg, evaluated with CLIPScore / DINOv2 / LPIPS.

## Locked decisions
- **Base model:** `stable-diffusion-v1-5/stable-diffusion-v1-5`
- **Control:** ControlNet-seg (`lllyasviel/control_v11p_sd15_seg`) primary; Canny (`lllyasviel/control_v11p_sd15_canny`) fallback
- **Dataset:** Oxford-IIIT Pet (37 breeds, trimaps) via `torchvision.datasets.OxfordIIITPet`
- **Conditions:** educational only — cone collar, bandaged paw, grooming, dental check, vaccination, weight check, post-bath drying, health exam
- **Eval:** CLIPScore + DINOv2 cosine + LPIPS + qualitative success/failure grid
- **Compute:** Google Colab Pro (T4/V100/A100); Mac M2 is dev/editing only
- **Demo:** Gradio inside Colab with `share=True`
- **Scope:** assignment-grade MVP

## Repo map
```
repo/
├── configs/          taxonomy.yaml, prompts.yaml, generation.yaml
├── src/
│   ├── data/         dataset.py, taxonomy.py, masks.py
│   ├── prompts/      templates.py, mapper.py
│   ├── pipelines/    loader.py, baseline.py, controlnet.py
│   ├── generation/   runner.py, io.py, seeds.py
│   ├── evaluation/   clip_score.py, dino_sim.py, lpips_diversity.py, report.py
│   └── app/          gradio_app.py
├── notebooks/        01–05
├── scripts/          download_data.py, generate_batch.py
├── docs/             PROMPT_STRATEGY.md, ETHICS.md, slides_outline.md
├── outputs/          gitignored — images + metadata.json per run
└── samples/          ~12 curated PNGs committed for README
```

## Run commands
```bash
# Colab bootstrap
!git clone <repo_url> && cd stable-diffusion && pip install -r requirements.txt

# Download dataset
python scripts/download_data.py

# Run full 2x2 ablation
python scripts/generate_batch.py --cells all --n-triples 10 --seeds 4

# Evaluate
python -m src.evaluation.report --in outputs --out outputs/eval

# Launch Gradio demo
python -m src.app.gradio_app
```

## Conventions
- **YAML-first config:** change breeds/conditions/environments/params in `configs/` without touching code
- **Metadata sidecars:** every generated image gets a `.json` sidecar (prompt, negative, seed, steps, cfg, model, source image path, hashes)
- **Seeds:** always sourced from `src/generation/seeds.py` for reproducibility
- **Ethics disclaimer:** mandatory on any rendered UI — "AI-generated illustration — not a medical reference"
- **2×2 ablation cells:** A = naive+noCtrl, B = naive+CtrlNet, C = structured+noCtrl, D = structured+CtrlNet

## Tracker
Always read and update `TRACKER.md` at session start and end.
See `PLAN.md` for full phased execution details.

## Do-not-do
- No clinical/graphic imagery, no blood, no gore
- No human medical advice
- No real-person faces
- Do NOT commit `outputs/` or `data/` to git
- Do NOT hardcode model paths — use `configs/generation.yaml`
