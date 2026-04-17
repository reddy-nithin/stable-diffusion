# Controlled Image Generation for AI Animal Care

Educational illustrations of pets in veterinary scenarios using Stable Diffusion 1.5 + ControlNet.

## Scenario

Veterinary clinics and pet-care educators need visual aids that show animals in common care situations (cone collar, grooming, health check, etc.) without relying on expensive photography. This project builds a controlled generation pipeline that takes structured inputs — `animal_type`, `breed`, `condition`, `environment` — and produces consistent, educational illustrations.

## Dataset

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 37 breeds, ~7,400 images with trimap segmentation masks. Used to provide ControlNet conditioning and breed-identity evaluation.

## Quickstart (Google Colab)

```python
# 1. Mount Drive and clone
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/reddy-nithin/stable-diffusion.git
%cd stable-diffusion
!pip install -r requirements.txt

# 2. Download dataset
!python scripts/download_data.py

# 3. Run the demo
!python -m src.app.gradio_app
```

## Pipeline

```
Oxford-IIIT Pet (trimap) ──► seg map ──┐
                                        ├──► SD 1.5 + ControlNet-seg ──► image + metadata.json
taxonomy.yaml + prompts.yaml ──────────┘
```

Ablation matrix (4 cells):

|                     | no-ControlNet | ControlNet-seg |
|---------------------|:---:|:---:|
| **naive prompt**    | A   | B   |
| **structured prompt** | C | **D** ← expected best |

## Evaluation

| Metric | Tool | What it measures |
|--------|------|-----------------|
| CLIPScore | `openai/clip-vit-base-patch32` | prompt-image alignment |
| DINOv2 cosine | `dinov2_vitb14` | breed identity + condition consistency |
| LPIPS | torchmetrics | diversity across seeds |

## Sample Outputs

*(See `samples/` for curated examples)*

## Ethics

All images are AI-generated educational illustrations — not medical references. See `docs/ETHICS.md`.

## Credits

- [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [ControlNet](https://github.com/lllyasviel/ControlNet) by lllyasviel
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — Parkhi et al.
- [diffusers](https://github.com/huggingface/diffusers) by Hugging Face
