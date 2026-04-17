"""CLIPScore evaluation — prompt-image alignment.

Computes CLIP cosine similarity between a prompt string and a generated
image using openai/clip-vit-base-patch32.  Higher = better prompt alignment.

Reference: Hessel et al. 2021, "CLIPScore: A Reference-free Evaluation
Metric for Image Captioning" (arXiv 2104.08718).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_MODEL_ID = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=1)
def _load_clip(device: str):
    model = CLIPModel.from_pretrained(_MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(_MODEL_ID)
    return model, processor


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def clip_score(image: Image.Image, prompt: str) -> float:
    """Return CLIP cosine similarity ∈ [-1, 1] for one (image, prompt) pair.

    In practice scores cluster between 0.20 – 0.40 for realistic captions.
    """
    dev = _device()
    model, processor = _load_clip(dev)

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    out = model(**inputs)
    img_emb  = F.normalize(out.image_embeds, dim=-1)   # (1, D)
    txt_emb  = F.normalize(out.text_embeds,  dim=-1)   # (1, D)
    score    = (img_emb * txt_emb).sum().item()
    return float(score)


@torch.no_grad()
def clip_scores_from_dir(
    output_dir: str | Path,
    prompt_key: str = "prompt_positive",
) -> list[dict]:
    """Score every PNG in output_dir using its sidecar JSON for the prompt.

    Returns a list of dicts: {stem, clip_score, cell, breed, condition, environment}.
    """
    import json

    out = Path(output_dir)
    records = []
    for img_path in sorted(out.glob("*.png")):
        sidecar = img_path.with_suffix(".json")
        if not sidecar.exists():
            continue
        meta = json.loads(sidecar.read_text())
        prompt = meta.get(prompt_key, "")
        image  = Image.open(img_path).convert("RGB")
        score  = clip_score(image, prompt)
        records.append({
            "stem":        img_path.stem,
            "cell":        meta.get("cell", "?"),
            "breed":       meta.get("breed", ""),
            "condition":   meta.get("condition", ""),
            "environment": meta.get("environment", ""),
            "seed":        meta.get("seed", -1),
            "clip_score":  score,
        })
    return records
