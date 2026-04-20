"""CLIP-guided rejection sampling — generate N+extra, return top-N by CLIPScore.

Usage:
    raw = [img1, img2, img3, img4, img5, img6]
    best = top_k_by_clip(raw, prompt="a beagle wearing a cone collar", keep=4)

Falls back silently to returning the first `keep` images if open_clip is unavailable
or scoring fails (e.g. no GPU memory).
"""
from __future__ import annotations

from PIL import Image


def top_k_by_clip(
    images: list[Image.Image],
    prompt: str,
    keep: int,
) -> list[Image.Image]:
    """Return the `keep` images with the highest CLIPScore against `prompt`."""
    if len(images) <= keep:
        return images

    try:
        import torch
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        with torch.no_grad():
            text = tokenizer([prompt]).to(device)
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            scores: list[float] = []
            for img in images:
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                img_features = model.encode_image(img_tensor)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                scores.append((text_features @ img_features.T).item())

        ranked = sorted(zip(scores, images), key=lambda x: x[0], reverse=True)
        return [img for _, img in ranked[:keep]]

    except Exception:
        return images[:keep]
