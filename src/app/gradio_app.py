"""Gradio demo — Controlled Image Generation for AI Animal Care.

UI layout
---------
Top: ethics disclaimer banner (always visible).

Left column — inputs:
  • Animal type (cat | dog) → filters breed dropdown
  • Breed dropdown
  • Condition dropdown
  • Environment dropdown
  • Style modifier dropdown
  • ControlNet toggle checkbox
  • Seed slider (0–9999)
  • Optional: upload your own pet image (enables canny override)
  • Generate button

Right column — outputs:
  • 4 generated images (one per seed variant), displayed as a gallery
  • Resolved positive prompt (textbox, read-only)
  • Resolved negative prompt (textbox, read-only, collapsed)
  • Control conditioning image used (shown when ControlNet is on)

Conditioning logic (Option C):
  - If user uploads an image  → Canny edges from that image
  - Else if Oxford dataset available → seg map from breed's trimap
  - Else → Canny from a synthetic placeholder (white rectangle on black)

Run
---
    python -m src.app.gradio_app              # local
    python -m src.app.gradio_app --share      # Colab / public tunnel
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy imports (heavy deps only loaded when generate is called)
# ---------------------------------------------------------------------------

def _load_taxonomy():
    from src.data.taxonomy import load_taxonomy
    return load_taxonomy()


@lru_cache(maxsize=1)
def _breed_index(data_root: str = "data") -> dict[str, int] | None:
    """Map breed name → first Oxford dataset index. Returns None if no dataset."""
    try:
        from src.data.dataset import OxfordPetDataset
        ds = OxfordPetDataset(root=data_root)
        idx: dict[str, int] = {}
        for i in range(len(ds)):
            _, _, breed, _ = ds[i]
            if breed not in idx:
                idx[breed] = i
            if len(idx) == 37:
                break
        return idx
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_dataset(data_root: str = "data"):
    try:
        from src.data.dataset import OxfordPetDataset
        return OxfordPetDataset(root=data_root)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Breed / condition / environment lists
# ---------------------------------------------------------------------------

def _build_breed_lists(tax: dict) -> tuple[list[str], list[str]]:
    cats = [b.replace("_", " ") for b in tax.get("cat_breeds", [])]
    dogs = [b.replace("_", " ") for b in tax.get("dog_breeds", [])]
    return sorted(cats), sorted(dogs)


def _condition_list(tax: dict) -> list[str]:
    return list(tax.get("conditions", {}).keys())


def _environment_list(tax: dict) -> list[str]:
    return list(tax.get("environments", {}).keys())


def _style_list(tax: dict) -> list[str]:
    return tax.get("style_modifiers", ["veterinary illustration"])


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def _make_fallback_control(size: int = 512) -> Image.Image:
    """Black image with a white rectangle — a last-resort control image."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    m = size // 4
    arr[m: size - m, m: size - m] = 255
    return Image.fromarray(arr)


def _get_control_image(
    breed_raw: str,
    uploaded_image: Image.Image | None,
    use_controlnet: bool,
) -> tuple[Image.Image | None, str]:
    """Return (control_image, modality_used).

    Priority: uploaded → Oxford seg → fallback canny.
    modality_used: 'canny' | 'seg' | 'none'
    """
    if not use_controlnet:
        return None, "none"

    if uploaded_image is not None:
        from src.data.masks import image_to_canny
        ctrl = image_to_canny(uploaded_image.resize((512, 512)))
        return ctrl, "canny"

    # Try Oxford dataset
    ds = _get_dataset()
    if ds is not None:
        breed_key = breed_raw.replace(" ", "_")
        idx_map = _breed_index()
        if idx_map and breed_key in idx_map:
            src_img, trimap, _, _ = ds[idx_map[breed_key]]
            from src.data.masks import trimap_to_seg_map
            ctrl = trimap_to_seg_map(trimap)
            return ctrl, "seg"
        # breed not in index — fall back to canny on src_img
        try:
            first_idx = next(iter(idx_map.values())) if idx_map else 0
            src_img, _, _, _ = ds[first_idx]
            from src.data.masks import image_to_canny
            ctrl = image_to_canny(src_img)
            return ctrl, "canny (fallback breed)"
        except Exception:
            pass

    # No dataset at all — synthetic fallback
    from src.data.masks import image_to_canny
    ctrl = image_to_canny(_make_fallback_control())
    return ctrl, "canny (no dataset)"


def generate(
    animal_type: str,
    breed: str,
    condition: str,
    environment: str,
    style: str,
    use_controlnet: bool,
    seed: int,
    uploaded_image,
    n_variants: int = 4,
) -> tuple[list[Image.Image], str, str, Image.Image | None, str]:
    """Core generation function wired to the Gradio interface.

    Returns
    -------
    (gallery_images, positive_prompt, negative_prompt, control_image, status)
    """
    from src.pipelines.baseline import run_baseline
    from src.pipelines.controlnet import run_controlnet
    from src.prompts.mapper import naive_input_to_prompt, structured_input_to_prompt

    breed_raw = breed
    species = "cat" if animal_type == "cat" else "dog"
    breed_key = breed.replace(" ", "_")

    # Build prompts
    struct_pp = structured_input_to_prompt(
        breed=breed_key,
        species=species,
        condition=condition,
        environment=environment,
        style=style if style else None,
    )

    # Control image
    ctrl_img, modality = _get_control_image(breed_raw, uploaded_image, use_controlnet)

    # Generate N variants with consecutive seeds
    images: list[Image.Image] = []
    seeds = [seed + i for i in range(n_variants)]

    for s in seeds:
        if use_controlnet and ctrl_img is not None:
            img, _, _ = run_controlnet(
                struct_pp,
                source_image=uploaded_image.resize((512, 512)) if uploaded_image else Image.new("RGB", (512, 512)),
                seed=s,
                breed=breed_key,
                species=species,
                condition=condition,
                environment=environment,
                cell="D",
                trimap=None,
                controlnet_type="canny" if "canny" in modality else "seg",
            )
        else:
            img, _ = run_baseline(
                struct_pp,
                seed=s,
                breed=breed_key,
                species=species,
                condition=condition,
                environment=environment,
                cell="C",
            )
        images.append(img)

    cn_info = f"ControlNet ON ({modality})" if use_controlnet else "ControlNet OFF"
    status = f"✓ Generated {n_variants} variants | {cn_info} | seeds {seeds[0]}–{seeds[-1]}"

    return images, struct_pp.positive, struct_pp.negative, ctrl_img, status


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_interface(data_root: str = "data") -> "gr.Blocks":
    import gradio as gr

    tax = _load_taxonomy()
    cat_breeds, dog_breeds = _build_breed_lists(tax)
    all_conditions   = _condition_list(tax)
    all_environments = _environment_list(tax)
    all_styles       = _style_list(tax)

    DISCLAIMER = (
        "⚠️ **AI-generated illustrations — NOT a medical or veterinary reference.** "
        "Images are for educational demonstration purposes only. "
        "Always consult a licensed veterinarian for animal health advice."
    )

    with gr.Blocks(
        title="AI Animal Care — Controlled Image Generator",
        theme=gr.themes.Base(
            primary_hue="emerald",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
            #disclaimer {
                background: linear-gradient(90deg, #1a3a1a, #1a2a1a);
                border: 1px solid #2e6b2e;
                border-radius: 8px;
                padding: 12px 18px;
                color: #a0d4a0;
                font-size: 0.9rem;
                margin-bottom: 12px;
            }
            #title-row { text-align: center; margin-bottom: 4px; }
            .generate-btn { background: linear-gradient(135deg, #1a7a40, #0e5a30) !important; }
            #ctrl-img { border: 2px solid #2e6b2e; border-radius: 8px; }
            footer { display: none !important; }
        """,
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.Markdown(
            "# 🐾 AI Animal Care — Controlled Image Generator",
            elem_id="title-row",
        )
        gr.Markdown(DISCLAIMER, elem_id="disclaimer")

        with gr.Row():
            # ── Left: inputs ─────────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Controls")

                animal_type = gr.Radio(
                    ["cat", "dog"],
                    value="dog",
                    label="Animal type",
                    info="Filters the breed list below.",
                )
                breed = gr.Dropdown(
                    choices=dog_breeds,
                    value=dog_breeds[0] if dog_breeds else None,
                    label="Breed",
                )
                condition = gr.Dropdown(
                    choices=all_conditions,
                    value=all_conditions[0] if all_conditions else None,
                    label="Veterinary condition",
                )
                environment = gr.Dropdown(
                    choices=all_environments,
                    value=all_environments[0] if all_environments else None,
                    label="Environment",
                )
                style = gr.Dropdown(
                    choices=all_styles,
                    value=all_styles[0] if all_styles else None,
                    label="Style modifier",
                )

                with gr.Row():
                    use_controlnet = gr.Checkbox(
                        value=True,
                        label="Enable ControlNet",
                        info="Shape conditioning from Oxford ref (seg) or your upload (canny).",
                    )

                seed = gr.Slider(
                    minimum=0,
                    maximum=9999,
                    step=1,
                    value=42,
                    label="Base seed",
                    info="4 variants generated at seed, seed+1, seed+2, seed+3.",
                )

                with gr.Accordion("📷 Upload your own pet photo (optional)", open=False):
                    uploaded_image = gr.Image(
                        type="pil",
                        label="Your pet photo",
                        sources=["upload", "webcam"],
                    )
                    gr.Markdown(
                        "_If provided, Canny edges from your photo are used as ControlNet conditioning "
                        "instead of the Oxford dataset reference. Works best with a clean side-profile shot._"
                    )

                generate_btn = gr.Button(
                    "✨ Generate 4 Variants",
                    variant="primary",
                    elem_classes=["generate-btn"],
                )

            # ── Right: outputs ───────────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Generated Illustrations")

                gallery = gr.Gallery(
                    label="Variations (4 seeds)",
                    columns=2,
                    rows=2,
                    height=560,
                    object_fit="contain",
                    show_label=True,
                )

                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=1,
                )

                with gr.Accordion("📝 Resolved prompts", open=False):
                    positive_out = gr.Textbox(
                        label="Positive prompt",
                        interactive=False,
                        lines=3,
                    )
                    negative_out = gr.Textbox(
                        label="Negative prompt",
                        interactive=False,
                        lines=2,
                    )

                with gr.Accordion("🎯 ControlNet conditioning image", open=False):
                    ctrl_img_out = gr.Image(
                        label="Control image used",
                        height=256,
                        elem_id="ctrl-img",
                        interactive=False,
                    )

        # ── Dynamic breed filter ─────────────────────────────────────────────
        def _update_breeds(atype: str):
            breeds = cat_breeds if atype == "cat" else dog_breeds
            return gr.update(choices=breeds, value=breeds[0] if breeds else None)

        animal_type.change(_update_breeds, inputs=animal_type, outputs=breed)

        # ── Generate ─────────────────────────────────────────────────────────
        def _generate_wrapper(atype, br, cond, env, sty, use_cn, s, upload):
            try:
                imgs, pos, neg, ctrl, status = generate(
                    animal_type=atype,
                    breed=br,
                    condition=cond,
                    environment=env,
                    style=sty,
                    use_controlnet=use_cn,
                    seed=int(s),
                    uploaded_image=upload,
                )
                return imgs, status, pos, neg, ctrl
            except Exception as e:
                import traceback
                return [], f"❌ Error: {e}", traceback.format_exc(), "", None

        generate_btn.click(
            fn=_generate_wrapper,
            inputs=[animal_type, breed, condition, environment, style,
                    use_controlnet, seed, uploaded_image],
            outputs=[gallery, status_box, positive_out, negative_out, ctrl_img_out],
        )

        # ── Examples ─────────────────────────────────────────────────────────
        gr.Examples(
            examples=[
                ["dog", "beagle",       "cone_collar",    "clinic",         "veterinary illustration", True,  42,  None],
                ["cat", "Siamese",      "health_exam",    "exam_room",      "educational diagram",     True,  137, None],
                ["dog", "pug",          "weight_check",   "clinic",         "professional pet photography", False, 2024, None],
                ["cat", "Persian",      "grooming",       "grooming salon", "soft watercolor illustration", True, 9999, None],
                ["dog", "golden retriever", "bandaged_paw", "home",         "veterinary illustration", True,  42,  None],
            ],
            inputs=[animal_type, breed, condition, environment, style,
                    use_controlnet, seed, uploaded_image],
            label="Quick examples",
            examples_per_page=5,
        )

        # ── Footer ───────────────────────────────────────────────────────────
        gr.Markdown(
            "---\n"
            "_Built with SD 1.5 + ControlNet-seg/canny | Oxford-IIIT Pet dataset | "
            "CLIPScore · DINOv2 · LPIPS evaluation pipeline_"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch the Gradio demo.")
    p.add_argument("--share",     action="store_true", help="Enable Gradio public tunnel (required for Colab).")
    p.add_argument("--port",      type=int, default=7860, help="Local port.")
    p.add_argument("--data-root", default="data", help="Oxford dataset root dir.")
    return p.parse_args()


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    args = _parse()
    print("Building Gradio interface…")
    demo = build_interface(data_root=args.data_root)
    print(f"Launching on port {args.port} (share={args.share})…")
    demo.launch(
        share=args.share,
        server_port=args.port,
        show_error=True,
    )
