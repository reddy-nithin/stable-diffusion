"""PawPrep — Breed-Aware Veterinary Illustrations.

Generates breed-specific veterinary illustrations from structured inputs
(species, breed, condition, environment, style) using SD 1.5 + ControlNet.

Run:
    python -m src.app.gradio_app              # local
    python -m src.app.gradio_app --share      # Colab / public tunnel
"""
from __future__ import annotations

import argparse
import random
import sys
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image


def _load_taxonomy():
    from src.data.taxonomy import load_taxonomy
    return load_taxonomy()


@lru_cache(maxsize=1)
def _breed_index(data_root: str = "data") -> dict[str, int] | None:
    """Map breed name → first Oxford dataset index for that breed."""
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


# Guidance mode display labels → internal keys
_GUIDANCE_MODES: list[tuple[str, str]] = [
    ("edges (canny)", "canny"),
    ("silhouette (seg)", "seg"),
    ("depth map", "depth"),
]
_GUIDANCE_LABELS = [label for label, _ in _GUIDANCE_MODES]
_LABEL_TO_KEY = {label: key for label, key in _GUIDANCE_MODES}


def _get_control_image(
    breed_raw: str,
    use_controlnet: bool,
    guidance_mode: str = "canny",
) -> tuple[Image.Image | None, str, str]:
    """Return (control_image, source_label, controlnet_type).

    Uploaded photos are no longer used here — they belong to the img2img path.
    Always derives conditioning from the Oxford breed reference so the output
    reflects the selected breed, not the owner's pet silhouette.
    """
    if not use_controlnet:
        return None, "none", "canny"

    ds = _get_dataset()
    if ds is None:
        return None, "no dataset (run download_data.py)", "canny"

    idx_map = _breed_index()
    if not idx_map:
        return None, "no breed index", "canny"

    breed_key = breed_raw.replace(" ", "_")
    lookup_key = breed_key if breed_key in idx_map else next(iter(idx_map))
    src_img, trimap, _, _ = ds[idx_map[lookup_key]]
    src_img = src_img.resize((512, 512), Image.LANCZOS)

    if guidance_mode == "seg":
        from src.data.masks import trimap_to_seg_map
        ctrl = trimap_to_seg_map(trimap)
        return ctrl, "breed trimap → seg map", "seg"

    if guidance_mode == "depth":
        try:
            from src.data.masks import image_to_depth
            ctrl = image_to_depth(src_img)
            return ctrl, "breed reference → depth map", "depth"
        except Exception:
            pass  # MiDaS unavailable — fall through to canny

    from src.data.masks import image_to_canny
    ctrl = image_to_canny(src_img)
    label = "breed reference → canny edges"
    if guidance_mode == "depth":
        label += " (depth unavailable, fell back)"
    return ctrl, label, "canny"


def generate(
    animal_type: str,
    breed: str,
    condition: str,
    environment: str,
    style: str,
    use_controlnet: bool,
    seed: int,
    uploaded_image,
    guidance_label: str = "edges (canny)",
    use_img2img: bool = False,
    n_variants: int = 4,
) -> tuple[list[Image.Image], str, str, Image.Image | None, str]:
    """Generate illustrations and return (images, positive, negative, control_img, status)."""
    from src.generation.rejection import top_k_by_clip
    from src.pipelines.baseline import run_baseline
    from src.pipelines.controlnet import run_controlnet, run_controlnet_img2img
    from src.prompts.mapper import structured_input_to_prompt

    guidance_mode = _LABEL_TO_KEY.get(guidance_label, "canny")
    species = "cat" if animal_type == "cat" else "dog"
    breed_key = breed.replace(" ", "_")

    prompt_pair = structured_input_to_prompt(
        breed=breed_key,
        species=species,
        condition=condition,
        environment=environment,
        style=style if style else None,
    )

    ctrl_img, ctrl_source, cn_type = _get_control_image(breed, use_controlnet, guidance_mode)

    # Generate n_variants + 2 candidates, then keep top n_variants via CLIPScore
    n_candidates = n_variants + 2
    seeds = [seed + i for i in range(n_candidates)]
    raw_images: list[Image.Image] = []

    for s in seeds:
        if use_img2img and uploaded_image is not None and ctrl_img is not None:
            img, _, _ = run_controlnet_img2img(
                prompt_pair,
                init_image=uploaded_image.resize((512, 512)),
                control_image=ctrl_img,
                seed=s,
                breed=breed_key,
                species=species,
                condition=condition,
                environment=environment,
                cell="D",
                controlnet_type=cn_type,
            )
        elif use_controlnet and ctrl_img is not None:
            img, _, _ = run_controlnet(
                prompt_pair,
                source_image=Image.new("RGB", (512, 512)),
                seed=s,
                breed=breed_key,
                species=species,
                condition=condition,
                environment=environment,
                cell="D",
                trimap=None,
                controlnet_type=cn_type,
                control_image=ctrl_img,
            )
        else:
            img, _ = run_baseline(
                prompt_pair,
                seed=s,
                breed=breed_key,
                species=species,
                condition=condition,
                environment=environment,
                cell="C",
            )
        raw_images.append(img)

    images = top_k_by_clip(raw_images, prompt_pair.positive, keep=n_variants)

    if use_img2img and uploaded_image is not None and ctrl_img is not None:
        cn_info = f"img2img + {ctrl_source}"
    elif use_controlnet and ctrl_img is not None:
        cn_info = ctrl_source
    else:
        cn_info = "no shape conditioning"

    condition_display = condition.replace("_", " ")
    status = (
        f"✓ {n_variants} of {n_candidates} illustrations (best by CLIP) | "
        f"{breed}, {condition_display} | {cn_info}"
    )

    return images, prompt_pair.positive, prompt_pair.negative, ctrl_img, status


def build_interface(data_root: str = "data") -> "gr.Blocks":
    import gradio as gr

    tax = _load_taxonomy()
    cat_breeds, dog_breeds = _build_breed_lists(tax)
    all_conditions   = _condition_list(tax)
    all_environments = _environment_list(tax)
    all_styles       = _style_list(tax)

    DISCLAIMER = (
        "⚠️ **AI-generated illustrations — not a medical or veterinary reference.** "
        "For educational and communication purposes only. "
        "Always consult a licensed veterinarian for animal health advice."
    )

    with gr.Blocks(
        title="PawPrep — Breed-Aware Veterinary Illustrations",
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
            #how-it-helps {
                background: #111c14;
                border: 1px solid #2a4a2a;
                border-radius: 8px;
                padding: 14px 18px;
                margin-bottom: 10px;
                font-size: 0.88rem;
                color: #b8d4b8;
            }
            #breed-warning {
                background: #2a1a08;
                border: 1px solid #7a4e1a;
                border-radius: 6px;
                padding: 8px 12px;
                color: #e8c07a;
                font-size: 0.82rem;
                margin-top: -2px;
                margin-bottom: 4px;
            }
            footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown(
            "# 🐾 PawPrep — Breed-Aware Veterinary Illustrations",
            elem_id="title-row",
        )
        gr.Markdown(
            "_Helping pet owners and clinics visualize veterinary care — "
            "tailored to your pet's exact breed._",
            elem_id="title-row",
        )
        gr.Markdown(DISCLAIMER, elem_id="disclaimer")

        gr.Markdown(
            """<div id="how-it-helps">
            <strong>How PawPrep helps:</strong> &nbsp;
            🏥 <em>Clinics</em> — generate breed-specific client education materials before procedures. &nbsp;
            🐾 <em>Pet owners</em> — see what your pet will look like during or after treatment, reducing anxiety. &nbsp;
            ✂️ <em>Groomers</em> — preview styling and wellness visits for any breed.
            </div>"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Customize Illustration")

                animal_type = gr.Radio(
                    ["cat", "dog"],
                    value="dog",
                    label="Species",
                )
                breed = gr.Dropdown(
                    choices=dog_breeds,
                    value=dog_breeds[0] if dog_breeds else None,
                    label="Breed",
                )
                breed_upload_warning = gr.Markdown(
                    "",
                    visible=False,
                    elem_id="breed-warning",
                )
                condition = gr.Dropdown(
                    choices=all_conditions,
                    value=all_conditions[0] if all_conditions else None,
                    label="Veterinary situation",
                )
                environment = gr.Dropdown(
                    choices=all_environments,
                    value=all_environments[0] if all_environments else None,
                    label="Setting",
                )
                style = gr.Dropdown(
                    choices=all_styles,
                    value=all_styles[0] if all_styles else None,
                    label="Illustration style",
                )

                with gr.Row():
                    use_controlnet = gr.Checkbox(
                        value=True,
                        label="Use breed shape reference",
                        info="Guides pose and proportions from Oxford breed photos.",
                    )

                guidance_mode_radio = gr.Radio(
                    choices=_GUIDANCE_LABELS,
                    value=_GUIDANCE_LABELS[0],
                    label="Shape guidance type",
                    info="How to extract the breed shape signal.",
                    visible=True,
                )

                with gr.Row():
                    seed = gr.Slider(
                        minimum=0,
                        maximum=9999,
                        step=1,
                        value=42,
                        label="Variation seed",
                        info="Change for different looks.",
                    )
                    shuffle_btn = gr.Button("🎲", scale=0, min_width=48)

                with gr.Accordion("📷 Upload your pet's photo (optional)", open=False):
                    uploaded_image = gr.Image(
                        type="pil",
                        label="Your pet's photo",
                        sources=["upload", "webcam"],
                    )
                    use_img2img = gr.Checkbox(
                        value=False,
                        label="Restyle my photo (img2img mode)",
                        info=(
                            "When checked, your photo becomes the starting point "
                            "and gets restyled into the veterinary scenario. "
                            "Breed conditioning still comes from the dropdown."
                        ),
                    )
                    gr.Markdown(
                        "**Without 'Restyle my photo':** your photo is ignored — "
                        "breed shape comes from the Oxford reference library.\n\n"
                        "**With 'Restyle my photo':** your pet's pose and structure "
                        "are preserved while the scene and style are changed."
                    )

                generate_btn = gr.Button(
                    "✨ Generate Illustrations",
                    variant="primary",
                    elem_classes=["generate-btn"],
                )

            with gr.Column(scale=2):
                gr.Markdown("### Your Illustrations")

                gallery = gr.Gallery(
                    label="Generated illustrations",
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

                with gr.Accordion("📝 Prompt details", open=False):
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

                with gr.Accordion("🔍 Shape reference used", open=False):
                    ctrl_img_out = gr.Image(
                        label="Shape conditioning image",
                        height=256,
                        elem_id="ctrl-img",
                        interactive=False,
                    )

        def _update_breeds(atype: str):
            breeds = cat_breeds if atype == "cat" else dog_breeds
            return gr.update(choices=breeds, value=breeds[0] if breeds else None)

        def _on_upload_change(img):
            if img is not None:
                return gr.update(
                    value=(
                        "📸 **Photo uploaded** — enable 'Restyle my photo' below to use it as "
                        "the generation starting point. Without that checkbox, the photo is ignored."
                    ),
                    visible=True,
                )
            return gr.update(value="", visible=False)

        def _on_use_controlnet_change(val: bool):
            return gr.update(visible=val)

        def _shuffle_seed():
            return gr.update(value=random.randint(0, 9999))

        animal_type.change(_update_breeds, inputs=animal_type, outputs=breed)
        uploaded_image.change(_on_upload_change, inputs=uploaded_image, outputs=breed_upload_warning)
        use_controlnet.change(_on_use_controlnet_change, inputs=use_controlnet, outputs=guidance_mode_radio)
        shuffle_btn.click(_shuffle_seed, outputs=seed)

        def _generate_wrapper(atype, br, cond, env, sty, use_cn, s, upload, g_label, img2img):
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
                    guidance_label=g_label,
                    use_img2img=img2img,
                )
                return imgs, status, pos, neg, ctrl
            except Exception as e:
                import traceback
                return [], f"❌ Error: {e}", traceback.format_exc(), "", None

        generate_btn.click(
            fn=_generate_wrapper,
            inputs=[animal_type, breed, condition, environment, style,
                    use_controlnet, seed, uploaded_image,
                    guidance_mode_radio, use_img2img],
            outputs=[gallery, status_box, positive_out, negative_out, ctrl_img_out],
        )

        gr.Examples(
            examples=[
                ["dog", "beagle",         "bandaged_paw",    "clinic",         "veterinary illustration",     True,  42,   None, "edges (canny)",     False],
                ["dog", "great pyrenees", "cone_collar",     "home",           "veterinary illustration",     True,  100,  None, "silhouette (seg)",  False],
                ["cat", "Persian",        "dental_check",    "exam_room",      "educational diagram",         True,  137,  None, "edges (canny)",     False],
                ["dog", "pug",            "weight_check",    "clinic",         "professional pet photography", False, 2024, None, "edges (canny)",     False],
                ["dog", "samoyed",        "grooming",        "grooming_salon", "soft watercolor illustration", True,  9999, None, "depth map",         False],
            ],
            inputs=[animal_type, breed, condition, environment, style,
                    use_controlnet, seed, uploaded_image,
                    guidance_mode_radio, use_img2img],
            label="Example scenarios",
            examples_per_page=5,
        )

        gr.Markdown(
            "---\n"
            "_PawPrep uses Stable Diffusion 1.5 with ControlNet (Canny / Seg / Depth) "
            "and the Oxford-IIIT Pet dataset (37 breeds). "
            "Best-of-6 selection via CLIPScore. "
            "Evaluation: CLIPScore · DINOv2 · LPIPS. "
            "Not a clinical reference._"
        )

    return demo


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch PawPrep.")
    p.add_argument("--share",     action="store_true", help="Enable public Gradio tunnel (required for Colab).")
    p.add_argument("--port",      type=int, default=7860)
    p.add_argument("--data-root", default="data")
    return p.parse_args()


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    args = _parse()
    print("Building PawPrep interface…")
    demo = build_interface(data_root=args.data_root)
    print(f"Launching on port {args.port} (share={args.share})…")
    demo.launch(
        share=args.share,
        server_port=args.port,
        show_error=True,
    )
