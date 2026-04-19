"""PawPrep — Breed-Aware Veterinary Illustrations.

Generates breed-specific veterinary illustrations from structured inputs
(species, breed, condition, environment, style) using SD 1.5 + ControlNet.

Run:
    python -m src.app.gradio_app              # local
    python -m src.app.gradio_app --share      # Colab / public tunnel
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
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


def _make_fallback_control(size: int = 512) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    m = size // 4
    arr[m: size - m, m: size - m] = 255
    return Image.fromarray(arr)


def _get_control_image(
    breed_raw: str,
    uploaded_image: Image.Image | None,
    use_controlnet: bool,
) -> tuple[Image.Image | None, str]:
    """Return (control_image, source_label).

    Priority: uploaded photo (canny) → Oxford breed photo (canny) → fallback.
    Using canny edges from real photos captures breed-specific shape detail
    (ear shape, snout, proportions) far better than generic seg maps.
    """
    if not use_controlnet:
        return None, "none"

    if uploaded_image is not None:
        from src.data.masks import image_to_canny
        ctrl = image_to_canny(uploaded_image.resize((512, 512)))
        return ctrl, "your photo"

    ds = _get_dataset()
    if ds is not None:
        breed_key = breed_raw.replace(" ", "_")
        idx_map = _breed_index()
        if idx_map:
            lookup_key = breed_key if breed_key in idx_map else next(iter(idx_map))
            src_img, _, _, _ = ds[idx_map[lookup_key]]
            from src.data.masks import image_to_canny
            ctrl = image_to_canny(src_img.resize((512, 512)))
            return ctrl, "breed reference"

    from src.data.masks import image_to_canny
    ctrl = image_to_canny(_make_fallback_control())
    return ctrl, "synthetic"


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
    """Generate illustrations and return (images, positive, negative, control_img, status)."""
    from src.pipelines.baseline import run_baseline
    from src.pipelines.controlnet import run_controlnet
    from src.prompts.mapper import structured_input_to_prompt

    species = "cat" if animal_type == "cat" else "dog"
    breed_key = breed.replace(" ", "_")

    prompt_pair = structured_input_to_prompt(
        breed=breed_key,
        species=species,
        condition=condition,
        environment=environment,
        style=style if style else None,
    )

    ctrl_img, ctrl_source = _get_control_image(breed, uploaded_image, use_controlnet)

    images: list[Image.Image] = []
    seeds = [seed + i for i in range(n_variants)]

    for s in seeds:
        if use_controlnet and ctrl_img is not None:
            img, _, _ = run_controlnet(
                prompt_pair,
                source_image=uploaded_image.resize((512, 512)) if uploaded_image else Image.new("RGB", (512, 512)),
                seed=s,
                breed=breed_key,
                species=species,
                condition=condition,
                environment=environment,
                cell="D",
                trimap=None,
                controlnet_type="canny",
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
        images.append(img)

    if use_controlnet:
        ref_label = "your photo" if uploaded_image else "breed reference"
        cn_info = f"shape conditioning active ({ref_label})"
    else:
        cn_info = "no shape conditioning"

    condition_display = condition.replace("_", " ")
    status = f"✓ {n_variants} illustrations generated | {breed}, {condition_display} | {cn_info}"

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
                        label="Use breed reference photo",
                        info="Improves accuracy by using a real Oxford reference image for this breed.",
                    )

                seed = gr.Slider(
                    minimum=0,
                    maximum=9999,
                    step=1,
                    value=42,
                    label="Variation seed",
                    info="Change this number for different looks.",
                )

                with gr.Accordion("📷 Upload your pet's photo (optional)", open=False):
                    uploaded_image = gr.Image(
                        type="pil",
                        label="Your pet's photo",
                        sources=["upload", "webcam"],
                    )
                    gr.Markdown(
                        "_When provided, your pet's own shape guides the illustration "
                        "for a more personalized result. Works best with a clear side-profile shot._"
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

                with gr.Accordion("🔍 Reference image used", open=False):
                    ctrl_img_out = gr.Image(
                        label="Shape reference (edge map)",
                        height=256,
                        elem_id="ctrl-img",
                        interactive=False,
                    )

        def _update_breeds(atype: str):
            breeds = cat_breeds if atype == "cat" else dog_breeds
            return gr.update(choices=breeds, value=breeds[0] if breeds else None)

        animal_type.change(_update_breeds, inputs=animal_type, outputs=breed)

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

        gr.Examples(
            examples=[
                ["dog", "beagle",          "bandaged_paw",    "clinic",         "veterinary illustration",    True,  42,   None],
                ["dog", "great pyrenees",  "cone_collar",     "home",           "veterinary illustration",    True,  100,  None],
                ["cat", "Persian",         "dental_check",    "exam_room",      "educational diagram",        True,  137,  None],
                ["dog", "pug",             "weight_check",    "clinic",         "professional pet photography", False, 2024, None],
                ["dog", "samoyed",         "grooming",        "grooming_salon", "soft watercolor illustration", True,  9999, None],
            ],
            inputs=[animal_type, breed, condition, environment, style,
                    use_controlnet, seed, uploaded_image],
            label="Example scenarios",
            examples_per_page=5,
        )

        gr.Markdown(
            "---\n"
            "_PawPrep uses Stable Diffusion 1.5 with ControlNet Canny conditioning "
            "and the Oxford-IIIT Pet dataset (37 breeds). "
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
