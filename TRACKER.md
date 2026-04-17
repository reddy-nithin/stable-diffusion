# Project Tracker

Last session note: Phase 6 complete + Colab environment fully debugged. See COLAB_SETUP below for the definitive working cell sequence. Next: Phase 7 — Deliverables (README, ETHICS, slides).
Active phase: 7

- [x] Phase 0 — Scaffold
- [x] Phase 1 — Data & taxonomy
- [x] Phase 2 — Prompt layer
- [x] Phase 3 — Baseline pipeline
- [x] Phase 4 — ControlNet pipeline
- [x] Phase 5 — Evaluation
- [x] Phase 6 — Gradio demo
- [ ] Phase 7 — Deliverables

---

## COLAB_SETUP — Definitive Working Cell Sequence

All bugs encountered and fixed during Colab bring-up. Use this sequence every session.

### Bugs fixed (all committed to main)
| Error | Root cause | Fix applied |
|-------|-----------|------------|
| `xformers` 38-min install | No binary wheel for Python 3.12 | Changed to `>=0.0.25` |
| `src.data` ModuleNotFoundError | `.gitignore data/` silently excluded `src/data/` | Fixed to `/data/`; committed missing files |
| `cached_download` ImportError | `diffusers==0.27.2` incompatible with `huggingface-hub>=0.25` | Upgraded to `0.30.3` |
| `clear_device_cache` ImportError | `accelerate==0.29.3` too old for Colab's `peft` | **Don't reinstall accelerate** — use Colab's version |
| CUDA version mismatch | `--force-reinstall accelerate` caused torchvision to reinstall with wrong CUDA | **Never reinstall torch/torchvision/accelerate in Colab** |

### KEY RULE
> Only install packages Colab does NOT ship. NEVER reinstall: torch, torchvision, accelerate, peft, transformers, numpy.

### Cell 1 — Install (run once per session, before any imports)
```python
!pip install -q diffusers==0.30.3 controlnet-aux torchmetrics[image] open_clip_torch pyyaml gradio
```

### Cell 2 — Environment setup (run after every restart)
```python
import os, sys
os.chdir('/content/stable-diffusion')
sys.path.insert(0, '/content/stable-diffusion')
!git -C /content/stable-diffusion pull
print("Ready:", os.getcwd())
```

### Cell 3 — Smoke test
```python
from src.app.gradio_app import generate
imgs, pos, neg, ctrl, status = generate(
    animal_type='dog', breed='beagle', condition='cone_collar',
    environment='clinic', style='veterinary illustration',
    use_controlnet=True, seed=42, uploaded_image=None, n_variants=1,
)
print(status)
imgs[0]
```

### Cell 4 — Launch demo
```python
from src.app.gradio_app import build_interface
demo = build_interface(data_root='data')
demo.launch(share=True, show_error=True)
```

### Notes
- First run of Cell 3 downloads SD 1.5 (~4 GB) — takes 2–3 min, cached after that
- Dataset (`/content/stable-diffusion/data/`) is NOT in git (gitignored). Run `!python scripts/download_data.py` once if missing.
- The Oxford dataset survives runtime restarts (saved to Colab disk, not RAM)
- SD model weights are cached in `~/.cache/huggingface/` — also survive restarts
