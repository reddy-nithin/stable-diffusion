"""Colab environment setup — run this as the FIRST cell in any notebook.

Usage (in a Colab notebook cell):
    %run scripts/colab_setup.py

This script upgrades the packages that Colab's base image ships at stale
versions, without touching torch / torchvision / accelerate / numpy which
are tightly coupled to Colab's CUDA build and must not be reinstalled.
"""
import subprocess, sys, os

# ── Package upgrades ──────────────────────────────────────────────────────────
# Must use --upgrade: `pip install X` is a no-op when X is already installed
# at an old pinned version from a previous session.
#
# Known Colab base-image conflicts (as of April 2026):
#   diffusers==0.30.3  →  FLAX_WEIGHTS_NAME removed in newer transformers
#   transformers==4.40 →  EncoderDecoderCache missing (peft needs >=4.41)
#   gradio==4.29.0     →  HfFolder removed from huggingface_hub
#   gradio-client      →  must match gradio version exactly
#
# DO NOT add: torchmetrics, open_clip_torch — they downgrade numpy and break
# controlnet-aux (skimage/scipy compiled against numpy 2.x).
PKGS = [
    "diffusers",
    "transformers",
    "gradio",
    "gradio-client",
    "controlnet-aux",
    "pyyaml",
    "compel",          # long-prompt (>77 token) support for SD 1.5
]

print("Installing/upgrading packages…")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--upgrade", *PKGS
])
print("✓ Packages ready")

# ── Repo root on sys.path ─────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.chdir(REPO_ROOT)
print(f"✓ Working dir: {os.getcwd()}")
print(f"✓ sys.path includes: {REPO_ROOT}")
