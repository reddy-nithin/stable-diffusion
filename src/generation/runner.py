"""Full 2×2 ablation runner.

Cells
-----
A  naive  prompt  +  no ControlNet   → outputs/A/
B  naive  prompt  +  ControlNet      → outputs/B/
C  structured prompt + no ControlNet → outputs/C/
D  structured prompt + ControlNet    → outputs/D/  ← expected best

Usage
-----
    from src.generation.runner import run_ablation
    run_ablation(cells=["A", "B", "C", "D"], n_triples=10, seeds=None)

Or via scripts/generate_batch.py (CLI wrapper).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import yaml
from tqdm.auto import tqdm

from src.data.dataset import OxfordPetDataset
from src.data.taxonomy import (
    all_conditions,
    all_environments,
    all_styles,
    load_taxonomy,
    species_for_breed,
)
from src.generation.io import save_image_with_sidecar
from src.generation.seeds import get_seeds
from src.pipelines.baseline import run_baseline
from src.pipelines.controlnet import run_controlnet
from src.prompts.mapper import naive_input_to_prompt, structured_input_to_prompt


# ---------------------------------------------------------------------------
# Triple building
# ---------------------------------------------------------------------------

def _build_triples(
    n: int,
    dataset: OxfordPetDataset,
    tax: dict,
    rng_seed: int = 0,
) -> list[dict]:
    """Sample n (dataset_idx, breed, species, condition, environment) dicts.

    Samples are drawn so breed × condition × environment are varied.
    Uses a fixed RNG seed for reproducibility across runs.
    """
    rng = random.Random(rng_seed)
    conds = list(all_conditions(tax).keys())
    envs  = list(all_environments(tax).keys())

    # Index all dataset items by breed for fast lookup
    breed_to_indices: dict[str, list[int]] = {}
    for i in range(len(dataset)):
        _, _, breed, _ = dataset[i]
        breed_to_indices.setdefault(breed, []).append(i)

    breeds = list(breed_to_indices.keys())
    triples: list[dict] = []
    seen: set[tuple] = set()

    attempts = 0
    while len(triples) < n and attempts < n * 20:
        attempts += 1
        breed   = rng.choice(breeds)
        cond    = rng.choice(conds)
        env     = rng.choice(envs)
        key     = (breed, cond, env)
        if key in seen:
            continue
        seen.add(key)
        ds_idx = rng.choice(breed_to_indices[breed])
        triples.append({
            "dataset_idx": ds_idx,
            "breed":       breed,
            "species":     species_for_breed(breed),
            "condition":   cond,
            "environment": env,
        })

    if len(triples) < n:
        raise RuntimeError(
            f"Could only build {len(triples)} unique triples (requested {n}). "
            "Try reducing n_triples or checking the dataset."
        )
    return triples


def _stem(cell: str, breed: str, cond: str, env: str, seed: int) -> str:
    slug = breed.lower().replace(" ", "_")
    return f"{cell}__{slug}__{cond}__{env}__{seed}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ablation(
    cells: Iterable[str] = ("A", "B", "C", "D"),
    n_triples: int = 10,
    seeds: list[int] | None = None,
    data_root: str = "data",
    gen_config_path: str = "configs/generation.yaml",
    tax_path: str = "configs/taxonomy.yaml",
    out_root: str = "outputs",
    controlnet_type: str | None = None,
    rng_seed: int = 0,
) -> dict[str, list[Path]]:
    """Run the 2×2 ablation for the requested cells.

    Parameters
    ----------
    cells           Which ablation cells to run. Subset of {"A","B","C","D"}.
    n_triples       Number of (breed, condition, environment) triples to use.
    seeds           Seed list. Defaults to generation.yaml seeds.
    data_root       Root dir for the Oxford dataset (must already be downloaded).
    gen_config_path Path to generation.yaml.
    tax_path        Path to taxonomy.yaml.
    out_root        Root output directory; per-cell subdirs created automatically.
    controlnet_type Override ControlNet modality ("seg"|"canny"). Reads config if None.
    rng_seed        Seed for the triple-sampling RNG (not the diffusion seeds).

    Returns
    -------
    Dict mapping cell label → list of saved image Paths.
    """
    cells = list(cells)
    seeds = seeds or get_seeds(gen_config_path)
    tax   = load_taxonomy(tax_path)

    dataset = OxfordPetDataset(root=data_root)
    triples = _build_triples(n_triples, dataset, tax, rng_seed=rng_seed)

    out_paths: dict[str, list[Path]] = {c: [] for c in cells}

    needs_baseline    = bool(set(cells) & {"A", "C"})
    needs_controlnet  = bool(set(cells) & {"B", "D"})

    total = len(triples) * len(seeds) * len(cells)
    pbar  = tqdm(total=total, desc="Ablation")

    for triple in triples:
        ds_idx  = triple["dataset_idx"]
        breed   = triple["breed"]
        species = triple["species"]
        cond    = triple["condition"]
        env     = triple["environment"]

        # Build prompt pairs once per triple
        naive_pp  = naive_input_to_prompt(breed=breed, condition=cond)
        struct_pp = structured_input_to_prompt(
            breed=breed, species=species, condition=cond, environment=env,
            tax_path=tax_path,
        )

        # Load the source image + trimap once per triple (lazy, only if needed)
        source_image = trimap = None
        if needs_controlnet:
            source_image, trimap, _, _ = dataset[ds_idx]

        for seed in seeds:
            stem_base = f"{breed.lower().replace(' ', '_')}__{cond}__{env}__{seed}"

            # ── Cell A: naive + no ControlNet ──────────────────────────────
            if "A" in cells:
                img, meta = run_baseline(
                    naive_pp,
                    seed=seed, breed=breed, species=species,
                    condition=cond, environment=env, cell="A",
                    gen_config_path=gen_config_path,
                )
                p, _ = save_image_with_sidecar(img, Path(out_root) / "A", f"A__{stem_base}", meta)
                out_paths["A"].append(p)
                pbar.update(1)

            # ── Cell B: naive + ControlNet ─────────────────────────────────
            if "B" in cells:
                img, meta, _ = run_controlnet(
                    naive_pp, source_image,
                    seed=seed, breed=breed, species=species,
                    condition=cond, environment=env, cell="B",
                    trimap=trimap, controlnet_type=controlnet_type,
                    gen_config_path=gen_config_path,
                )
                p, _ = save_image_with_sidecar(img, Path(out_root) / "B", f"B__{stem_base}", meta)
                out_paths["B"].append(p)
                pbar.update(1)

            # ── Cell C: structured + no ControlNet ────────────────────────
            if "C" in cells:
                img, meta = run_baseline(
                    struct_pp,
                    seed=seed, breed=breed, species=species,
                    condition=cond, environment=env, cell="C",
                    gen_config_path=gen_config_path,
                )
                p, _ = save_image_with_sidecar(img, Path(out_root) / "C", f"C__{stem_base}", meta)
                out_paths["C"].append(p)
                pbar.update(1)

            # ── Cell D: structured + ControlNet ───────────────────────────
            if "D" in cells:
                img, meta, _ = run_controlnet(
                    struct_pp, source_image,
                    seed=seed, breed=breed, species=species,
                    condition=cond, environment=env, cell="D",
                    trimap=trimap, controlnet_type=controlnet_type,
                    gen_config_path=gen_config_path,
                )
                p, _ = save_image_with_sidecar(img, Path(out_root) / "D", f"D__{stem_base}", meta)
                out_paths["D"].append(p)
                pbar.update(1)

    pbar.close()
    return out_paths
