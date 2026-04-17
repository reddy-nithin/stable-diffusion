"""CLI wrapper around src.generation.runner.run_ablation.

Usage
-----
    python scripts/generate_batch.py --cells all --n-triples 10 --seeds 4
    python scripts/generate_batch.py --cells A C --n-triples 5
    python scripts/generate_batch.py --cells B D --controlnet-type canny
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.runner import run_ablation
from src.generation.seeds import get_seeds


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the 2×2 ablation batch (cells A–D).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--cells",
        nargs="+",
        default=["all"],
        help='Cells to run: any subset of A B C D, or "all".',
    )
    p.add_argument(
        "--n-triples",
        type=int,
        default=10,
        help="Number of (breed, condition, environment) triples per cell.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds to use (takes first N from generation.yaml). Default: use all.",
    )
    p.add_argument(
        "--controlnet-type",
        default=None,
        choices=["seg", "canny"],
        help="Override ControlNet modality (reads generation.yaml primary if omitted).",
    )
    p.add_argument(
        "--out-root",
        default="outputs",
        help="Root output directory.",
    )
    p.add_argument(
        "--data-root",
        default="data",
        help="Root directory for Oxford-IIIT Pet dataset.",
    )
    p.add_argument(
        "--gen-config",
        default="configs/generation.yaml",
        help="Path to generation.yaml.",
    )
    p.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Seed for triple-sampling RNG (not diffusion seeds).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve cell list
    all_cells = ["A", "B", "C", "D"]
    if args.cells == ["all"]:
        cells = all_cells
    else:
        cells = [c.upper() for c in args.cells]
        invalid = set(cells) - set(all_cells)
        if invalid:
            print(f"ERROR: unknown cells {invalid}. Use A, B, C, D, or 'all'.", file=sys.stderr)
            sys.exit(1)

    # Resolve seeds
    all_seeds = get_seeds(args.gen_config)
    seeds = all_seeds[: args.seeds] if args.seeds else all_seeds

    print(f"Cells        : {cells}")
    print(f"Triples      : {args.n_triples}")
    print(f"Seeds        : {seeds}")
    print(f"ControlNet   : {args.controlnet_type or '(from config)'}")
    print(f"Output root  : {args.out_root}")
    print()

    out_paths = run_ablation(
        cells=cells,
        n_triples=args.n_triples,
        seeds=seeds,
        data_root=args.data_root,
        gen_config_path=args.gen_config,
        out_root=args.out_root,
        controlnet_type=args.controlnet_type,
        rng_seed=args.rng_seed,
    )

    print("\nDone. Images written:")
    for cell, paths in out_paths.items():
        print(f"  Cell {cell}: {len(paths)} images → {args.out_root}/{cell}/")


if __name__ == "__main__":
    main()
