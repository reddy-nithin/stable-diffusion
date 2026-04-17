"""Aggregated evaluation report.

Reads outputs/{A,B,C,D}/ and runs all three metrics (CLIPScore, DINOv2 breed
identity + condition consistency, LPIPS diversity), then:

1. Writes outputs/eval/summary.csv — one row per image, all metrics.
2. Writes outputs/eval/cell_summary.csv — one row per cell, aggregated means.
3. Writes outputs/eval/report.md — markdown table for the slide deck.
4. For each cell, copies the top-CLIPScore image and worst-CLIPScore image to
   outputs/eval/best/ and outputs/eval/worst/ for the qualitative grid.

CLI usage
---------
    python -m src.evaluation.report --in outputs --out outputs/eval
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def build_report(
    output_root: str | Path = "outputs",
    eval_dir: str | Path = "outputs/eval",
    cells: list[str] = None,
    dataset=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all three metrics and return (per_image_df, cell_summary_df).

    Parameters
    ----------
    output_root  Parent of per-cell output directories.
    eval_dir     Where to write CSV / markdown / best+worst images.
    cells        Cells to include; default ["A","B","C","D"].
    dataset      OxfordPetDataset for DINOv2 breed-identity references.
                 If None, breed_identity is skipped (NaN column in output).
    """
    from src.evaluation.clip_score import clip_scores_from_dir
    from src.evaluation.dino_sim import dino_scores_from_dirs
    from src.evaluation.lpips_diversity import lpips_scores_from_dirs

    cells     = cells or ["A", "B", "C", "D"]
    out_root  = Path(output_root)
    eval_path = Path(eval_dir)
    eval_path.mkdir(parents=True, exist_ok=True)

    # ── 1. CLIPScore (per-image) ────────────────────────────────────────────
    print("Computing CLIPScore…")
    clip_records: list[dict] = []
    for cell in cells:
        cell_dir = out_root / cell
        if cell_dir.exists():
            clip_records.extend(clip_scores_from_dir(cell_dir))
    clip_df = pd.DataFrame(clip_records).set_index("stem")

    # ── 2. DINOv2 (per-image + back-filled per-triple consistency) ──────────
    print("Computing DINOv2 scores…")
    dino_records = dino_scores_from_dirs(out_root, cells, dataset=dataset)
    dino_df = (
        pd.DataFrame(dino_records)
        .set_index("stem")[["breed_identity", "condition_consistency"]]
    )

    # ── 3. LPIPS diversity (per triple → repeated per image) ────────────────
    print("Computing LPIPS diversity…")
    lpips_records = lpips_scores_from_dirs(out_root, cells)
    lpips_df = (
        pd.DataFrame(lpips_records)
        .set_index("stem")[["lpips_diversity"]]
    )

    # ── 4. Merge ────────────────────────────────────────────────────────────
    per_image = clip_df.join(dino_df, how="left").join(lpips_df, how="left")
    per_image = per_image.reset_index()

    summary_path = eval_path / "summary.csv"
    per_image.to_csv(summary_path, index=False)
    print(f"  Wrote {summary_path}")

    # ── 5. Cell-level aggregation ────────────────────────────────────────────
    agg_cols = ["clip_score", "breed_identity", "condition_consistency", "lpips_diversity"]
    existing_cols = [c for c in agg_cols if c in per_image.columns]
    cell_summary = (
        per_image.groupby("cell")[existing_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    cell_summary.columns = ["_".join(c) for c in cell_summary.columns]
    cell_summary = cell_summary.reset_index()

    cell_csv = eval_path / "cell_summary.csv"
    cell_summary.to_csv(cell_csv, index=False)
    print(f"  Wrote {cell_csv}")

    # ── 6. Markdown report ──────────────────────────────────────────────────
    _write_markdown_report(cell_summary, per_image, eval_path)

    # ── 7. Best / worst images per cell ─────────────────────────────────────
    _extract_best_worst(per_image, out_root, eval_path)

    return per_image, cell_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_markdown_report(
    cell_summary: pd.DataFrame,
    per_image: pd.DataFrame,
    eval_path: Path,
) -> None:
    """Write a markdown table + findings paragraph to outputs/eval/report.md."""
    lines = [
        "# Evaluation Report",
        "",
        "## Cell-level metric summary",
        "",
        "| Cell | Prompt | ControlNet | CLIPScore ↑ | Breed Identity ↑ | Consistency ↑ | LPIPS Diversity ↑ |",
        "|------|--------|-----------|-------------|-----------------|--------------|-------------------|",
    ]

    cell_meta = {
        "A": ("Naive",      "No"),
        "B": ("Naive",      "Yes"),
        "C": ("Structured", "No"),
        "D": ("Structured", "Yes"),
    }

    for _, row in cell_summary.iterrows():
        cell = row["cell"]
        prompt_type, cn = cell_meta.get(cell, ("?", "?"))

        def _fmt(col: str) -> str:
            if col in row and pd.notna(row[col]):
                return f"{row[col]:.4f}"
            return "—"

        lines.append(
            f"| {cell} | {prompt_type} | {cn} "
            f"| {_fmt('clip_score_mean')} ± {_fmt('clip_score_std')} "
            f"| {_fmt('breed_identity_mean')} "
            f"| {_fmt('condition_consistency_mean')} "
            f"| {_fmt('lpips_diversity_mean')} |"
        )

    lines += [
        "",
        "## Findings",
        "",
        "> **Fill in after reviewing the numbers above.**",
        "",
        "- CLIPScore: Cell D (structured + ControlNet) expected to lead — prompt specificity directly improves alignment.",
        "- Breed Identity: ControlNet cells (B, D) should score higher — shape conditioning anchors the animal's silhouette.",
        "- Condition Consistency: Higher structured-prompt cells expected — more detailed prompt reduces semantic drift across seeds.",
        "- LPIPS Diversity: Should remain > 0.30 in all cells — confirm the model isn't mode-collapsing.",
        "",
        "## Best & Worst images per cell",
        "",
        "See `outputs/eval/best/` and `outputs/eval/worst/` directories.",
        "Best = highest CLIPScore per cell; Worst = lowest CLIPScore per cell.",
    ]

    report_path = eval_path / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"  Wrote {report_path}")


def _extract_best_worst(
    per_image: pd.DataFrame,
    out_root: Path,
    eval_path: Path,
) -> None:
    """Copy best and worst CLIPScore image per cell to eval/best/ and eval/worst/."""
    best_dir  = eval_path / "best"
    worst_dir = eval_path / "worst"
    best_dir.mkdir(exist_ok=True)
    worst_dir.mkdir(exist_ok=True)

    if "clip_score" not in per_image.columns:
        print("  Skipping best/worst extraction (clip_score not available).")
        return

    for cell, grp in per_image.groupby("cell"):
        grp_sorted = grp.sort_values("clip_score")
        for label, row in [("worst", grp_sorted.iloc[0]), ("best", grp_sorted.iloc[-1])]:
            stem = row["stem"]
            # stem contains the cell prefix (e.g. "A__beagle__...") — find directory
            src_dir = out_root / cell
            src = src_dir / f"{stem}.png"
            if src.exists():
                dest_dir = best_dir if label == "best" else worst_dir
                shutil.copy2(src, dest_dir / f"{cell}_{label}.png")

                # Copy sidecar too
                sidecar = src.with_suffix(".json")
                if sidecar.exists():
                    shutil.copy2(sidecar, dest_dir / f"{cell}_{label}.json")

    print(f"  Best/worst images → {best_dir.relative_to(out_root.parent)}/")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate evaluation report for the 2×2 ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in",  dest="in_dir",  default="outputs",      help="Ablation output root.")
    p.add_argument("--out", dest="out_dir", default="outputs/eval",  help="Eval output directory.")
    p.add_argument(
        "--cells", nargs="+", default=["A", "B", "C", "D"],
        help="Which cells to include.",
    )
    p.add_argument(
        "--skip-dino", action="store_true",
        help="Skip DINOv2 scoring (faster; omits breed_identity and consistency cols).",
    )
    p.add_argument(
        "--data-root", default="data",
        help="Oxford dataset root (needed for DINOv2 reference images).",
    )
    return p.parse_args()


if __name__ == "__main__":
    # Ensure repo root on path
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    args = _parse()

    dataset = None
    if not args.skip_dino:
        try:
            from src.data.dataset import OxfordPetDataset
            dataset = OxfordPetDataset(root=args.data_root)
            print(f"Loaded Oxford dataset: {len(dataset)} images")
        except Exception as e:
            print(f"WARNING: Could not load dataset ({e}). Skipping breed_identity.")

    per_image_df, cell_df = build_report(
        output_root=args.in_dir,
        eval_dir=args.out_dir,
        cells=args.cells,
        dataset=dataset,
    )

    print("\n── Cell summary ──")
    print(cell_df.to_string(index=False))
    print(f"\nFull per-image results: {args.out_dir}/summary.csv")
