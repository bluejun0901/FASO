import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def ema(values: List[float], alpha: float = 0.98) -> List[float]:
    """Exponential moving average smoothing (reference: notebooks/comp_graph.ipynb)."""
    smoothed: List[float] = []
    m: Optional[float] = None
    for v in values:
        m = v if m is None else alpha * m + (1 - alpha) * v
        smoothed.append(m)
    return smoothed


def extract_run_and_step(model_path: str) -> Tuple[str, Optional[int]]:
    """Split a model path like 'experiment1/.../checkpoint-12000' into (run_dir, step)."""
    # Expect suffix 'checkpoint-<int>'
    if "checkpoint-" not in model_path:
        return model_path, None
    run_dir, ckpt = model_path.rsplit("/checkpoint-", 1)
    # ckpt may have trailing components if any (rare); pick leading int
    m = re.match(r"(\d+)", ckpt)
    step = int(m.group(1)) if m else None
    return run_dir, step


def load_dir_to_name(mapping_path: Path) -> Dict[str, str]:
    if not mapping_path.exists():
        return {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def plot_file(json_path: Path, out_dir: Path, dir_to_name: Dict[str, str], smooth: bool = False) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            records: List[Dict] = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping {json_path} (invalid JSON)")
            return

    # Group by ref_model so each figure compares target runs vs the same reference
    from collections import defaultdict
    by_ref: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        # Ensure required fields
        if not isinstance(rec, dict):
            continue
        ref = rec.get("ref_model")
        target = rec.get("target_model")
        win_rate = rec.get("win_rate")
        if not ref or not target or win_rate in (None, ""):
            continue
        by_ref[ref].append(rec)

    if not by_ref:
        print(f"No valid entries with win_rate found in {json_path}")
        return

    for ref_model, items in by_ref.items():
        # Organize items by target run
        runs: Dict[str, List[Tuple[int, float]]] = {}
        for it in items:
            t_run, step = extract_run_and_step(str(it.get("target_model")))
            if step is None:
                # Skip entries without a checkpoint step
                continue
            runs.setdefault(t_run, []).append((step, float(it.get("win_rate"))))

        if not runs:
            print(f"No target runs with checkpoint steps in {json_path} (ref={ref_model})")
            continue

        # Compute normalization per run (percent of max step in the file for that run)
        run_max: Dict[str, int] = {r: max(s for s, _ in pts) for r, pts in runs.items()}

        plt.figure(figsize=(10, 5))
        for i, (run_dir, pts) in enumerate(sorted(runs.items(), key=lambda kv: dir_to_name.get(kv[0], kv[0]))):
            pts_sorted = sorted(pts, key=lambda x: x[0])
            steps = [s for s, _ in pts_sorted]
            wrs = [w for _, w in pts_sorted]
            pct = [100.0 * s / run_max[run_dir] if run_max[run_dir] > 0 else 0.0 for s in steps]

            label = dir_to_name.get(run_dir, run_dir)
            if smooth and len(wrs) >= 3:
                wrs_s = ema(wrs)
                plt.plot(pct, wrs_s, label=f"{label} (ema)", color=f"C{i}")
                plt.scatter(pct, wrs, s=15, color=f"C{i}", alpha=0.4)
            else:
                plt.plot(pct, wrs, label=label, color=f"C{i}")
                plt.scatter(pct, wrs, s=15, color=f"C{i}")

        base = json_path.stem
        ref_short = sanitize_filename(ref_model)
        title = f"Win Rate vs Training Progress — {base}\nRef: {ref_model}"
        plt.title(title)
        plt.xlabel("Training Progress (% of run checkpoints in file)")
        plt.ylabel("Win Rate")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()

        out_file = out_dir / f"winrate_{sanitize_filename(base)}__ref_{ref_short}.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved {out_file}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot and compare win rates across training checkpoints.")
    parser.add_argument("--comparisons-dir", type=Path, default=Path("dataset/comparisons"), help="Directory with comparison JSON files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to save PNG figures")
    parser.add_argument("--smooth", action="store_true", help="Apply EMA smoothing to win rate curves")
    args = parser.parse_args()

    comp_dir: Path = args.comparisons_dir
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dir_to_name = load_dir_to_name(comp_dir / "dir_to_name.json")

    # Process all JSON files in comparisons dir except the mapping file
    json_files = [p for p in comp_dir.glob("*.json") if p.name != "dir_to_name.json"]
    if not json_files:
        print(f"No JSON files found in {comp_dir}")
        return

    for jf in sorted(json_files):
        plot_file(jf, out_dir, dir_to_name, smooth=args.smooth)


if __name__ == "__main__":
    main()

