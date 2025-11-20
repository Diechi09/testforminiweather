"""
plot_scaling.py
----------------
Utility script to visualise strong and weak scaling results for the miniWeather
mini-app. Reads CSV files in results/ and writes PNG/SVG plots back to that
folder. Intended for post-processing after Slurm experiments.
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def read_csv(path):
    """Return list of rows (dict) from a CSV file or empty list if missing."""
    if not path.exists():
        print(f"[WARN] CSV {path} not found. Skipping plot.")
        return []
    with path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_strong_scaling(rows):
    """Plot speedup and efficiency for strong scaling experiments."""
    if not rows:
        return
    nodes = [int(r.get("nodes", r.get("ranks", 0))) for r in rows]
    ranks = [int(r.get("ranks", 1)) for r in rows]
    times = [float(r["total_time"]) for r in rows]
    baseline = times[0]
    speedup = [baseline / t for t in times]
    efficiency = [s / ranks[i] for i, s in enumerate(speedup)]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(nodes, speedup, marker="o")
    ax[0].set_xlabel("Nodes")
    ax[0].set_ylabel("Speedup")
    ax[0].set_title("Strong scaling speedup")
    ax[0].grid(True)

    ax[1].plot(nodes, [e * 100 for e in efficiency], marker="o")
    ax[1].set_xlabel("Nodes")
    ax[1].set_ylabel("Efficiency (%)")
    ax[1].set_title("Strong scaling efficiency")
    ax[1].grid(True)

    for ext in ["png", "svg"]:
        out = RESULTS_DIR / f"strong_scaling.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[INFO] Wrote {out}")


def plot_weak_scaling(rows):
    """Plot time per step vs nodes for weak scaling experiments."""
    if not rows:
        return
    nodes = [int(r.get("nodes", r.get("ranks", 0))) for r in rows]
    t_per_step = [float(r.get("time_per_step", r.get("time", 0.0))) for r in rows]

    plt.figure(figsize=(5, 4))
    plt.plot(nodes, t_per_step, marker="o")
    plt.xlabel("Nodes")
    plt.ylabel("Time per step (s)")
    plt.title("Weak scaling: time per step")
    plt.grid(True)
    for ext in ["png", "svg"]:
        out = RESULTS_DIR / f"weak_scaling.{ext}"
        plt.savefig(out, bbox_inches="tight")
        print(f"[INFO] Wrote {out}")


def main():
    strong_rows = read_csv(RESULTS_DIR / "strong_scaling.csv")
    weak_rows = read_csv(RESULTS_DIR / "weak_scaling.csv")
    plot_strong_scaling(strong_rows)
    plot_weak_scaling(weak_rows)


if __name__ == "__main__":
    main()
