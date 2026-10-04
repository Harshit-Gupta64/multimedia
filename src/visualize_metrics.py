"""
src/visualize_metrics.py - Benchmark Visualization (headless, publication-ready)
----------------------------------------------------------------------------------
Reads benchmark_results.csv and generates two high-resolution figures:
  1. reports/latency_distribution.png  - Boxplot: GK inference vs scheduler latency
  2. reports/bandwidth_savings.png     - Grouped bar: raw vs obfuscated size + savings %

Run headless (no GUI window):
    .venv\Scripts\python.exe src/visualize_metrics.py
"""
import os
import sys
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Must set backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Ensure project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CSV_PATH     = "benchmark_results.csv"
REPORTS_DIR  = "reports"
SCENARIO_KEYS  = ["low", "medium", "high"]
SCENARIO_LABELS = {
    "low":    "Low Density\n(0-2 obj)",
    "medium": "Medium Density\n(3-5 obj)",
    "high":   "High Density\n(>5 obj)",
}
PALETTE = {
    "low":    "#4C72B0",
    "medium": "#DD8452",
    "high":   "#55A868",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(path: str) -> Dict[str, List[dict]]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            "Run 'python src/benchmark.py' first to generate benchmark_results.csv"
        )
    buckets: Dict[str, List[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            buckets[row["scenario"]].append(row)
    return buckets


def _float_col(rows: List[dict], col: str) -> List[float]:
    return [float(r[col]) for r in rows if r.get(col) not in ("", None)]


# ── Figure 1 : Latency Distribution ──────────────────────────────────────────

def plot_latency_distribution(buckets: Dict[str, List[dict]], out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "Gatekeeper & Scheduler Latency Distribution\nby Semantic Density Scenario",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # --- Left: GK inference latency boxplot (exclude cold-start rep=1 outlier) ---
    ax_gk = axes[0]
    gk_data = []
    labels  = []
    for key in SCENARIO_KEYS:
        rows = buckets.get(key, [])
        vals = [float(r["gk_latency_ms"]) for r in rows if r.get("rep", "1") != "1"]
        gk_data.append(vals)
        labels.append(SCENARIO_LABELS[key])

    bp = ax_gk.boxplot(
        gk_data, patch_artist=True, widths=0.45,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    for patch, key in zip(bp["boxes"], SCENARIO_KEYS):
        patch.set_facecolor(PALETTE[key])
        patch.set_alpha(0.85)

    ax_gk.set_title("YOLOv8 Inference Latency\n(excl. cold-start)", fontsize=11)
    ax_gk.set_ylabel("Latency (ms)", fontsize=10)
    ax_gk.set_xticklabels(labels, fontsize=9)
    ax_gk.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax_gk.set_axisbelow(True)

    # --- Right: Scheduler decision latency bar (mean ± std, in µs) ---
    ax_sched = axes[1]
    means, stds, xs = [], [], []
    for i, key in enumerate(SCENARIO_KEYS):
        rows = buckets.get(key, [])
        vals = [float(r["sched_latency_ms"]) * 1000 for r in rows]  # ms -> µs
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        xs.append(i)

    bars = ax_sched.bar(
        xs, means, yerr=stds, capsize=5, width=0.45,
        color=[PALETTE[k] for k in SCENARIO_KEYS], alpha=0.85,
        error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
    )
    for bar, mean in zip(bars, means):
        ax_sched.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.15,
            f"{mean:.0f} µs", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax_sched.set_title("Scheduler Decision Latency\n(mean ± σ, µs)", fontsize=11)
    ax_sched.set_ylabel("Latency (µs)", fontsize=10)
    ax_sched.set_xticks(xs)
    ax_sched.set_xticklabels(labels, fontsize=9)
    ax_sched.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax_sched.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {os.path.abspath(out_path)}")


# ── Figure 2 : Bandwidth Savings ─────────────────────────────────────────────

def plot_bandwidth_savings(buckets: Dict[str, List[dict]], out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Bandwidth Reduction by Gaussian Obfuscation\nRaw Frame vs Obfuscated JPEG",
        fontsize=14, fontweight="bold", y=1.01,
    )

    x      = np.arange(len(SCENARIO_KEYS))
    width  = 0.32

    # --- Left: grouped bar - raw KB vs obfuscated KB ---
    ax_bar = axes[0]
    raw_means, obf_means = [], []
    for key in SCENARIO_KEYS:
        rows = buckets.get(key, [])
        raw_means.append(np.mean(_float_col(rows, "raw_bytes")) / 1024)
        obf_means.append(np.mean(_float_col(rows, "obf_bytes")) / 1024)

    bars_raw = ax_bar.bar(x - width/2, raw_means, width, label="Raw Frame",
                          color="#4C72B0", alpha=0.85)
    bars_obf = ax_bar.bar(x + width/2, obf_means, width, label="Obfuscated JPEG",
                          color="#55A868", alpha=0.85)

    for bar, val in zip(bars_raw, raw_means):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f} KB", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_obf, obf_means):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f} KB", ha="center", va="bottom", fontsize=8)

    ax_bar.set_title("Frame Size Comparison (KB)", fontsize=11)
    ax_bar.set_ylabel("Size (KB)", fontsize=10)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([SCENARIO_LABELS[k] for k in SCENARIO_KEYS], fontsize=9)
    ax_bar.legend(fontsize=9)
    ax_bar.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax_bar.set_axisbelow(True)

    # --- Right: bandwidth reduction % per scenario ---
    ax_pct = axes[1]
    pct_means, pct_stds = [], []
    for key in SCENARIO_KEYS:
        rows = buckets.get(key, [])
        vals = _float_col(rows, "bandwidth_reduction_pct")
        pct_means.append(np.mean(vals))
        pct_stds.append(np.std(vals))

    bar_pct = ax_pct.bar(
        x, pct_means, yerr=pct_stds, capsize=5, width=0.45,
        color=[PALETTE[k] for k in SCENARIO_KEYS], alpha=0.85,
        error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
    )
    for bar, pct in zip(bar_pct, pct_means):
        ax_pct.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax_pct.set_title("Bandwidth Reduction (%)", fontsize=11)
    ax_pct.set_ylabel("Reduction (%)", fontsize=10)
    ax_pct.set_ylim(0, 55)
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels([SCENARIO_LABELS[k] for k in SCENARIO_KEYS], fontsize=9)
    ax_pct.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax_pct.set_axisbelow(True)

    # Annotation arrow
    ax_pct.annotate(
        "Gaussian blur reduces\nJPEG entropy uniformly",
        xy=(0, pct_means[0]), xytext=(0.6, pct_means[0] + 12),
        arrowprops={"arrowstyle": "->", "color": "#555"},
        fontsize=8, color="#555",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {os.path.abspath(out_path)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print(f"Reading {CSV_PATH} …")
    buckets = load_csv(CSV_PATH)
    total   = sum(len(v) for v in buckets.values())
    print(f"Loaded {total} rows across {len(buckets)} scenarios.")

    plot_latency_distribution(buckets, f"{REPORTS_DIR}/latency_distribution.png")
    plot_bandwidth_savings   (buckets, f"{REPORTS_DIR}/bandwidth_savings.png")

    print("\n[DONE] All figures generated in reports/")


if __name__ == "__main__":
    main()

