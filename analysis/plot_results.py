"""Draw the measured two-round ranges; requires matplotlib, not a GPU."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "data/processed/2026-10-03/bench-per-round.csv").open() as f:
    rows = [r for r in csv.DictReader(f) if r["suite"] == "main"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
profiles = [("tp-nccl", "TP compiled + graph", "#2463A6"),
            ("ep-nccl", "EP compiled + graph", "#158477"),
            ("tp-nccl-eager", "TP eager", "#9263A4")]
for profile, label, color in profiles:
    for ax, key in zip(axes, ["output_throughput", "p50_tpot_ms"]):
        means, lower, upper = [], [], []
        for c in [1, 2, 4, 8]:
            values = [float(r[key]) for r in rows if r["profile"] == profile and int(r["concurrency"]) == c]
            assert len(values) == 2
            avg = sum(values) / 2
            means.append(avg)
            lower.append(avg - min(values))
            upper.append(max(values) - avg)
        ax.errorbar([1, 2, 4, 8], means, yerr=[lower, upper],
                    color=color, label=label, marker="o", linewidth=2,
                    markersize=5, capsize=3)
        ax.annotate(f"{means[-1]:.1f}", (8, means[-1]), xytext=(7, 0),
                    textcoords="offset points", va="center", color=color, fontsize=9)
for ax in axes:
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xlim(.65, 9.1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Configured concurrency")
    ax.grid(axis="y", alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Output throughput (tokens/s)")
axes[1].set_ylabel("Mean of per-round P50 TPOT (ms)")
axes[0].set_title("Throughput under fixed load", loc="left", fontsize=12)
axes[1].set_title("Time per output token", loc="left", fontsize=12)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle("Qwen3-30B-A3B FP8 | 2 RTX 3090 | vLLM 0.30.0", fontsize=13, y=.99)
fig.text(.05, .015, "1024 input / 256 output tokens. Points: 2-round means. Bars: observed min/max, not confidence intervals.", fontsize=8.3, color="#555555")
fig.tight_layout(rect=[0, .05, 1, .95])
dest = ROOT / "docs/assets"
dest.mkdir(parents=True, exist_ok=True)
fig.savefig(dest / "performance.png", dpi=180)
fig.savefig(dest / "performance.svg")
print(dest / "performance.png")
