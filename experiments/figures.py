#!/usr/bin/env python3
"""Paper figures from the oracle tables (CPU, matplotlib): evidence-sufficiency subsampling curves and the
footprint-noise ceiling. Reads <data-root>/experiments/oracles/*.csv, writes PDF + PNG under <repo>/paper/figures (local
folder, not part of the public repository). Numbers are macro-F1 in percent on the held-out car models (split_platform).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ORACLES = Path(C.DATA_ROOT) / "experiments" / "oracles"
COLORS = {"size": "#1f5fbf", "bev_ice": "#c2410c"}        # blue / dark orange: distinct for every colour-vision type
LABELS = {"size": "size (3 classes)", "bev_ice": "BEV vs ICE"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)


def fig_evidence(out: Path, achieved: dict[str, float] | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(ORACLES / "subsample.csv")
    df = df[df["split_col"] == "split_platform"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=150)
    for task in ("size", "bev_ice"):
        s = df[df["task"] == task].sort_values("m_windows")
        if not len(s):
            continue
        y, e = 100 * s["macro_f1"].to_numpy(), 100 * s["macro_f1_std"].to_numpy()
        ax.plot(s["m_windows"], y, color=COLORS[task], linewidth=2, marker="o", markersize=5, label=LABELS[task])
        ax.fill_between(s["m_windows"], y - e, y + e, color=COLORS[task], alpha=0.15, linewidth=0)
        ax.annotate(LABELS[task], (s["m_windows"].iloc[-1], y[-1]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8, color="#374151")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("windows sampled per drive (drives with at least 32 windows)")
    ax.set_ylabel("drive-level macro-F1 (%)")
    ax.set_ylim(40, 100)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def fig_footprint_noise(out: Path, achieved: float | None = None, ceiling: float | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(ORACLES / "footprint_noise.csv")
    df = df[(df["split_col"] == "split_platform") & (df["task"] == "size")]
    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=150)
    iid = df[df["tau"] == 0.0].sort_values("sigma")
    bias = df[df["sigma"] == 0.0].sort_values("tau")
    ax.plot(iid["sigma"], 100 * iid["macro_f1_mean"], color=COLORS["size"], linewidth=2, marker="o", markersize=5,
            label="per-drive noise (sigma)")
    ax.plot(bias["tau"], 100 * bias["macro_f1_mean"], color="#6b7280", linewidth=2, marker="s", markersize=5, linestyle="--",
            label="per-model bias (tau)")
    if achieved is not None:
        ax.axhline(achieved, color=COLORS["bev_ice"], linewidth=1.2, linestyle=":")
        ax.annotate(f"achieved {achieved:.0f}", (0.0, achieved), textcoords="offset points", xytext=(4, -11), fontsize=8, color=COLORS["bev_ice"])
    if ceiling is not None:
        ax.axhline(ceiling, color="#374151", linewidth=1.0, linestyle="-.")
        ax.annotate(f"per-model majority oracle {ceiling:.0f}", (0.0, ceiling), textcoords="offset points", xytext=(4, 4), fontsize=8, color="#374151")
    ax.axhline(90, color="#9ca3af", linewidth=0.8)
    ax.annotate("target 90", (0.30, 90), textcoords="offset points", xytext=(-2, 3), ha="right", fontsize=8, color="#6b7280")
    ax.set_xlabel("footprint error of an unseen model (m^2)")
    ax.set_ylabel("drive-level macro-F1 (%)")
    ax.set_ylim(40, 101)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "paper" / "figures")
    ap.add_argument("--achieved-size", type=float, default=None, help="achieved drive-level size macro-F1 (%%) to mark")
    ap.add_argument("--ceiling-size", type=float, default=None,
                    help="per-car-model majority-oracle macro-F1 (%%) to mark (default: mean over the reference runs)")
    a = ap.parse_args(argv)
    mo = pd.read_csv(ORACLES / "majority_oracle.csv")
    mo = mo[(mo["split_col"] == "split_platform") & (mo["task"] == "size") & (mo["arch"].isin(["gru", "resnet", "transformer", "mamba2"]))]
    ceiling = a.ceiling_size if a.ceiling_size is not None else (100 * mo["majority_macro_f1"].mean() if len(mo) else None)
    achieved = a.achieved_size if a.achieved_size is not None else (100 * mo["drive_macro_f1"].mean() if len(mo) else None)
    p1 = fig_evidence(a.out_dir / "evidence_sufficiency")
    p2 = fig_footprint_noise(a.out_dir / "footprint_noise_oracle", achieved=achieved, ceiling=ceiling)
    print(f"wrote {p1.name}.pdf/.png and {p2.name}.pdf/.png under {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
