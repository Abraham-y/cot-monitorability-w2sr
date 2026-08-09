"""Generate the 2-panel length-binned acknowledgment figure for the poster.

Left:  All cued samples (the unconditional Task H result — gap closes at mid).
Right: Restricted to influenced=1 (safety-relevant subset — gap reopens at mid).

Reads:
  results/reanalysis/H_length_binned_ack.json
  results/reanalysis/H_length_binned_ack_by_influenced.json
Writes:
  results/figs/length_binned_ack.pdf  (overwrites the single-panel version)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / "results/figs/length_binned_ack.pdf"

BASELINE_COLOR = "#8C1515"  # Stanford cardinal
W2SR_COLOR     = "#2E2D29"  # near-black

BIN_LABELS = {
    2: "short\n[1.5k, 2.0k) chars",
    3: "mid\n[2.0k, 8.7k) chars",
    4: "long\n[8.7k, 38k) chars",
}


def load_rows(json_path: Path, key: str | None = None):
    with json_path.open() as f:
        d = json.load(f)
    return d["rows"] if key is None else d[key]["rows"]


def panel(ax, rows, title, show_ylabel: bool, include_bins=(3, 4)):
    rows = [r for r in rows if r["bin"] in include_bins]
    x = np.arange(len(rows))
    width = 0.36

    bvals = [100 * r["baseline"]["rate"] for r in rows]
    berrs_lo = [100 * (r["baseline"]["rate"] - r["baseline"]["ci95"][0]) for r in rows]
    berrs_hi = [100 * (r["baseline"]["ci95"][1] - r["baseline"]["rate"]) for r in rows]

    wvals = [100 * r["w2sr"]["rate"] for r in rows]
    werrs_lo = [100 * (r["w2sr"]["rate"] - r["w2sr"]["ci95"][0]) for r in rows]
    werrs_hi = [100 * (r["w2sr"]["ci95"][1] - r["w2sr"]["rate"]) for r in rows]

    ax.bar(x - width/2, bvals, width,
           yerr=[berrs_lo, berrs_hi],
           color=BASELINE_COLOR, label="baseline R1-7B",
           error_kw=dict(ecolor="black", capsize=3, lw=1.0))
    ax.bar(x + width/2, wvals, width,
           yerr=[werrs_lo, werrs_hi],
           color=W2SR_COLOR, label="W2SR weak",
           error_kw=dict(ecolor="black", capsize=3, lw=1.0))

    # Sample counts under each bar
    for i, r in enumerate(rows):
        ax.text(i - width/2, -4,
                f"n={r['baseline']['n']}", ha="center", va="top",
                fontsize=8, color=BASELINE_COLOR)
        ax.text(i + width/2, -4,
                f"n={r['w2sr']['n']}", ha="center", va="top",
                fontsize=8, color=W2SR_COLOR)

    # p-value annotation above the bar pair
    for i, r in enumerate(rows):
        p = r["fisher_two_sided_p"]
        if p < 0.005:
            stamp = f"p = {p:.3f}"
        elif p < 0.05:
            stamp = f"p = {p:.3f}"
        else:
            stamp = f"p = {p:.2f}"
        top = max(bvals[i] + berrs_hi[i], wvals[i] + werrs_hi[i])
        ax.text(i, top + 4, stamp, ha="center", va="bottom",
                fontsize=9, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([BIN_LABELS[r["bin"]] for r in rows], fontsize=9)
    ax.set_ylim(0, 95)
    if show_ylabel:
        ax.set_ylabel("acknowledgment rate (%)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)


def main():
    rows_all  = load_rows(REPO / "results/reanalysis/H_length_binned_ack.json")
    rows_inf  = load_rows(
        REPO / "results/reanalysis/H_length_binned_ack_by_influenced.json",
        key="influenced_1")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)

    panel(axL, rows_all,
          "All cued samples\n(gap closes at matched mid-length)",
          show_ylabel=True)
    panel(axR, rows_inf,
          "Restricted to influenced=1 (safety-relevant subset)\n"
          "(gap reopens substantially at matched mid-length)",
          show_ylabel=False)

    axL.legend(loc="upper left", fontsize=9, frameon=False)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
