"""Generate the mechanism-and-recovery figure for the paper.

Panel A: length-binned acknowledgment (baseline vs W2SR) on all cued samples
   and on the influenced=1 subset (from Task H).
Panel B: Task J recovery test (ack rate under three system-prompt conditions).

Reads:
  results/reanalysis/H_length_binned_ack.json
  results/reanalysis/H_length_binned_ack_by_influenced.json
  results/reanalysis/J_inference_recovery.json
Writes:
  results/figs/mechanism_and_recovery.pdf
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/figs/mechanism_and_recovery.pdf"

BASELINE_COLOR = "#8C1515"  # Stanford cardinal
W2SR_COLOR = "#2E2D29"
LONG_COT_COLOR = "#4E7B4E"  # muted green
ACK_COLOR = "#5B7DAF"       # muted blue
ICL_COLOR = "#A85B00"       # muted orange

BIN_LABELS = {2: "short", 3: "mid", 4: "long"}


def load_H_data():
    all_rows = json.loads((REPO / "results/reanalysis/H_length_binned_ack.json").read_text())["rows"]
    inf_data = json.loads((REPO / "results/reanalysis/H_length_binned_ack_by_influenced.json").read_text())
    inf_rows = inf_data["influenced_1"]["rows"]
    return all_rows, inf_rows


def load_J_data():
    return json.loads((REPO / "results/reanalysis/J_inference_recovery.json").read_text())


def panel_A_length(ax, all_rows, inf_rows):
    """Length-binned ack: all cued vs influenced=1, showing the story."""
    # Focus on bins that appear in both
    focus_bins = [3, 4]
    x = np.arange(len(focus_bins) * 2)
    width = 0.36
    labels = []
    baseline_rates = []
    baseline_errs_lo = []
    baseline_errs_hi = []
    w2sr_rates = []
    w2sr_errs_lo = []
    w2sr_errs_hi = []
    # Group-major order so the halves match the divider/spanning labels and
    # the paper caption: [all/mid, all/long | infl/mid, infl/long].
    for group, rows in [("all", all_rows), ("infl", inf_rows)]:
        for b in focus_bins:
            row = next((r for r in rows if r["bin"] == b), None)
            if row is None:
                # placeholder
                baseline_rates.append(0); w2sr_rates.append(0)
                baseline_errs_lo.append(0); baseline_errs_hi.append(0)
                w2sr_errs_lo.append(0); w2sr_errs_hi.append(0)
                labels.append(BIN_LABELS[b])
                continue
            br, w = row["baseline"], row["w2sr"]
            baseline_rates.append(100 * br["rate"])
            baseline_errs_lo.append(100 * (br["rate"] - br["ci95"][0]))
            baseline_errs_hi.append(100 * (br["ci95"][1] - br["rate"]))
            w2sr_rates.append(100 * w["rate"])
            w2sr_errs_lo.append(100 * (w["rate"] - w["ci95"][0]))
            w2sr_errs_hi.append(100 * (w["ci95"][1] - w["rate"]))
            labels.append(BIN_LABELS[b])

    ax.bar(x - width/2, baseline_rates, width,
           yerr=[baseline_errs_lo, baseline_errs_hi],
           color=BASELINE_COLOR, label="baseline R1-7B",
           error_kw=dict(ecolor="black", capsize=3, lw=0.8))
    ax.bar(x + width/2, w2sr_rates, width,
           yerr=[w2sr_errs_lo, w2sr_errs_hi],
           color=W2SR_COLOR, label="W2SR weak",
           error_kw=dict(ecolor="black", capsize=3, lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("acknowledgment rate (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("(A) Length-binned ack (Task H)\n"
                 "all cued samples vs influenced=1 subset",
                 fontsize=10)
    ax.axvline(1.5, color="gray", ls="--", lw=0.7, alpha=0.7)
    ax.text(0.5, 78, "all cued", ha="center", fontsize=8, color="gray")
    ax.text(2.5, 92, "influenced=1\n(safety-relevant)", ha="center",
            fontsize=8, color="gray")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_B_recovery(ax, jd):
    """Task J recovery: four conditions."""
    conds = ["none", "long_cot", "acknowledge", "icl_example"]
    labels = ["no system\nprompt", "long-CoT\nprompt",
              "acknowledge-\ninfluences", "few-shot\nICL example"]
    colors = [W2SR_COLOR, LONG_COT_COLOR, ACK_COLOR, ICL_COLOR]
    rates = [100 * jd[c]["rate"] for c in conds]
    ns = [jd[c]["n"] for c in conds]
    from math import sqrt
    # Wilson-ish approximation for CI
    def wilson(k, n, z=1.96):
        if n == 0: return (0, 0)
        p = k/n
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        half = (z * sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
        return max(0, center-half), min(1, center+half)
    errs_lo, errs_hi = [], []
    for c in conds:
        k = jd[c]["ack"]; n = jd[c]["n"]
        lo, hi = wilson(k, n)
        errs_lo.append(100 * (jd[c]["rate"] - lo))
        errs_hi.append(100 * (hi - jd[c]["rate"]))
    x = np.arange(len(conds))
    ax.bar(x, rates, 0.6, color=colors,
           yerr=[errs_lo, errs_hi],
           error_kw=dict(ecolor="black", capsize=4, lw=0.8))
    # Baseline R1-7B reference line
    ax.axhline(25.0, color=BASELINE_COLOR, ls=":", lw=1.2,
               label="baseline R1-7B (25.0%)")
    for i, (r, n) in enumerate(zip(rates, ns)):
        ax.text(i, r + max(errs_hi[i], 1) + 1, f"{r:.1f}%",
                ha="center", fontsize=8)
        ax.text(i, -2.5, f"n={n}", ha="center", va="top", fontsize=7,
                color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 32)
    ax.set_ylabel("acknowledgment rate (%)", fontsize=10)
    ax.set_title("(B) Inference-time recovery (Task J)\n"
                 "W2SR ack under four system-prompt conditions",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    all_rows, inf_rows = load_H_data()
    jd = load_J_data()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.5, 3.4))
    panel_A_length(axL, all_rows, inf_rows)
    panel_B_recovery(axR, jd)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
