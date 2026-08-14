"""Regenerate results/figs/dissociation_bars.pdf (Figure 1) from committed JSON.

The committed PDF predated this script and was hand-produced, so it silently
went stale when the answer-extraction rule changed (see
_common.LOOSE_FALLBACK_IS_AN_ANSWER): it still plotted baseline influence at
35.8%, the pre-correction value, against a current value of 25.6%. This script
exists so the figure is reproducible and cannot drift again.

Deliberate difference from the superseded figure: there is no 1/3 "chance"
line. Influence here is switch-to-cue, i.e. the *unconditional* rate at which a
model's answer equals the cue target, which has no 1/3 null. The 1/3 reference
belongs to the flip-to-cue rate *among flippers*, a different quantity reported
in the text (Task 2). Drawing it on this panel invited the reading that
baseline sits "well above chance" when its corrected influence is 25.6%, below
1/3.

Input:  results/reanalysis/D_self_distillation_negcontrol.json
Output: results/figs/dissociation_bars.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _common import REPO

SRC = REPO / "results/reanalysis/D_self_distillation_negcontrol.json"
OUT = REPO / "results/figs/dissociation_bars.pdf"

# (json key, display label)
ARMS = [
    ("baseline R1-7B",          "baseline\nR1-7B"),
    ("W2SR weak (R1-1.5B)",     "W2SR weak\n(R1-1.5B)"),
    ("W2SR strong (R1-14B)",    "W2SR strong\n(R1-14B)"),
    ("Self-A (R1-7B, 4k bud)",  "Self-A\n(self, 4k bud)"),
    ("Self-B (R1-7B, 8k bud)",  "Self-B\n(self, 8k bud)"),
]


def main() -> None:
    d = json.loads(SRC.read_text())
    ack = [100 * d["summary"][k]["ack_rate"] for k, _ in ARMS]
    inf = [100 * d["influence"][k]["rate"] for k, _ in ARMS]
    labels = [lab for _, lab in ARMS]

    x = range(len(ARMS))
    w = 0.38
    fig, axis = plt.subplots(figsize=(9.0, 3.9))

    axis.bar([i - w / 2 for i in x], ack, w,
             label="Acknowledgment (verbalization)",
             color="#3a3a3a", edgecolor="black", linewidth=0.6)
    axis.bar([i + w / 2 for i in x], inf, w,
             label="Influence (behavior: answer = cue target)",
             color="#c9c9c9", edgecolor="black", linewidth=0.6)

    for i, (a, b) in enumerate(zip(ack, inf)):
        axis.text(i - w / 2, a + 1.1, f"{a:.1f}", ha="center", fontsize=9)
        axis.text(i + w / 2, b + 1.1, f"{b:.1f}", ha="center", fontsize=9)

    axis.set_xticks(list(x))
    axis.set_xticklabels(labels, fontsize=9.5)
    axis.set_ylabel("rate (%)")
    axis.set_ylim(0, max(max(ack), max(inf)) * 1.22)
    axis.legend(frameon=False, fontsize=9.5, loc="upper left")
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(REPO)}")
    for (k, _), a, b in zip(ARMS, ack, inf):
        print(f"  {k:26s} ack {a:5.1f}%   influence {b:5.1f}%")


if __name__ == "__main__":
    main()
