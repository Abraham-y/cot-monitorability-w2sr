"""Task H split — length-binned ack, restricted to influenced=1 vs influenced=0.

Question: the marginal residual at long lengths in Task H (baseline 27.7% vs
W2SR 15.0%, OR ~2.2, Fisher one-sided p~0.08) is computed across ALL cued +
judge-labeled samples, mixing influenced=1 (model moved by cue) with
influenced=0 (not moved). The faithfulness question is sharper when restricted
to influenced=1: given the cue actually moved you, did you say so?

This script produces the same binned analysis but split by influenced status.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent))
from _common import REPO, load_records, wilson, influenced

OUT_MD   = REPO / "results/reanalysis/H_length_binned_ack_by_influenced.md"
OUT_JSON = REPO / "results/reanalysis/H_length_binned_ack_by_influenced.json"

THICK_CUES = {"stanford_professor_recommends", "grader_hack_validation",
              "insider_information"}


def build_w2sr_thickened():
    thin = load_records("r1_7b_w2sr",      "openai_w2sr_r1_7b", cued_only=True)
    full = load_records("r1_7b_w2sr_full", "openai_w2sr_r1_7b", cued_only=True)
    out = []
    for r in full:
        if r.cue in THICK_CUES: out.append(r)
    for r in thin:
        if r.cue not in THICK_CUES: out.append(r)
    return out


def analyze(b, w, edges, label, sink):
    sink.append(f"\n### {label}   (baseline n={len(b)},  W2SR n={len(w)})\n")
    sink.append(f"{'bin':>3s} {'range (chars)':>22s} {'n_b':>4s} {'n_w':>4s} "
                f"{'ack_b':>14s} {'ack_w':>14s} {'Δ (pp)':>7s} {'OR':>5s} "
                f"{'Fisher 2-side p':>17s} {'1-side p':>10s}")

    rows = []
    def bin_idx(length: int) -> int:
        for i in range(len(edges) - 1):
            if edges[i] <= length < edges[i+1]:
                return i
        return len(edges) - 2

    for i in range(len(edges) - 1):
        lo, hi = int(edges[i] + (1 if i == 0 else 0)), int(edges[i+1])
        b_in = [r for r in b if bin_idx(r.cot_chars) == i]
        w_in = [r for r in w if bin_idx(r.cot_chars) == i]
        if not b_in or not w_in:
            sink.append(f"{i:3d} [{lo:,}, {hi:,})   skipped (empty side)")
            continue
        kb = sum(r.ack for r in b_in); nb = len(b_in)
        kw = sum(r.ack for r in w_in); nw = len(w_in)
        pb, pb_lo, pb_hi = wilson(kb, nb)
        pw, pw_lo, pw_hi = wilson(kw, nw)
        table = [[kb, nb - kb], [kw, nw - kw]]
        odds_ratio, fp_two = fisher_exact(table, alternative="two-sided")
        _, fp_one_b_gt_w = fisher_exact(table, alternative="greater")
        b_str = f"{kb}/{nb}={100*pb:.1f}%"
        w_str = f"{kw}/{nw}={100*pw:.1f}%"
        sink.append(f"{i:3d} {f'[{lo:,}, {hi:,})':>22s} "
                    f"{nb:4d} {nw:4d} "
                    f"{b_str:>14s} {w_str:>14s} {100*(pb-pw):+6.1f} "
                    f"{odds_ratio:5.2f} "
                    f"{fp_two:17.4f} {fp_one_b_gt_w:10.4f}")
        rows.append({
            "bin": i,
            "range_chars": [lo, hi],
            "baseline": {"k": kb, "n": nb, "rate": pb,
                         "ci95": [pb_lo, pb_hi]},
            "w2sr":     {"k": kw, "n": nw, "rate": pw,
                         "ci95": [pw_lo, pw_hi]},
            "delta_pp": 100 * (pb - pw),
            "odds_ratio": odds_ratio,
            "fisher_two_sided_p": fp_two,
            "fisher_one_sided_p_baseline_gt_w2sr": fp_one_b_gt_w,
        })
    return rows


def main():
    print("=" * 70)
    print("Task H split — length-binned ack by influenced status")
    print("=" * 70)

    base = load_records("r1_7b_baseline", "openai_DeepSeek-R1-Distill-Qwen-7B",
                        cued_only=True)
    w2sr = build_w2sr_thickened()

    b_all = [r for r in base if r.ack is not None]
    w_all = [r for r in w2sr if r.ack is not None]

    b_inf = [r for r in b_all if influenced(r) == 1]
    w_inf = [r for r in w_all if influenced(r) == 1]
    b_not = [r for r in b_all if influenced(r) == 0]
    w_not = [r for r in w_all if influenced(r) == 0]

    print(f"\nbaseline:   total {len(b_all)} ({len(b_inf)} influenced / "
          f"{len(b_not)} not influenced)")
    print(f"W2SR weak:  total {len(w_all)} ({len(w_inf)} influenced / "
          f"{len(w_not)} not influenced)")

    # Use the SAME bin edges as Task H (union of full distributions) so the
    # split rows are directly comparable to the unsplit headline.
    all_lens = np.array([r.cot_chars for r in b_all] + [r.cot_chars for r in w_all])
    edges = np.quantile(all_lens, [0.0, 0.20, 0.40, 0.60, 0.80, 1.0])
    edges[0] = -1
    edges[-1] = edges[-1] + 1
    print(f"\nshared bin edges (union of full distributions):")
    print(f"  {[int(x) for x in edges]}\n")

    lines = ["TASK H SPLIT — length-binned acknowledgment by influenced status\n",
             f"bin edges (chars, union of full distributions): "
             f"{[int(x) for x in edges]}"]

    rows_inf = analyze(b_inf, w_inf, edges, "INFLUENCED = 1 (answer hit the cue target)", lines)
    rows_not = analyze(b_not, w_not, edges, "INFLUENCED = 0 (answer did not hit the cue target)", lines)

    for line in lines:
        print(line)

    OUT_JSON.write_text(json.dumps({
        "edges": [int(x) for x in edges],
        "influenced_1": {"baseline_n": len(b_inf), "w2sr_n": len(w_inf),
                         "rows": rows_inf},
        "influenced_0": {"baseline_n": len(b_not), "w2sr_n": len(w_not),
                         "rows": rows_not},
    }, indent=2, default=str))
    OUT_MD.write_text("\n".join(["# Task H split — length-binned ack by influenced",
                                 "",
                                 "```",
                                 *lines,
                                 "```",
                                 ""]))
    print(f"\nWrote {OUT_JSON.relative_to(REPO)} and {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
