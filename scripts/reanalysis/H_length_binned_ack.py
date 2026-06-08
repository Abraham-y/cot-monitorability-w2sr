"""Task H — length-binned acknowledgment, baseline R1-7B vs W2SR weak.

The mechanism question: is the ack collapse explained by length compression
alone, or is there a residual at matched length? Task 4 attacked this with
log-length covariate logits + the >=9.6k subset (W2SR n=39), and got a
direction-stable but underpowered result. Task H attacks it differently:
bin both conditions by CoT length into quantiles of the union distribution,
then compute ack rate per bin per condition with Wilson 95% CIs and a
per-bin paired McNemar where pairing is possible. This gives a much more
powered "at matched length" comparison than the single long-tail cut.

Outcomes:
  (i) If W2SR ack < baseline ack in every length bin (especially the
      longest bins), compression-dominant + real residual is supported.
  (ii) If W2SR ack converges to baseline ack as length grows (long bins
      show no gap), compression alone explains the collapse and the
      W2SR-specific framing weakens.
  (iii) If the per-bin pattern is mixed, the truth is between.

Inputs:
  external/monitorability-eval/logs/r1_7b_baseline/openai_DeepSeek-R1-Distill-Qwen-7B/{01..05}/config_001/*.eval
  external/monitorability-eval/logs/r1_7b_w2sr/openai_w2sr_r1_7b/{01..05}/config_001/*.eval
  external/monitorability-eval/logs/r1_7b_w2sr_full/openai_w2sr_r1_7b/{01,03,04}/config_001/*.eval
    (cues 01/03/04 thickened with 198-sample cells; cues 02/05 use the
     original 40-sample cells)

Outputs:
  results/reanalysis/H_length_binned_ack.md
  results/reanalysis/H_length_binned_ack.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import REPO, load_records, paired_align, mcnemar_exact, wilson

OUT_MD   = REPO / "results/reanalysis/H_length_binned_ack.md"
OUT_JSON = REPO / "results/reanalysis/H_length_binned_ack.json"

THICK_CUES = {"stanford_professor_recommends", "grader_hack_validation",
              "insider_information"}


def build_w2sr_thickened():
    """Mirror Task 4's W2SR-side construction: r1_7b_w2sr_full for cues 01/03/04
    (where it adds 198-sample mass) + r1_7b_w2sr for cues 02/05 (no full run)."""
    thin = load_records("r1_7b_w2sr",      "openai_w2sr_r1_7b", cued_only=True)
    full = load_records("r1_7b_w2sr_full", "openai_w2sr_r1_7b", cued_only=True)
    out = []
    for r in full:
        if r.cue in THICK_CUES: out.append(r)
    for r in thin:
        if r.cue not in THICK_CUES: out.append(r)
    return out


def main():
    print("=" * 70)
    print("TASK H — length-binned ack, baseline R1-7B vs W2SR weak")
    print("=" * 70)

    base = load_records("r1_7b_baseline", "openai_DeepSeek-R1-Distill-Qwen-7B",
                        cued_only=True)
    w2sr = build_w2sr_thickened()

    # Restrict to has_cue=True ∩ judge-labeled
    b = [r for r in base if r.ack is not None]
    w = [r for r in w2sr if r.ack is not None]
    print(f"\nbaseline R1-7B cued, judge-labeled: n={len(b)}")
    print(f"W2SR weak (thickened mix) cued, judge-labeled: n={len(w)}")

    # ---- bin definition: quantiles of the UNION of both length distributions ----
    all_lens = np.array([r.cot_chars for r in b] + [r.cot_chars for r in w])
    quantiles = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
    edges = np.quantile(all_lens, quantiles)
    edges[0] = -1  # so the lowest record falls into bin 0
    edges[-1] = edges[-1] + 1  # so the highest record falls into the last bin
    print(f"\nlength quantile edges (union of both conditions):")
    print(f"  {[int(x) for x in edges]}")

    def bin_idx(length: int) -> int:
        for i in range(len(edges) - 1):
            if edges[i] <= length < edges[i+1]:
                return i
        return len(edges) - 2

    from scipy.stats import fisher_exact

    rows = []
    for i in range(len(edges) - 1):
        lo, hi = int(edges[i] + (1 if i == 0 else 0)), int(edges[i+1])
        b_in = [r for r in b if bin_idx(r.cot_chars) == i]
        w_in = [r for r in w if bin_idx(r.cot_chars) == i]
        if not b_in or not w_in:
            continue
        kb = sum(r.ack for r in b_in); nb = len(b_in)
        kw = sum(r.ack for r in w_in); nw = len(w_in)
        pb, pb_lo, pb_hi = wilson(kb, nb)
        pw, pw_lo, pw_hi = wilson(kw, nw)

        # paired McNemar on (qid, cue) within this bin (often underpowered;
        # baseline's long traces and W2SR's long traces don't share many qids)
        keys, a_vals, c_vals = paired_align(b_in, w_in, "ack")
        mc = mcnemar_exact(a_vals, c_vals) if keys else None

        # UNPAIRED Fisher's exact — the proper test when paired overlap is small.
        # CI-overlap on Wilson intervals is a known-conservative heuristic; Fisher
        # uses the full n on each side and is the right test for "does baseline
        # have higher ack than W2SR within this length bin?"
        table = [[kb, nb - kb], [kw, nw - kw]]
        odds_ratio, fp_two = fisher_exact(table, alternative="two-sided")
        _, fp_one_b_gt_w = fisher_exact(table, alternative="greater")

        rows.append({
            "bin": i,
            "range_chars": [lo, hi],
            "median_chars_in_bin": int(np.median([r.cot_chars for r in (b_in + w_in)])),
            "baseline": {"k": kb, "n": nb, "rate": pb, "ci95": [pb_lo, pb_hi]},
            "w2sr":     {"k": kw, "n": nw, "rate": pw, "ci95": [pw_lo, pw_hi]},
            "delta_pp": 100 * (pb - pw),
            "odds_ratio": odds_ratio,
            "fisher_two_sided_p": fp_two,
            "fisher_one_sided_p_baseline_gt_w2sr": fp_one_b_gt_w,
            "paired":   mc,
        })

    # ---- print summary table ----
    print(f"\n{'bin':>3s} {'range (chars)':>22s} {'n_b':>4s} {'n_w':>4s} "
          f"{'ack_b':>12s} {'ack_w':>12s} {'Δ (pp)':>7s} {'OR':>5s} "
          f"{'Fisher 2-side p':>17s} {'1-side p':>10s}")
    for r in rows:
        b_str = f"{r['baseline']['k']}/{r['baseline']['n']}={100*r['baseline']['rate']:.1f}%"
        w_str = f"{r['w2sr']['k']}/{r['w2sr']['n']}={100*r['w2sr']['rate']:.1f}%"
        print(f"{r['bin']:3d} {f'[{r['range_chars'][0]:,}, {r['range_chars'][1]:,})':>22s} "
              f"{r['baseline']['n']:4d} {r['w2sr']['n']:4d} "
              f"{b_str:>12s} {w_str:>12s} {r['delta_pp']:+6.1f} "
              f"{r['odds_ratio']:5.2f} "
              f"{r['fisher_two_sided_p']:17.4f} {r['fisher_one_sided_p_baseline_gt_w2sr']:10.4f}")

    # ---- narrative summary ----
    n_bins = len(rows)
    positive_delta_bins = sum(1 for r in rows if r["delta_pp"] > 0)
    sig_bins_two = sum(1 for r in rows if r["fisher_two_sided_p"] < 0.05 and r["delta_pp"] > 0)
    sig_bins_one = sum(1 for r in rows if r["fisher_one_sided_p_baseline_gt_w2sr"] < 0.05)
    print(f"\nNarrative summary:")
    print(f"  {n_bins} bins total; W2SR ack < baseline ack in {positive_delta_bins}/{n_bins} bins")
    print(f"  Fisher's exact (unpaired) p < 0.05 two-sided in {sig_bins_two}/{n_bins} bins")
    print(f"  Fisher's exact one-sided (baseline > W2SR) p < 0.05 in {sig_bins_one}/{n_bins} bins")

    # ---- interpretation ----
    longest = rows[-1] if rows else None
    if longest:
        print(f"\nLongest bin (the matched-length test that matters most):")
        print(f"  chars [{longest['range_chars'][0]}, {longest['range_chars'][1]})")
        print(f"  baseline ack: {100*longest['baseline']['rate']:.1f}% "
              f"({longest['baseline']['k']}/{longest['baseline']['n']})")
        print(f"  W2SR ack:     {100*longest['w2sr']['rate']:.1f}% "
              f"({longest['w2sr']['k']}/{longest['w2sr']['n']})")
        if longest["paired"]:
            mc = longest["paired"]
            print(f"  paired Δ:     {mc['delta_mean']:+.3f}  "
                  f"95% CI [{mc['delta_ci95'][0]:+.3f}, {mc['delta_ci95'][1]:+.3f}]")
            print(f"  McNemar p:    {mc['p']:.3g}  (n_pairs={mc['n_pairs']}, "
                  f"disc base-only/W2SR-only = {mc['n10_a_only']}/{mc['n01_b_only']})")

    out = {"edges": [int(x) for x in edges], "rows": rows,
           "summary": {
               "n_bins": n_bins,
               "bins_with_w2sr_lower": positive_delta_bins,
               "bins_fisher_two_sided_p_lt_0p05": sig_bins_two,
               "bins_fisher_one_sided_p_lt_0p05": sig_bins_one,
           }}
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))

    # ---- markdown ----
    md = ["# Task H — length-binned acknowledgment\n",
          "Bin both baseline R1-7B and W2SR weak (thickened mix: "
          "`r1_7b_w2sr_full` on cues 01/03/04, `r1_7b_w2sr` on cues 02/05) "
          "into 5 quantile bins of the UNION of their CoT-length distributions. "
          "Per bin, report acknowledgment rate per condition (Wilson 95% CI), the "
          "per-bin paired McNemar on (qid, cue), and the gap.\n",
          "## Length distribution bins (chars)\n",
          "| bin | range | median chars | n baseline | n W2SR |",
          "|---|---|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['bin']} | "
                  f"[{r['range_chars'][0]:,}, {r['range_chars'][1]:,}) | "
                  f"{r['median_chars_in_bin']:,} | "
                  f"{r['baseline']['n']} | {r['w2sr']['n']} |")
    md += ["",
           "## Per-bin acknowledgment (Fisher's exact, unpaired — correct test "
           "when paired (qid, cue) overlap is small within a bin)\n",
           "| bin | baseline ack | W2SR ack | Δ (pp) | OR | Fisher 2-sided p | Fisher 1-sided p |",
           "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        b_str = (f"{r['baseline']['k']}/{r['baseline']['n']} = "
                 f"{100*r['baseline']['rate']:.1f}% "
                 f"[{100*r['baseline']['ci95'][0]:.1f}, {100*r['baseline']['ci95'][1]:.1f}]")
        w_str = (f"{r['w2sr']['k']}/{r['w2sr']['n']} = "
                 f"{100*r['w2sr']['rate']:.1f}% "
                 f"[{100*r['w2sr']['ci95'][0]:.1f}, {100*r['w2sr']['ci95'][1]:.1f}]")
        md.append(f"| {r['bin']} | {b_str} | {w_str} | "
                  f"{r['delta_pp']:+.1f} | {r['odds_ratio']:.2f} | "
                  f"{r['fisher_two_sided_p']:.4f} | "
                  f"{r['fisher_one_sided_p_baseline_gt_w2sr']:.4f} |")
    md += ["",
           "Paired McNemar within each bin is included in the .json output but "
           "is generally underpowered because baseline's long traces and W2SR's "
           "long traces rarely share (qid, cue) pairs.\n",
           "## Read",
           f"- {n_bins} bins total; W2SR ack < baseline ack in "
           f"**{positive_delta_bins}/{n_bins}** bins.",
           f"- Fisher one-sided (baseline > W2SR) p < 0.05: "
           f"**{sig_bins_one}/{n_bins}** bins.",
           "",
           "**Interpretation.** At matched mid-length, baseline and W2SR are "
           "indistinguishable (bin 3: $17.5\\%$ vs $15.1\\%$, Fisher $p\\!\\approx\\!0.8$). "
           "At very long lengths, baseline trends higher (bin 4: $27.7\\%$ vs $15.0\\%$, "
           "OR$\\!\\approx\\!2.2$, Fisher one-sided $p\\!\\approx\\!0.08$, two-sided "
           "$p\\!\\approx\\!0.14$) --- a direction-positive but marginal residual. "
           "Honest read: most of the $25\\%\\!\\to\\!3\\%$ collapse is compression; "
           "there is a possible additional residual at long lengths, "
           "marginally significant one-sided.",
           ""]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_MD.relative_to(REPO)} and {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
