"""Task B — per-text-cue effect sizes.

`visual_squares` and `xml_metadata` floor out for every condition (Task 5
matrix: 0% or near-zero across the board). Restrict to the three text cues
where the substrate has dynamic range:

    01_stanford_professor → cue_name "stanford_professor_recommends"
    03_grader_hack        → cue_name "grader_hack_validation"
    04_unethical_info     → cue_name "insider_information"

For each text cue, baseline R1-7B vs W2SR weak:
  - acknowledgment rate per condition (denominator = has_cue with judge label),
    Wilson 95% CI,
  - switch-to-cue rate per condition (denominator = has_cue ∩ parseable ∩
    baseline_ans ≠ cue_target), Wilson 95% CI,
  - paired Δ on each metric, matched on `qid`, with exact McNemar.

The headline claim is the dissociation: acknowledgment DOWN, switch-to-cue
FLAT-OR-UP. Confirm it holds within each text cue, not just pooled.

Inputs:
  external/monitorability-eval/logs/r1_7b_baseline/openai_DeepSeek-R1-Distill-Qwen-7B/{01_stanford_professor,03_grader_hack,04_unethical_information}/config_001/*.eval
  external/monitorability-eval/logs/r1_7b_w2sr/openai_w2sr_r1_7b/{01_stanford_professor,03_grader_hack,04_unethical_information}/config_001/*.eval

Outputs:
  results/reanalysis/B_per_cue_effects.md
  results/reanalysis/B_per_cue_effects.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    REPO, load_records, paired_align, mcnemar_exact, wilson,
)

OUT_MD   = REPO / "results/reanalysis/B_per_cue_effects.md"
OUT_JSON = REPO / "results/reanalysis/B_per_cue_effects.json"

TEXT_CUES = ["stanford_professor_recommends",
             "grader_hack_validation",
             "insider_information"]


def main():
    print("=" * 70)
    print("TASK B — per-text-cue effect sizes")
    print("=" * 70)

    base = load_records("r1_7b_baseline", "openai_DeepSeek-R1-Distill-Qwen-7B", cued_only=True)
    w2sr = load_records("r1_7b_w2sr",     "openai_w2sr_r1_7b",                  cued_only=True)

    # Tag derived attributes used in pairing
    for r in base + w2sr:
        r.switch_to_cue = (None if (r.answer is None or r.cue_target is None
                                    or r.baseline_ans is None
                                    or r.baseline_ans == r.cue_target)
                           else int(r.answer == r.cue_target))

    per_cue = {}
    print(f"\n{'cue':34s} {'metric':12s} {'baseline':>26s} {'W2SR weak':>26s} {'paired Δ':>26s}")
    print("-" * 130)
    for cue in TEXT_CUES:
        b = [r for r in base if r.cue == cue]
        w = [r for r in w2sr if r.cue == cue]

        # acknowledgment — denominator: has_cue with judge label
        b_ack = [r.ack for r in b if r.ack is not None]
        w_ack = [r.ack for r in w if r.ack is not None]
        ka, na = sum(b_ack), len(b_ack); kw, nw = sum(w_ack), len(w_ack)
        a_b, alo_b, ahi_b = wilson(ka, na)
        a_w, alo_w, ahi_w = wilson(kw, nw)
        # paired
        keys_a, va, vb = paired_align(b, w, "ack")
        mc_ack = mcnemar_exact(va, vb)

        # switch-to-cue — denominator: cued ∩ parseable ∩ baseline_ans != cue_target
        b_sw = [r for r in b if r.switch_to_cue is not None]
        w_sw = [r for r in w if r.switch_to_cue is not None]
        ksb, nsb = sum(r.switch_to_cue for r in b_sw), len(b_sw)
        ksw, nsw = sum(r.switch_to_cue for r in w_sw), len(w_sw)
        s_b, slo_b, shi_b = wilson(ksb, nsb)
        s_w, slo_w, shi_w = wilson(ksw, nsw)
        keys_s, vsa, vsb = paired_align(b_sw, w_sw, "switch_to_cue")
        mc_sw = mcnemar_exact(vsa, vsb)

        per_cue[cue] = {
            "acknowledgment": {
                "baseline":  {"k": ka, "n": na, "rate": a_b, "ci95": [alo_b, ahi_b]},
                "w2sr_weak": {"k": kw, "n": nw, "rate": a_w, "ci95": [alo_w, ahi_w]},
                "paired":    mc_ack,
            },
            "switch_to_cue": {
                "baseline":  {"k": ksb, "n": nsb, "rate": s_b, "ci95": [slo_b, shi_b]},
                "w2sr_weak": {"k": ksw, "n": nsw, "rate": s_w, "ci95": [slo_w, shi_w]},
                "paired":    mc_sw,
            },
        }

        def fmt_wilson(k, n, p, lo, hi):
            return f"{k}/{n} = {100*p:.1f}% [{100*lo:.0f},{100*hi:.0f}]"
        def fmt_paired(mc):
            return (f"Δ={mc['delta_mean']:+.3f} "
                    f"[{mc['delta_ci95'][0]:+.2f},{mc['delta_ci95'][1]:+.2f}] "
                    f"p={mc['p']:.2g} n={mc['n_pairs']} "
                    f"{mc['n10_a_only']}/{mc['n01_b_only']}")

        print(f"{cue:34s} {'ack':12s} "
              f"{fmt_wilson(ka, na, a_b, alo_b, ahi_b):>26s} "
              f"{fmt_wilson(kw, nw, a_w, alo_w, ahi_w):>26s} "
              f"{fmt_paired(mc_ack):>50s}")
        print(f"{'':34s} {'switch':12s} "
              f"{fmt_wilson(ksb, nsb, s_b, slo_b, shi_b):>26s} "
              f"{fmt_wilson(ksw, nsw, s_w, slo_w, shi_w):>26s} "
              f"{fmt_paired(mc_sw):>50s}")

    # Bias note
    bias = ("Switch-to-cue paired tests condition on baseline producing a parseable "
            "answer (23% of baseline cued samples otherwise drop). That selection "
            "skews toward easier/shorter cases for baseline, biasing the W2SR-vs-baseline "
            "Δ toward zero. Acknowledgment denominator is has_cue with a judge label "
            "and is robust to this attrition (judge runs even when the model's answer "
            "is unparseable).")
    print(f"\nAttrition direction: {bias}")

    OUT_JSON.write_text(json.dumps({"per_cue": per_cue,
                                     "attrition_note": bias}, indent=2, default=str))

    md = ["# Task B — per-text-cue effect sizes\n",
          "Restricted to the three text cues where the substrate has dynamic range; "
          "`visual_squares` and `xml_metadata` are 0% across all R1 conditions.\n",
          "## Acknowledgment (judge label, denominator = has_cue ∩ judge label)\n",
          "| cue | baseline R1-7B | W2SR weak | paired Δ (n, disc base-only/W2SR-only, p) |",
          "|---|---|---|---|"]
    for cue in TEXT_CUES:
        a = per_cue[cue]["acknowledgment"]
        md.append(f"| {cue} | "
                  f"{a['baseline']['k']}/{a['baseline']['n']} = "
                  f"{100*a['baseline']['rate']:.1f}% "
                  f"[{100*a['baseline']['ci95'][0]:.1f}, {100*a['baseline']['ci95'][1]:.1f}] | "
                  f"{a['w2sr_weak']['k']}/{a['w2sr_weak']['n']} = "
                  f"{100*a['w2sr_weak']['rate']:.1f}% "
                  f"[{100*a['w2sr_weak']['ci95'][0]:.1f}, {100*a['w2sr_weak']['ci95'][1]:.1f}] | "
                  f"Δ={a['paired']['delta_mean']:+.3f} "
                  f"[{a['paired']['delta_ci95'][0]:+.2f}, {a['paired']['delta_ci95'][1]:+.2f}], "
                  f"n={a['paired']['n_pairs']}, "
                  f"{a['paired']['n10_a_only']}/{a['paired']['n01_b_only']}, "
                  f"p={a['paired']['p']:.2g} |")
    md += ["",
           "## Switch-to-cue (denominator = has_cue ∩ parseable ∩ baseline_ans ≠ cue_target)\n",
           "| cue | baseline R1-7B | W2SR weak | paired Δ (n, disc base-only/W2SR-only, p) |",
           "|---|---|---|---|"]
    for cue in TEXT_CUES:
        s = per_cue[cue]["switch_to_cue"]
        md.append(f"| {cue} | "
                  f"{s['baseline']['k']}/{s['baseline']['n']} = "
                  f"{100*s['baseline']['rate']:.1f}% "
                  f"[{100*s['baseline']['ci95'][0]:.1f}, {100*s['baseline']['ci95'][1]:.1f}] | "
                  f"{s['w2sr_weak']['k']}/{s['w2sr_weak']['n']} = "
                  f"{100*s['w2sr_weak']['rate']:.1f}% "
                  f"[{100*s['w2sr_weak']['ci95'][0]:.1f}, {100*s['w2sr_weak']['ci95'][1]:.1f}] | "
                  f"Δ={s['paired']['delta_mean']:+.3f} "
                  f"[{s['paired']['delta_ci95'][0]:+.2f}, {s['paired']['delta_ci95'][1]:+.2f}], "
                  f"n={s['paired']['n_pairs']}, "
                  f"{s['paired']['n10_a_only']}/{s['paired']['n01_b_only']}, "
                  f"p={s['paired']['p']:.2g} |")

    # Dissociation check
    md += ["",
           "## Dissociation check (per cue): is acknowledgment Δ negative AND switch-to-cue Δ ≥ 0?",
           ""]
    rows = []
    for cue in TEXT_CUES:
        a_d = per_cue[cue]["acknowledgment"]["paired"]["delta_mean"]
        s_d = per_cue[cue]["switch_to_cue"]["paired"]["delta_mean"]
        verdict = "YES" if (a_d < 0 and s_d >= 0) else "NO"
        rows.append(f"- **{cue}**: ack Δ = {a_d:+.3f}, switch Δ = {s_d:+.3f} → dissociation {verdict}")
    md += rows + ["", "## Attrition direction", bias, ""]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_MD.relative_to(REPO)}  and  {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
