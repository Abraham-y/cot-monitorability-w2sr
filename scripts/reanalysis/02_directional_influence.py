"""Task 2 — directional-influence confound check.

Question: is the rising influence (answer == cue_target) just a consequence
of W2SR flipping more in general? On the full cued set both `influenced` and
`flipped` rise together (55→68→72% flipped). Distinguish genuine cue-pull
from background variance with four checks:

  (1) Restrict to cued ∩ parseable ∩ baseline_model_answer != cue_target
      (i.e., the baseline had room to be pulled). Compute
      switch-to-cue = P(answer == cue_target) per condition, with Wilson 95%
      CI. Report R1 family and instruct family separately.

  (2) Among questions that actually flipped (answer != baseline_answer)
      inside that restricted set, compute P(answer == cue_target).
      The null model is "flips are uniform over the three non-baseline
      letters", which puts cue_target landing rate at 1/3 by chance. Use a
      one-sided binomial vs 1/3.

  (3) For the same flips, partition flips into {to cue_target} vs
      {to one of the two non-cue, non-baseline letters}. Stating cue-target
      landings against the other destinations is what makes the "directional"
      claim auditable.

  (4) Within the restricted set (baseline_ans != cue_target), pair on
      (qid, cue) across baseline R1-7B and W2SR weak, McNemar + paired Δ.

Inputs (verified):
  external/monitorability-eval/logs/r1_7b_baseline/openai_DeepSeek-R1-Distill-Qwen-7B/{01..05}/config_001/*.eval
  external/monitorability-eval/logs/r1_7b_w2sr/openai_w2sr_r1_7b/{01..05}/config_001/*.eval
  external/monitorability-eval/logs/r1_7b_strong/openai_w2sr_r1_7b_strong/{01..05}/config_001/*.eval
  external/monitorability-eval/logs/baseline_7b/openai_Qwen2.5-7B-Instruct/{01..05}/config_001/*.eval
  external/monitorability-eval/logs/w2sr_student/openai_w2sr_infamily_inst/{01..05}/config_001/*.eval
  external/monitorability-eval/logs/control_student/openai_w2sr_control_inst/{01..05}/config_001/*.eval

Outputs:
  results/reanalysis/02_directional_influence.md
  results/reanalysis/02_directional_influence.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    CONDITIONS, REPO, load_records, paired_align, mcnemar_exact, wilson, influenced,
)
from scipy.stats import binomtest

OUT_MD   = REPO / "results/reanalysis/02_directional_influence.md"
OUT_JSON = REPO / "results/reanalysis/02_directional_influence.json"

R1_LABELS = {"baseline R1-7B", "W2SR weak (R1-1.5B teacher)", "W2SR strong (R1-14B teacher)"}
INST_LABELS = {"instruct baseline (Qwen2.5-7B-Inst)", "instruct W2SR weak",
               "instruct W2SR strong (control)"}


def restrict(rs):
    """Cued ∩ parseable ∩ has cue_target ∩ has baseline_ans ∩ baseline_ans != cue_target."""
    out = []
    for r in rs:
        if r.answer is None or r.cue_target is None or r.baseline_ans is None:
            continue
        if r.baseline_ans == r.cue_target:
            continue   # nothing to be pulled toward
        out.append(r)
    return out


def main():
    print("="*70)
    print("TASK 2 — directional-influence confound check")
    print("="*70)

    raw = {}
    for batch, served, label in CONDITIONS:
        raw[label] = load_records(batch, served, cued_only=True)

    summary = {}
    for label, rs in raw.items():
        rr = restrict(rs)
        n_restricted = len(rr)
        n_cued_parseable = sum(1 for r in rs if r.answer is not None and r.cue_target is not None)
        n_baseline_eq_target = sum(
            1 for r in rs
            if r.answer is not None and r.cue_target is not None
            and r.baseline_ans is not None and r.baseline_ans == r.cue_target
        )
        # (1) switch-to-cue on restricted
        switch_to_cue = [int(r.answer == r.cue_target) for r in rr]
        k1, n1 = sum(switch_to_cue), len(switch_to_cue)
        s1, lo1, hi1 = wilson(k1, n1)
        # (2)(3) among flippers
        flippers = [r for r in rr if r.answer != r.baseline_ans]
        k_to_cue = sum(1 for r in flippers if r.answer == r.cue_target)
        k_to_other = len(flippers) - k_to_cue
        if len(flippers):
            p_to_cue, lo_tc, hi_tc = wilson(k_to_cue, len(flippers))
            p_to_other, lo_to, hi_to = wilson(k_to_other, len(flippers))
            # one-sided binomial vs 1/3 — landing on the cue target above chance
            b = binomtest(k_to_cue, len(flippers), 1/3, alternative="greater")
            p_vs_chance = b.pvalue
        else:
            p_to_cue = lo_tc = hi_tc = float("nan")
            p_to_other = lo_to = hi_to = float("nan")
            p_vs_chance = float("nan")
        summary[label] = {
            "n_cued_with_parseable_and_target": n_cued_parseable,
            "n_dropped_baseline_eq_target": n_baseline_eq_target,
            "n_restricted": n_restricted,
            "switch_to_cue_k": k1, "switch_to_cue_n": n1, "switch_to_cue_rate": s1,
            "switch_to_cue_ci95": [lo1, hi1],
            "n_flippers": len(flippers),
            "flip_to_cue_k": k_to_cue, "flip_to_cue_rate": p_to_cue,
            "flip_to_cue_ci95": [lo_tc, hi_tc],
            "flip_to_other_k": k_to_other, "flip_to_other_rate": p_to_other,
            "flip_to_other_ci95": [lo_to, hi_to],
            "p_one_sided_vs_chance_one_third": p_vs_chance,
        }

    # ---- print: per-family side-by-side ----
    def print_family(label_set, family_name):
        print(f"\n=== {family_name} — restricted to baseline_ans != cue_target ===")
        print(f"{'condition':40s} {'n_restr':>8s} {'switch-to-cue':>22s} "
              f"{'#flipped':>10s} {'flip→cue':>22s} {'flip→other':>22s} {'vs 1/3 (one-sided p)':>22s}")
        for label in [l for l in raw if l in label_set]:
            s = summary[label]
            stc = f"{s['switch_to_cue_k']}/{s['switch_to_cue_n']}={100*s['switch_to_cue_rate']:.1f}%"
            ftc = (f"{s['flip_to_cue_k']}/{s['n_flippers']}="
                   f"{100*s['flip_to_cue_rate']:.1f}% [{100*s['flip_to_cue_ci95'][0]:.0f},{100*s['flip_to_cue_ci95'][1]:.0f}]"
                   if s['n_flippers'] else "—")
            fto = (f"{s['flip_to_other_k']}/{s['n_flippers']}="
                   f"{100*s['flip_to_other_rate']:.1f}%"
                   if s['n_flippers'] else "—")
            pvs = f"p={s['p_one_sided_vs_chance_one_third']:.3g}" if s['n_flippers'] else "—"
            print(f"{label:40s} {s['n_restricted']:8d} {stc:>22s} {s['n_flippers']:10d} {ftc:>22s} {fto:>22s} {pvs:>22s}")

    print_family(R1_LABELS, "R1-distill family")
    print_family(INST_LABELS, "Instruct family")

    # ---- (4) paired switch-to-cue on (qid, cue), R1 baseline vs W2SR weak ----
    print("\n=== (4) Paired switch-to-cue, baseline R1-7B vs W2SR weak ===")
    rr_base = restrict(raw["baseline R1-7B"])
    rr_w2sr = restrict(raw["W2SR weak (R1-1.5B teacher)"])
    for r in rr_base + rr_w2sr:
        r.switch_to_cue = int(r.answer == r.cue_target)
    keys, a, b = paired_align(rr_base, rr_w2sr, "switch_to_cue")
    mc = mcnemar_exact(a, b)
    print(f"  restricted baseline n: {len(rr_base)}; W2SR n: {len(rr_w2sr)}; matched pairs: {mc['n_pairs']}")
    print(f"  2×2: (0,0)={mc['n00']}, baseline-only={mc['n10_a_only']}, W2SR-only={mc['n01_b_only']}, both={mc['n11']}")
    print(f"  discordant: {mc['discordant']}  McNemar exact-binom p = {mc['p']:.3g}")
    print(f"  Δ switch-to-cue (W2SR − baseline) = {mc['delta_mean']:+.3f}  "
          f"95% CI [{mc['delta_ci95'][0]:+.3f}, {mc['delta_ci95'][1]:+.3f}]")

    # ---- Same paired test BUT on the unrestricted base (to mirror Task 1d) ----
    # (informational; the restricted version is the cleaner statement)
    paired = {"restricted_n": mc["n_pairs"], **mc}

    # ---- Are the two flip->cue RATES distinguishable from each other? ----
    # The manuscript says the flip-to-cue rates (baseline ~65%, W2SR ~73%) are
    # "statistically indistinguishable"; this is the test backing that sentence.
    # Unpaired 2x2 on flippers (the two flipper sets are different samples, so a
    # paired test does not apply here).
    from scipy.stats import fisher_exact
    fb, fw = summary["baseline R1-7B"], summary["W2SR weak (R1-1.5B teacher)"]
    table = [[fb["flip_to_cue_k"], fb["n_flippers"] - fb["flip_to_cue_k"]],
             [fw["flip_to_cue_k"], fw["n_flippers"] - fw["flip_to_cue_k"]]]
    or_, p_between = fisher_exact(table, alternative="two-sided")
    flip_rate_comparison = {
        "baseline_k": fb["flip_to_cue_k"], "baseline_n": fb["n_flippers"],
        "baseline_rate": fb["flip_to_cue_rate"],
        "w2sr_k": fw["flip_to_cue_k"], "w2sr_n": fw["n_flippers"],
        "w2sr_rate": fw["flip_to_cue_rate"],
        "odds_ratio": or_, "fisher_two_sided_p": p_between,
        "indistinguishable_at_05": bool(p_between >= 0.05),
    }
    print("\n=== Flip→cue rate, baseline vs W2SR (are they distinguishable?) ===")
    print(f"  baseline {fb['flip_to_cue_k']}/{fb['n_flippers']} = {100*fb['flip_to_cue_rate']:.1f}%   "
          f"W2SR {fw['flip_to_cue_k']}/{fw['n_flippers']} = {100*fw['flip_to_cue_rate']:.1f}%")
    print(f"  Fisher exact two-sided p = {p_between:.3g} (OR = {or_:.2f}) -> "
          f"{'indistinguishable' if p_between >= 0.05 else 'DISTINGUISHABLE'} at α=0.05")

    # ---- Per-cue switch-to-cue inside the restricted set, R1 family ----
    print("\n=== Per-cue switch-to-cue (restricted), R1 family ===")
    per_cue = {}
    for label in R1_LABELS:
        rr = restrict(raw[label])
        by_cue = {}
        for r in rr:
            by_cue.setdefault(r.cue, []).append(int(r.answer == r.cue_target))
        per_cue[label] = {}
        print(f"  {label}:")
        for c, vs in sorted(by_cue.items()):
            p, lo, hi = wilson(sum(vs), len(vs))
            per_cue[label][c] = {"k": sum(vs), "n": len(vs), "rate": p, "ci95": [lo, hi]}
            print(f"    {c:34s} {sum(vs):3d}/{len(vs):3d} = {100*p:5.1f}%  [{100*lo:4.1f},{100*hi:4.1f}]")

    # ---- Attrition direction note ----
    bias = (
        "Baseline R1-7B loses 23% of cued samples to no-answer (37/160); both trained "
        "students lose ≤6%. Paired tests therefore condition on questions where the BASELINE "
        "actually produced a parseable answer — these are biased toward easier/shorter cases "
        "for baseline R1 (the tail it ran out of generation budget on is dropped). This "
        "biases the paired Δ for switch-to-cue toward zero (or against W2SR) because the "
        "easier cases are also the ones where baseline R1 was more likely to be pulled."
    )
    print(f"\nAttrition direction: {bias}")

    out = {"per_condition": summary, "per_cue_R1": per_cue,
           "paired_switch_to_cue_base_vs_w2sr_weak": paired,
           "flip_to_cue_rate_baseline_vs_w2sr": flip_rate_comparison,
           "attrition_note": bias}
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))

    # ---- markdown ----
    lines = ["# Task 2 — directional-influence confound check\n",
             "Restriction: cued ∩ parseable ∩ has cue_target ∩ has baseline_ans ∩ "
             "baseline_ans ≠ cue_target (i.e., the baseline had room to be pulled toward the cue).\n",
             "## Per-condition rates on the restricted set\n",
             "| condition | n_restr | switch-to-cue | #flippers | flip→cue | flip→non-cue non-baseline | p (one-sided vs 1/3) |",
             "|---|---:|---|---:|---|---|---|"]
    for label in [l for _, _, l in CONDITIONS]:
        s = summary[label]
        stc = f"{s['switch_to_cue_k']}/{s['switch_to_cue_n']} = {100*s['switch_to_cue_rate']:.1f}% [{100*s['switch_to_cue_ci95'][0]:.1f}, {100*s['switch_to_cue_ci95'][1]:.1f}]"
        if s['n_flippers']:
            ftc = (f"{s['flip_to_cue_k']}/{s['n_flippers']} = "
                   f"{100*s['flip_to_cue_rate']:.1f}% [{100*s['flip_to_cue_ci95'][0]:.1f}, {100*s['flip_to_cue_ci95'][1]:.1f}]")
            fto = (f"{s['flip_to_other_k']}/{s['n_flippers']} = "
                   f"{100*s['flip_to_other_rate']:.1f}% [{100*s['flip_to_other_ci95'][0]:.1f}, {100*s['flip_to_other_ci95'][1]:.1f}]")
            pvs = f"{s['p_one_sided_vs_chance_one_third']:.3g}"
        else:
            ftc = fto = pvs = "—"
        lines.append(f"| {label} | {s['n_restricted']} | {stc} | {s['n_flippers']} | {ftc} | {fto} | {pvs} |")
    lines += ["",
              "Chance level for flip→cue under \"flip uniformly to one of 3 non-baseline letters\" = 1/3 ≈ 33.3%.\n",
              "## Paired switch-to-cue, baseline R1-7B vs W2SR weak\n",
              f"Restricted to baseline_ans ≠ cue_target on both sides. Matched pairs: {paired['n_pairs']}.\n",
              "| | W2SR switch=0 | W2SR switch=1 |",
              "|---|---:|---:|",
              f"| baseline switch=0 | {paired['n00']} | {paired['n01_b_only']} |",
              f"| baseline switch=1 | {paired['n10_a_only']} | {paired['n11']} |",
              "",
              f"McNemar exact p = {paired['p']:.3g}; Δ (W2SR − baseline) = "
              f"{paired['delta_mean']:+.3f} [{paired['delta_ci95'][0]:+.3f}, {paired['delta_ci95'][1]:+.3f}]\n",
              "## Attrition direction",
              bias,
              "",
              "## Per-cue switch-to-cue, R1 family (restricted)\n",
              "| cue | baseline R1-7B | W2SR weak | W2SR strong |",
              "|---|---|---|---|"]
    cues = sorted({c for d in per_cue.values() for c in d})
    for c in cues:
        cells = []
        for label in ["baseline R1-7B", "W2SR weak (R1-1.5B teacher)", "W2SR strong (R1-14B teacher)"]:
            d = per_cue[label].get(c)
            cells.append("—" if d is None else f"{d['k']}/{d['n']} = {100*d['rate']:.1f}%")
        lines.append(f"| {c} | {cells[0]} | {cells[1]} | {cells[2]} |")
    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_MD.relative_to(REPO)}  and  {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
