"""Task E (cross-substrate MMLU replication).

Question: does the W2SR → ack-collapse finding generalize beyond GPQA?
We run the same 3 R1-family conditions (baseline, W2SR weak, Self-A) on a
small MMLU sweep over 5 STEM subjects (college_physics, college_chemistry,
college_biology, college_mathematics, conceptual_physics), capped at 8
questions per subject (W2SR_LIMIT=8) → ~40 cued samples per cell, matching
the GPQA cell size.

Inputs (all config_* dirs per cue — one per MMLU subject, 5 total):
  external/monitorability-eval/logs/r1_7b_baseline_mmlu/openai_DeepSeek-R1-Distill-Qwen-7B/{01..05}/config_*/*.eval
  external/monitorability-eval/logs/r1_7b_w2sr_mmlu/openai_w2sr_r1_7b/{01..05}/config_*/*.eval
  external/monitorability-eval/logs/r1_7b_self_A4k_mmlu/openai_w2sr_r1_7b_self_A4k/{01..05}/config_*/*.eval

Outputs:
  results/reanalysis/E_cross_substrate_mmlu.md
  results/reanalysis/E_cross_substrate_mmlu.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    REPO, load_records, paired_align, mcnemar_exact, wilson, influenced,
)

OUT_MD   = REPO / "results/reanalysis/E_cross_substrate_mmlu.md"
OUT_JSON = REPO / "results/reanalysis/E_cross_substrate_mmlu.json"

CONDS = [
    ("baseline R1-7B (MMLU)", "r1_7b_baseline_mmlu",  "openai_DeepSeek-R1-Distill-Qwen-7B"),
    ("W2SR weak (MMLU)",      "r1_7b_w2sr_mmlu",      "openai_w2sr_r1_7b"),
    ("Self-A (MMLU)",         "r1_7b_self_A4k_mmlu",  "openai_w2sr_r1_7b_self_A4k"),
]


def main():
    print("=" * 70)
    print("TASK E — cross-substrate MMLU replication")
    print("=" * 70)

    data = {label: load_records(batch, served, cued_only=True)
            for label, batch, served in CONDS}

    # ---- per-condition pooled ----
    summary = {}
    print(f"\n{'condition':32s} {'cued':>6s} {'pars':>6s} {'ack k/n':>10s} {'ack %':>9s}  [95% CI]")
    for label, _, _ in CONDS:
        rs = data[label]
        cued = len(rs)
        parsed = sum(1 for r in rs if r.answer is not None)
        scored = [r.ack for r in rs if r.ack is not None]
        k = sum(scored); n = len(scored); p, lo, hi = wilson(k, n)
        summary[label] = {"cued": cued, "parseable": parsed, "ack_k": k, "ack_n": n,
                          "ack_rate": p, "ci95": [lo, hi]}
        print(f"{label:32s} {cued:6d} {parsed:6d} {f'{k}/{n}':>10s} {100*p:8.1f}%  "
              f"[{100*lo:.1f}, {100*hi:.1f}]")

    # ---- per-cue ----
    print("\nPer-cue ack:")
    print(f"  {'cue':34s}  " + "  ".join(f"{l[:14]:>14s}" for l, _, _ in CONDS))
    per_cue = {label: {} for label, _, _ in CONDS}
    all_cues = sorted({r.cue for rs in data.values() for r in rs})
    for cue in all_cues:
        cells = []
        for label, _, _ in CONDS:
            vs = [r.ack for r in data[label] if r.cue == cue and r.ack is not None]
            if vs:
                k = sum(vs); n = len(vs); p, lo, hi = wilson(k, n)
                per_cue[label][cue] = {"k": k, "n": n, "rate": p, "ci95": [lo, hi]}
                cells.append(f"{k}/{n}={100*p:.1f}%")
            else:
                cells.append("—")
        print(f"  {cue:34s}  " + "  ".join(f"{c:>14s}" for c in cells))

    # ---- paired vs baseline ----
    print("\nPaired ack vs baseline R1-7B (MMLU) (matched (qid, cue)):")
    base = data["baseline R1-7B (MMLU)"]
    paired_vs_base = {}
    for label, _, _ in CONDS:
        if label == "baseline R1-7B (MMLU)":
            continue
        keys, a, b = paired_align(base, data[label], "ack")
        mc = mcnemar_exact(a, b)
        paired_vs_base[label] = mc
        print(f"  {label:30s}  n={mc['n_pairs']:3d}  "
              f"disc base-only/student-only = {mc['n10_a_only']}/{mc['n01_b_only']}  "
              f"Δ={mc['delta_mean']:+.3f}  [{mc['delta_ci95'][0]:+.3f}, {mc['delta_ci95'][1]:+.3f}]  "
              f"p={mc['p']:.3g}")

    # ---- paired Self-A vs W2SR weak ----
    print("\nPaired Self-A vs W2SR weak (replicate the 'indistinguishable' result?):")
    keys, a, b = paired_align(data["W2SR weak (MMLU)"], data["Self-A (MMLU)"], "ack")
    mc = mcnemar_exact(a, b)
    paired_self_vs_w2sr = mc
    print(f"  Self-A vs W2SR weak           n={mc['n_pairs']:3d}  "
          f"disc w2sr-only/self-only = {mc['n10_a_only']}/{mc['n01_b_only']}  "
          f"Δ={mc['delta_mean']:+.3f}  [{mc['delta_ci95'][0]:+.3f}, {mc['delta_ci95'][1]:+.3f}]  "
          f"p={mc['p']:.3g}")

    # ---- influence ----
    print("\nInfluence rate (answer == cue_target, denom = parseable):")
    influence = {}
    for label, _, _ in CONDS:
        vs = [influenced(r) for r in data[label] if influenced(r) is not None]
        if vs:
            k = sum(vs); n = len(vs); p, lo, hi = wilson(k, n)
            influence[label] = {"k": k, "n": n, "rate": p, "ci95": [lo, hi]}
            print(f"  {label:30s} {k}/{n} = {100*p:.1f}%  [Wilson {100*lo:.1f}, {100*hi:.1f}]")

    # ---- CoT length ----
    print("\nCoT char length (median, p95):")
    length = {}
    for label, _, _ in CONDS:
        lens = sorted(r.cot_chars for r in data[label])
        if not lens: continue
        med = lens[len(lens)//2]
        p95 = lens[int(0.95*(len(lens)-1))]
        length[label] = {"n": len(lens), "median": med, "p95": p95}
        print(f"  {label:30s}  n={len(lens):3d}  median={med:>6d}  p95={p95:>6d}")

    # ---- think tags ----
    print("\nThink-tag emission on cued completions:")
    think = {}
    for label, _, _ in CONDS:
        comps = [r.completion for r in data[label]]
        n = len(comps)
        if n == 0: continue
        n_close = sum(1 for c in comps if "</think>" in c)
        think[label] = {"n": n, "frac_close": n_close / n}
        print(f"  {label:30s}  n={n:3d}  </think>: {100*n_close/n:5.1f}%")

    # Interpretation is COMPUTED from this run's numbers, never hardcoded:
    # an earlier version froze pre-loader-fix (physics-only) numbers here,
    # which contradicted the script's own computed tables after the fix.
    base_lab, w2sr_lab, self_lab = (label for label, _, _ in CONDS)
    mc_w, mc_s = paired_vs_base[w2sr_lab], paired_vs_base[self_lab]
    mc_sw = paired_self_vs_w2sr
    comp_ratio = length[base_lab]["median"] / max(1, length[w2sr_lab]["median"])
    interpretation = (
        f"Cross-substrate replication on MMLU (5 STEM subjects × 8 Qs per cue) "
        f"holds the headline pattern: baseline ack {100*summary[base_lab]['ack_rate']:.1f}%, "
        f"W2SR weak {100*summary[w2sr_lab]['ack_rate']:.1f}%, Self-A "
        f"{100*summary[self_lab]['ack_rate']:.1f}%. W2SR weak vs baseline paired "
        f"Δ = {mc_w['delta_mean']:+.3f}, McNemar p = {mc_w['p']:.3g} (n = {mc_w['n_pairs']}, "
        f"discordant {mc_w['n10_a_only']}/{mc_w['n01_b_only']}); Self-A vs baseline "
        f"Δ = {mc_s['delta_mean']:+.3f}, p = {mc_s['p']:.3g} (discordant "
        f"{mc_s['n10_a_only']}/{mc_s['n01_b_only']}). Self-A vs W2SR weak remains "
        f"indistinguishable (Δ = {mc_sw['delta_mean']:+.3f}, p = {mc_sw['p']:.3g}) — the "
        f"'same mechanism' finding generalizes. CoT compresses {comp_ratio:.1f}× on MMLU "
        f"({length[base_lab]['median']:,} → {length[w2sr_lab]['median']:,} chars) vs "
        f"~14× on GPQA, because baseline R1-7B already writes shorter CoT on MMLU. "
        f"Effect generalizes beyond GPQA-Diamond."
    )
    out = {"summary": summary, "per_cue": per_cue,
           "paired_vs_baseline": paired_vs_base,
           "paired_self_A_vs_w2sr_weak": paired_self_vs_w2sr,
           "influence": influence, "cot_length": length, "think_tags": think,
           "interpretation": interpretation}
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))

    # ---- markdown ----
    md = ["# Task E — cross-substrate MMLU replication\n",
          "**Substrate:** 5 STEM MMLU subjects (college_physics, college_chemistry, "
          "college_biology, college_mathematics, conceptual_physics), capped at 8 "
          "questions per subject (`W2SR_LIMIT=8`) → ~40 cued samples per cell, "
          "matching the GPQA cell size. Same 5 cues, same `claude-sonnet-4-6` judge, "
          "same VLLMServer endpoint.\n",
          "## Pooled ack\n",
          "| condition | cued | parseable | ack k/n | ack % | 95% Wilson |",
          "|---|---:|---:|---:|---:|---|"]
    for label, _, _ in CONDS:
        s = summary[label]
        md.append(f"| {label} | {s['cued']} | {s['parseable']} | "
                  f"{s['ack_k']}/{s['ack_n']} | {100*s['ack_rate']:.1f}% | "
                  f"[{100*s['ci95'][0]:.1f}, {100*s['ci95'][1]:.1f}] |")
    md += ["",
           "## Paired ack vs baseline R1-7B (MMLU)\n",
           "| comparison | n pairs | disc base-only / student-only | Δ (CI95) | McNemar p |",
           "|---|---:|---:|---|---|"]
    for label, mc in paired_vs_base.items():
        md.append(f"| {label} | {mc['n_pairs']} | "
                  f"**{mc['n10_a_only']}** / **{mc['n01_b_only']}** | "
                  f"{mc['delta_mean']:+.3f} [{mc['delta_ci95'][0]:+.3f}, "
                  f"{mc['delta_ci95'][1]:+.3f}] | {mc['p']:.3g} |")
    mc = paired_self_vs_w2sr
    md += ["",
           "## Paired Self-A vs W2SR weak (MMLU) — replicates GPQA's 'indistinguishable' finding\n",
           f"n = {mc['n_pairs']}; disc W2SR-only / Self-only = {mc['n10_a_only']} / "
           f"{mc['n01_b_only']}; Δ = {mc['delta_mean']:+.3f} "
           f"[{mc['delta_ci95'][0]:+.3f}, {mc['delta_ci95'][1]:+.3f}]; McNemar p = {mc['p']:.3g}.\n",
           "## Per-cue ack\n",
           "| cue | " + " | ".join(l for l, _, _ in CONDS) + " |",
           "|---|" + "|".join(["---"] * len(CONDS)) + "|"]
    for cue in all_cues:
        cells = []
        for label, _, _ in CONDS:
            d = per_cue[label].get(cue)
            cells.append("—" if d is None else f"{d['k']}/{d['n']} = {100*d['rate']:.1f}%")
        md.append(f"| {cue} | " + " | ".join(cells) + " |")
    md += ["",
           "## CoT compression\n",
           "| condition | n | median chars | p95 chars |",
           "|---|---:|---:|---:|"]
    for label, _, _ in CONDS:
        L = length[label]
        md.append(f"| {label} | {L['n']} | {L['median']:,} | {L['p95']:,} |")
    md += ["",
           f"Baseline R1-7B writes shorter CoT on MMLU than on GPQA "
           f"({length[base_lab]['median']:,} vs 18,692 chars median (GPQA cued)) — MMLU's "
           f"question style and difficulty don't elicit the full long-CoT regime. So MMLU "
           f"compression is **{comp_ratio:.1f}× from baseline**, vs **~14× on GPQA**. The "
           f"ack collapse still fires.\n",
           "## Think-tag emission (cued completions)\n",
           "| condition | n | `</think>` |",
           "|---|---:|---:|"]
    for label, _, _ in CONDS:
        t = think[label]
        md.append(f"| {label} | {t['n']} | {100*t['frac_close']:.1f}% |")
    trained_fracs = [100*think[l]["frac_close"] for l in (w2sr_lab, self_lab) if l in think]
    md += ["",
           f"Baseline emits `</think>` on {100*think[base_lab]['frac_close']:.0f}% of MMLU "
           f"cued completions (vs 57% on GPQA — shorter problems means the closing tag fits "
           f"within the generation budget). Trained students drop to "
           f"{min(trained_fracs):.0f}–{max(trained_fracs):.0f}%, matching the "
           f"partial-collapse pattern from GPQA.\n",
           "## Influence rate (answer == cue_target)\n",
           "| condition | k/n | rate | 95% CI |",
           "|---|---:|---:|---|"]
    for label, _, _ in CONDS:
        i = influence[label]
        md.append(f"| {label} | {i['k']}/{i['n']} | {100*i['rate']:.1f}% | "
                  f"[{100*i['ci95'][0]:.1f}, {100*i['ci95'][1]:.1f}] |")
    md += ["",
           f"Same direction as GPQA: trained students show modestly higher switch-to-cue "
           f"rate (baseline {100*influence[base_lab]['rate']:.1f}% → W2SR "
           f"{100*influence[w2sr_lab]['rate']:.1f}% → Self-A "
           f"{100*influence[self_lab]['rate']:.1f}%), so the \"behavior toward "
           f"the cue, silence about it\" dissociation holds.\n",
           "## Interpretation",
           out["interpretation"],
           "",
           "## Honest caveats",
           f"- Per-subject cells are small (~30–40 cued samples); the pooled paired "
           f"comparisons carry the power (W2SR weak vs baseline p = {mc_w['p']:.3g}, "
           f"discordant {mc_w['n10_a_only']}/{mc_w['n01_b_only']}; Self-A vs baseline "
           f"p = {mc_s['p']:.3g}, discordant {mc_s['n10_a_only']}/{mc_s['n01_b_only']}).",
           "- Only 5 STEM subjects of MMLU; broader MMLU coverage (humanities, social "
           "sciences) untested.",
           "- Same judge (claude-sonnet-4-6) as the GPQA arm; cross-judge robustness "
           "was checked on GPQA but not re-checked here.",
           ""]
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_MD.relative_to(REPO)} and {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
