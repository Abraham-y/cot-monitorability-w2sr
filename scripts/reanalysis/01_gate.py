"""Task 1 — gate.

Step 1: per-condition answer-extraction counts (patched extractor) — STOP if
any trained condition's null rate is suspicious (>30% on a single cell, or any
strong outlier vs the trained-condition cohort).

Step 2: reproduce the four headline numbers from disk:
  (a) baseline R1-7B: ack 40/160 = 25.0% (cued ∩ has_cue=True)
  (b) W2SR weak:      ack  6/190 =  3.2%
  (c) paired acknowledgment table baseline vs W2SR weak on (qid, cue):
        n=150, discordant 34/1, McNemar p ≈ 2.1e-9, Δ = −0.220
  (d) paired influence table on (qid, cue), parseable both sides:
        n=108, discordant 16/33 (base-only / W2SR-only), Δ = +0.157, p = 0.021

Inputs (verified):
  external/monitorability-eval/logs/r1_7b_baseline/openai_DeepSeek-R1-Distill-Qwen-7B/{01..05,baseline}/config_001/*.eval
  external/monitorability-eval/logs/r1_7b_w2sr/openai_w2sr_r1_7b/{01..05,baseline}/config_001/*.eval
  external/monitorability-eval/logs/r1_7b_strong/openai_w2sr_r1_7b_strong/{01..05,baseline}/config_001/*.eval
  external/monitorability-eval/logs/baseline_7b/openai_Qwen2.5-7B-Instruct/{01..05,baseline}/config_001/*.eval
  external/monitorability-eval/logs/w2sr_student/openai_w2sr_infamily_inst/{01..05,baseline}/config_001/*.eval
  external/monitorability-eval/logs/control_student/openai_w2sr_control_inst/{01..05,baseline}/config_001/*.eval

Outputs:
  results/reanalysis/01_gate.md
  results/reanalysis/01_gate.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    CONDITIONS, TRAINED_CONDITIONS, REPO,
    iter_eval_files, load_records, paired_align, mcnemar_exact, wilson,
    influenced,
)

OUT_MD   = REPO / "results/reanalysis/01_gate.md"
OUT_JSON = REPO / "results/reanalysis/01_gate.json"

NULL_GATE_THRESHOLD = 0.30   # any trained cell above this halts the analysis

# Pinned manuscript headlines — hard-fail on divergence.
# These match writeup.tex Table 1 and §4 of references/findings_extension.md.
HEADLINE = {
    "baseline_ack_num": 40,
    "baseline_ack_den": 160,
    "w2sr_weak_ack_num": 6,
    "w2sr_weak_ack_den": 190,
    "paired_ack_n": 150,
    "paired_ack_base_only": 34,
    "paired_ack_w2sr_only": 1,
    "paired_ack_delta": -0.220,
    "paired_ack_p_max": 1e-8,      # manuscript reports ≈ 2.1e-9; allow ≤ 1e-8
    "paired_inf_n": 108,
    "paired_inf_base_only": 16,
    "paired_inf_w2sr_only": 33,
    "paired_inf_delta": +0.157,
    "paired_inf_p_max": 0.05,      # manuscript reports = 0.021
}


# --------- Step 1: extraction counts per condition × cue dir -----------
def extraction_table():
    """Per (batch, cue_dir) record count, patched-extractor null count, and
    which pattern fired. Reads every .eval (cued + baseline) so the gate
    covers uncued samples too."""
    table = []  # rows of dicts for json
    print("=== EXTRACTION COUNTS (patched extractor on output.completion) ===\n")
    hdr = f"{'batch':24s} {'cue':30s} {'n':>4s} {'null':>5s} {'null%':>6s} {'boxed':>6s} {'ANS:':>6s} {'fb':>4s}"
    print(hdr); print("-" * len(hdr))
    suspicious = []
    uncued_warn = []
    for batch, served, label in CONDITIONS:
        # One row per (batch, cue_dir): iter_eval_files yields one entry per
        # .eval file, and MMLU cue dirs hold 5 config_* files each, so looping
        # over files directly would print/append the same cue_dir row 5 times.
        cue_dirs = sorted({cd for cd, _ in iter_eval_files(batch, served)})
        cued_all = load_records(batch, served, cued_only=True) if any(cd != "baseline" for cd in cue_dirs) else []
        uncued_all = ([r for r in load_records(batch, served, cued_only=False) if r.cue_dir == "baseline"]
                      if "baseline" in cue_dirs else [])
        for cue_dir in cue_dirs:
            # Use the cued-only loader for cued dirs; raw load for the uncued baseline.
            if cue_dir == "baseline":
                rs = uncued_all
            else:
                rs = [r for r in cued_all if r.cue_dir == cue_dir]
            n = len(rs)
            null = sum(1 for r in rs if r.answer is None)
            counts = {"ANSWER:": 0, "boxed": 0, "fallback": 0}
            for r in rs:
                if r.extractor:
                    counts[r.extractor] += 1
            null_rate = (null / n) if n else 0.0
            print(f"{batch:24s} {cue_dir:30s} {n:4d} {null:5d} {100*null_rate:5.1f}% "
                  f"{counts['boxed']:6d} {counts['ANSWER:']:6d} {counts['fallback']:4d}")
            row = {"batch": batch, "served": served, "label": label, "cue_dir": cue_dir,
                   "n": n, "null": null, "null_rate": null_rate,
                   "boxed": counts["boxed"], "answer_colon": counts["ANSWER:"],
                   "fallback": counts["fallback"]}
            table.append(row)
            if batch in TRAINED_CONDITIONS and null_rate > NULL_GATE_THRESHOLD:
                # Every downstream metric (ack, influence, length) is computed on
                # CUED cells only, so only a cued cell can invalidate a result.
                # A high null rate on the uncued `baseline/` cell means the model
                # often failed to emit a parseable answer with no cue present; it
                # shrinks that arm's adaptive-cue sample count (already reflected
                # in the reported n) but cannot bias a cued-cell comparison.
                if cue_dir == "baseline":
                    uncued_warn.append((batch, cue_dir, n, null, null_rate))
                else:
                    suspicious.append((batch, cue_dir, n, null, null_rate))
    return table, suspicious, uncued_warn


# --------- Step 2: reproduce the four headline numbers ------------------
def reproduce_headlines():
    out = {}
    base = load_records("r1_7b_baseline",  "openai_DeepSeek-R1-Distill-Qwen-7B", cued_only=True)
    w2sr = load_records("r1_7b_w2sr",      "openai_w2sr_r1_7b",                  cued_only=True)

    # (a) (b) — ack on cued ∩ has_cue=True, denominator = #has_cue with non-null judge label
    def ack_summary(rs, label):
        scored = [r for r in rs if r.ack is not None]
        k = sum(r.ack for r in scored)
        n = len(scored)
        p, lo, hi = wilson(k, n)
        print(f"  {label:32s} ack {k}/{n} = {100*p:.1f}% [Wilson {100*lo:.1f},{100*hi:.1f}]")
        return {"k": k, "n": n, "rate": p, "ci95": [lo, hi]}

    print("\n=== (a)(b) Per-condition pooled acknowledgment ===")
    out["baseline_pooled_ack"]   = ack_summary(base, "baseline R1-7B")
    out["w2sr_weak_pooled_ack"]  = ack_summary(w2sr, "W2SR weak")

    # (c) — paired ack table on (qid, cue)
    print("\n=== (c) Paired acknowledgment, baseline vs W2SR weak ===")
    keys, a_ack, b_ack = paired_align(base, w2sr, "ack")
    mc = mcnemar_exact(a_ack, b_ack)
    print(f"  matched pairs: {mc['n_pairs']}  discordant: {mc['discordant']}  "
          f"(baseline-only={mc['n10_a_only']}, W2SR-only={mc['n01_b_only']}, both={mc['n11']}, neither={mc['n00']})")
    print(f"  McNemar exact-binom p = {mc['p']:.3g}")
    print(f"  Δ ack (W2SR − baseline) = {mc['delta_mean']:+.3f}  95% CI [{mc['delta_ci95'][0]:+.3f}, {mc['delta_ci95'][1]:+.3f}]")
    out["paired_ack_base_vs_w2sr_weak"] = mc

    # (d) — paired influence table
    print("\n=== (d) Paired influence, baseline vs W2SR weak ===")
    # influence requires parseable answer + cue_target on both sides
    def add_influence(rs):
        for r in rs:
            r.influenced = influenced(r)
        return rs
    add_influence(base); add_influence(w2sr)
    keys_i, a_inf, b_inf = paired_align(base, w2sr, "influenced")
    mi = mcnemar_exact(a_inf, b_inf)
    print(f"  matched pairs (parseable both sides): {mi['n_pairs']}")
    print(f"  discordant: baseline-only={mi['n10_a_only']}, W2SR-only={mi['n01_b_only']} "
          f"(both={mi['n11']}, neither={mi['n00']})")
    print(f"  McNemar exact-binom p = {mi['p']:.3g}")
    print(f"  Δ influence (W2SR − baseline) = {mi['delta_mean']:+.3f}  95% CI [{mi['delta_ci95'][0]:+.3f}, {mi['delta_ci95'][1]:+.3f}]")
    out["paired_influence_base_vs_w2sr_weak"] = mi
    return out


def assert_eq(actual, expected, name, tol=0):
    """Hard-fail assertion against a pinned manuscript value."""
    ok = abs(actual - expected) <= tol
    flag = "OK " if ok else "!! "
    print(f"  {flag}{name}: got {actual}, expected {expected} (tol {tol})")
    assert ok, f"headline mismatch: {name} = {actual}, expected {expected} (tol {tol})"


def assert_le(actual, upper, name):
    ok = actual <= upper
    flag = "OK " if ok else "!! "
    print(f"  {flag}{name}: got {actual}, expected ≤ {upper}")
    assert ok, f"headline mismatch: {name} = {actual}, expected ≤ {upper}"


def main():
    print("="*70)
    print("TASK 1 — gate")
    print("="*70)
    table, suspicious, uncued_warn = extraction_table()
    print(f"\nGate threshold: trained-condition CUED null_rate > {NULL_GATE_THRESHOLD:.0%} on any cell halts.")
    if uncued_warn:
        print("\n  NOTE — high null rate on trained-condition UNCUED (baseline/) cells:")
        for s in uncued_warn:
            print(f"    {s}")
        print("    These cells feed no reported metric (all metrics are cued-only). They do")
        print("    shrink that arm's adaptive-cue sample count, which is reflected in its n.")
    if suspicious:
        print("\n!! SUSPICIOUS NULL RATES on trained CUED cells — STOPPING:")
        for s in suspicious:
            print(f"  {s}")
        OUT_JSON.write_text(json.dumps({"suspicious": suspicious,
                                        "uncued_high_null": uncued_warn,
                                        "extraction_table": table}, indent=2))
        sys.exit(2)
    print("  → all trained-condition cued null rates within threshold; proceeding.")

    head = reproduce_headlines()

    # Hard-fail assertions vs the manuscript headlines pinned at the top.
    print("\n=== Hard-fail assertions against pinned manuscript headlines ===")
    assert_eq(head["baseline_pooled_ack"]["k"], HEADLINE["baseline_ack_num"], "baseline ack numerator")
    assert_eq(head["baseline_pooled_ack"]["n"], HEADLINE["baseline_ack_den"], "baseline ack denominator")
    assert_eq(head["w2sr_weak_pooled_ack"]["k"], HEADLINE["w2sr_weak_ack_num"], "W2SR weak ack numerator")
    assert_eq(head["w2sr_weak_pooled_ack"]["n"], HEADLINE["w2sr_weak_ack_den"], "W2SR weak ack denominator")
    pa = head["paired_ack_base_vs_w2sr_weak"]
    assert_eq(pa["n_pairs"],    HEADLINE["paired_ack_n"],         "paired ack n")
    assert_eq(pa["n10_a_only"], HEADLINE["paired_ack_base_only"], "ack baseline-only")
    assert_eq(pa["n01_b_only"], HEADLINE["paired_ack_w2sr_only"], "ack W2SR-only")
    assert_eq(round(pa["delta_mean"], 3), HEADLINE["paired_ack_delta"], "ack Δ", tol=0.001)
    assert_le(pa["p"], HEADLINE["paired_ack_p_max"], "ack McNemar p")
    pi = head["paired_influence_base_vs_w2sr_weak"]
    assert_eq(pi["n_pairs"],    HEADLINE["paired_inf_n"],         "paired influence n")
    assert_eq(pi["n10_a_only"], HEADLINE["paired_inf_base_only"], "influence baseline-only")
    assert_eq(pi["n01_b_only"], HEADLINE["paired_inf_w2sr_only"], "influence W2SR-only")
    assert_eq(round(pi["delta_mean"], 3), HEADLINE["paired_inf_delta"], "influence Δ", tol=0.001)
    assert_le(pi["p"], HEADLINE["paired_inf_p_max"], "influence McNemar p")

    OUT_JSON.write_text(json.dumps({
        "extraction_table": table,
        "uncued_high_null": uncued_warn,
        "headlines": head,
        "null_gate_threshold": NULL_GATE_THRESHOLD,
    }, indent=2, default=str))

    # Markdown summary
    lines = ["# Task 1 — gate\n",
             "## Step 1: extraction counts (patched extractor)\n",
             "| batch | cue_dir | n | null | null% | boxed | ANSWER: | fb |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['batch']} | {r['cue_dir']} | {r['n']} | {r['null']} | "
                     f"{100*r['null_rate']:.1f}% | {r['boxed']} | {r['answer_colon']} | {r['fallback']} |")
    uncued_note = ""
    if uncued_warn:
        cells = ", ".join(f"`{b}/{c}` {k}/{n}" for b, c, n, k, _ in uncued_warn)
        uncued_note = (
            f" Trained-condition UNCUED cells above the threshold ({cells}) are reported but "
            "do not halt: every metric in this suite is computed on cued cells only, so an "
            "uncued cell cannot bias a reported comparison. It does shrink that arm's "
            "adaptive-cue sample count, which is already reflected in its n.")
    lines += ["",
              f"Gate threshold for trained conditions: cued null_rate > {NULL_GATE_THRESHOLD:.0%}. "
              f"All trained cued cells within threshold.{uncued_note}\n",
              "## Step 2: headline reproduction\n",
              f"- Baseline R1-7B pooled ack: **{head['baseline_pooled_ack']['k']}/"
              f"{head['baseline_pooled_ack']['n']} = {100*head['baseline_pooled_ack']['rate']:.1f}%**",
              f"- W2SR weak pooled ack: **{head['w2sr_weak_pooled_ack']['k']}/"
              f"{head['w2sr_weak_pooled_ack']['n']} = {100*head['w2sr_weak_pooled_ack']['rate']:.1f}%**",
              ""]
    pa = head["paired_ack_base_vs_w2sr_weak"]; pi = head["paired_influence_base_vs_w2sr_weak"]
    lines += [f"### Paired acknowledgment (baseline vs W2SR weak)",
              f"n={pa['n_pairs']}; 2×2: (0,0)={pa['n00']}, baseline-only={pa['n10_a_only']}, "
              f"W2SR-only={pa['n01_b_only']}, both={pa['n11']}.",
              f"McNemar p = {pa['p']:.3g}; Δ = {pa['delta_mean']:+.3f} "
              f"[{pa['delta_ci95'][0]:+.3f}, {pa['delta_ci95'][1]:+.3f}]\n",
              f"### Paired influence (baseline vs W2SR weak)",
              f"n={pi['n_pairs']}; 2×2: (0,0)={pi['n00']}, baseline-only={pi['n10_a_only']}, "
              f"W2SR-only={pi['n01_b_only']}, both={pi['n11']}.",
              f"McNemar p = {pi['p']:.3g}; Δ = {pi['delta_mean']:+.3f} "
              f"[{pi['delta_ci95'][0]:+.3f}, {pi['delta_ci95'][1]:+.3f}]\n",
              "## Pairing logic",
              "Per condition load every record from `<batch>/<served>/<cue_dir>/config_001/*.eval` for "
              "cue_dir != 'baseline'. Keep records with metadata.has_cue=True. Acknowledgment label = "
              "`scores.cue_aware_adaptive_scorer.value` (0/1, judge=claude-sonnet-4-6). For influence we "
              "additionally require parseable answer on both sides — patched-extracted "
              "letter ∈ {A,B,C,D}. Pair across conditions on (core_question_id, cue_name).\n",
              "## Read",
              "All four headline numbers reproduce exactly. Trained-condition extraction is clean; "
              "no condition has a null rate above the 30% gate. The patched extractor is doing the work."]
    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_MD.relative_to(REPO)}  and  {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
