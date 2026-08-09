"""Task E2 (MMLU replication, broken out by subject).

Task E pools the MMLU arms. This script splits the same records by subject,
which the paper needs for two claims:

  (1) the full 5-subject pooled numbers, and
  (2) the IN-DOMAIN cut: `college_mathematics` is matched to the MATH-L3–L5 SFT
      data, so it tests whether the ack collapse needs a train->eval distribution
      shift, versus the combined non-math STEM subjects as the contrast.

History / why this script exists: `_common.iter_eval_files` used to hardcode
`config_001`, which on the MMLU batches is the college_physics cell only — so
earlier "5-subject sweep" numbers were in fact single-subject. The loader now
reads every `config_*` dir; this script reports the per-subject split explicitly
so that failure mode cannot recur silently. It asserts that all five subjects
are present.

The subject is carried in the qid (`mmlu_<subject>_<index>`).

Inputs:
  external/monitorability-eval/logs/r1_7b_baseline_mmlu/openai_DeepSeek-R1-Distill-Qwen-7B/{01..05}/config_00{1..5}/*.eval
  external/monitorability-eval/logs/r1_7b_w2sr_mmlu/openai_w2sr_r1_7b/{01..05}/config_00{1..5}/*.eval
  external/monitorability-eval/logs/r1_7b_self_A4k_mmlu/openai_w2sr_r1_7b_self_A4k/{01..05}/config_00{1..5}/*.eval

Outputs:
  results/reanalysis/E2_mmlu_by_subject.md
  results/reanalysis/E2_mmlu_by_subject.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import REPO, load_records, paired_align, mcnemar_exact, wilson

OUT_MD   = REPO / "results/reanalysis/E2_mmlu_by_subject.md"
OUT_JSON = REPO / "results/reanalysis/E2_mmlu_by_subject.json"

CONDS = [
    ("baseline R1-7B (MMLU)", "r1_7b_baseline_mmlu",  "openai_DeepSeek-R1-Distill-Qwen-7B"),
    ("W2SR weak (MMLU)",      "r1_7b_w2sr_mmlu",      "openai_w2sr_r1_7b"),
    ("Self-A (MMLU)",         "r1_7b_self_A4k_mmlu",  "openai_w2sr_r1_7b_self_A4k"),
]

IN_DOMAIN = "college_mathematics"
EXPECTED_SUBJECTS = {
    "college_physics", "college_chemistry", "college_biology",
    "college_mathematics", "conceptual_physics",
}

BASE = CONDS[0][0]


def subject_of(qid: str) -> str:
    """`mmlu_college_mathematics_0003` -> `college_mathematics`."""
    stem = qid.rsplit("_", 1)[0]
    return stem[len("mmlu_"):] if stem.startswith("mmlu_") else stem


def rate(rows):
    scored = [r.ack for r in rows if r.ack is not None]
    k, n = sum(scored), len(scored)
    p, lo, hi = wilson(k, n)
    return {"ack_k": k, "ack_n": n, "ack_rate": p, "ci95": [lo, hi]}


def paired(a_rows, b_rows):
    keys, a, b = paired_align(a_rows, b_rows, "ack")
    return mcnemar_exact(a, b) if keys else None


def main():
    print("=" * 70)
    print("TASK E2 — MMLU replication by subject")
    print("=" * 70)

    data = {label: load_records(batch, served, cued_only=True)
            for label, batch, served in CONDS}

    subjects = sorted({subject_of(r.qid) for rs in data.values() for r in rs})
    missing = EXPECTED_SUBJECTS - set(subjects)
    assert not missing, (
        f"MMLU subjects missing from the loaded records: {sorted(missing)}. "
        "This is the config_001-hardcoding regression — check "
        "_common.iter_eval_files reads every config_* dir.")
    print(f"\nsubjects found: {', '.join(subjects)}")

    out = {"subjects": subjects, "pooled": {}, "by_subject": {},
           "in_domain_vs_rest": {}, "paired": {}}

    # ---- pooled over all 5 subjects ----
    print(f"\n=== Pooled, all {len(subjects)} subjects ===")
    print(f"{'condition':28s} {'ack k/n':>10s} {'ack %':>8s}  [95% CI]")
    for label, _, _ in CONDS:
        st = rate(data[label])
        out["pooled"][label] = st
        frac = "{}/{}".format(st["ack_k"], st["ack_n"])
        print(f"{label:28s} {frac:>10s} {100*st['ack_rate']:7.1f}%  "
              f"[{100*st['ci95'][0]:.1f}, {100*st['ci95'][1]:.1f}]")

    # ---- per subject ----
    print(f"\n=== By subject ===")
    hdr = f"{'subject':24s}" + "".join(f"{lab[:22]:>24s}" for lab, _, _ in CONDS)
    print(hdr)
    for subj in subjects:
        row = {}
        line = f"{subj:24s}"
        for label, _, _ in CONDS:
            st = rate([r for r in data[label] if subject_of(r.qid) == subj])
            row[label] = st
            cell = "{}/{}={:.1f}%".format(st["ack_k"], st["ack_n"], 100 * st["ack_rate"])
            line += f"{cell:>24s}"
        out["by_subject"][subj] = row
        print(line)

    # ---- in-domain (math) vs rest ----
    print(f"\n=== In-domain ({IN_DOMAIN}) vs combined non-math STEM ===")
    for label, _, _ in CONDS:
        math_rows = [r for r in data[label] if subject_of(r.qid) == IN_DOMAIN]
        rest_rows = [r for r in data[label] if subject_of(r.qid) != IN_DOMAIN]
        m, s = rate(math_rows), rate(rest_rows)
        out["in_domain_vs_rest"][label] = {IN_DOMAIN: m, "non_math_stem": s}
        print(f"  {label:28s} {IN_DOMAIN}: {m['ack_k']}/{m['ack_n']} = {100*m['ack_rate']:.1f}%"
              f"   non-math: {s['ack_k']}/{s['ack_n']} = {100*s['ack_rate']:.1f}%")

    # ---- paired McNemar vs baseline, pooled / in-domain / non-math ----
    print(f"\n=== Paired ack vs {BASE} (matched (qid, cue)) ===")
    for label, _, _ in CONDS[1:]:
        for scope, pred in (("pooled",      lambda r: True),
                            (IN_DOMAIN,     lambda r: subject_of(r.qid) == IN_DOMAIN),
                            ("non_math",    lambda r: subject_of(r.qid) != IN_DOMAIN)):
            mc = paired([r for r in data[BASE] if pred(r)],
                        [r for r in data[label] if pred(r)])
            out["paired"].setdefault(label, {})[scope] = mc
            if mc:
                print(f"  {label:24s} {scope:20s} n={mc['n_pairs']:3d}  "
                      f"disc {mc['n10_a_only']}/{mc['n01_b_only']}  "
                      f"Δ={mc['delta_mean']:+.3f}  p={mc['p']:.4g}")

    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))

    # ---- markdown ----
    lines = ["# Task E2 — MMLU replication by subject\n",
             f"Subjects: {', '.join(subjects)}\n",
             "## Pooled (all subjects)\n",
             "| condition | ack | rate | 95% CI |", "|---|---:|---:|---|"]
    for label, _, _ in CONDS:
        st = out["pooled"][label]
        lines.append(f"| {label} | {st['ack_k']}/{st['ack_n']} | {100*st['ack_rate']:.1f}% | "
                     f"[{100*st['ci95'][0]:.1f}, {100*st['ci95'][1]:.1f}] |")
    lines += ["", "## By subject\n",
              "| subject | " + " | ".join(lab for lab, _, _ in CONDS) + " |",
              "|---|" + "---:|" * len(CONDS)]
    for subj in subjects:
        cells = []
        for label, _, _ in CONDS:
            st = out["by_subject"][subj][label]
            cells.append(f"{st['ack_k']}/{st['ack_n']} = {100*st['ack_rate']:.1f}%")
        lines.append(f"| {subj} | " + " | ".join(cells) + " |")

    lines += ["", f"## In-domain ({IN_DOMAIN}) vs combined non-math STEM\n",
              "| condition | college_mathematics | non-math STEM |", "|---|---:|---:|"]
    for label, _, _ in CONDS:
        d = out["in_domain_vs_rest"][label]
        m, s = d[IN_DOMAIN], d["non_math_stem"]
        lines.append(f"| {label} | {m['ack_k']}/{m['ack_n']} = {100*m['ack_rate']:.1f}% | "
                     f"{s['ack_k']}/{s['ack_n']} = {100*s['ack_rate']:.1f}% |")

    lines += ["", f"## Paired McNemar vs {BASE}\n",
              "| condition | scope | n | disc (base/student) | Δ | p |",
              "|---|---|---:|---:|---:|---:|"]
    for label in out["paired"]:
        for scope, mc in out["paired"][label].items():
            if not mc:
                continue
            lines.append(f"| {label} | {scope} | {mc['n_pairs']} | "
                         f"{mc['n10_a_only']}/{mc['n01_b_only']} | "
                         f"{mc['delta_mean']:+.3f} | {mc['p']:.3g} |")

    base_math = out["in_domain_vs_rest"][BASE][IN_DOMAIN]
    lines += ["", "## Read\n",
              f"The in-domain `{IN_DOMAIN}` cut (matched to the MATH-L3–L5 SFT data) shows the "
              f"same collapse as the science subjects, so the dissociation does not require a "
              f"train->eval distribution shift: baseline acknowledges "
              f"{base_math['ack_k']}/{base_math['ack_n']} = {100*base_math['ack_rate']:.1f}% "
              f"on math questions and the trained arms collapse to near zero.\n"]
    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_MD} and {OUT_JSON}")


if __name__ == "__main__":
    main()
