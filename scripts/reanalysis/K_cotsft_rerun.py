#!/usr/bin/env python3
"""Task K — CoT-preserving SFT rerun of the W2SR-weak arm.

The paper's R1-substrate arms were trained on answer-only supervision (the
R1-Distill chat template stripped the reasoning span; disclosed in the paper).
This task evaluates the SAME recipe re-run with the fixed renderer
(src/train_student.py build_sft_text), i.e. supervision that preserves the
teacher CoT, and asks: does acknowledgment still collapse?

Conditions:
  r1_7b_w2sr_cotsft — R1-7B student, R1-1.5B teacher traces, CoT-preserving SFT
  vs r1_7b_baseline (paired on qid, cue) and vs r1_7b_w2sr (original arm).

Outputs: results/reanalysis/K_cotsft_rerun.{json,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    REPO, load_records, wilson, mcnemar_exact, paired_align, influenced,
    Record,
)

OUT_DIR = REPO / "results/reanalysis"

COTSFT = ("r1_7b_w2sr_cotsft", "openai_w2sr_r1_7b_cotsft", "W2SR weak, CoT-preserving SFT")
BASELINE = ("r1_7b_baseline", "openai_DeepSeek-R1-Distill-Qwen-7B", "baseline R1-7B")
ORIG = ("r1_7b_w2sr", "openai_w2sr_r1_7b", "W2SR weak, answer-only SFT (paper)")


def arm_summary(rows: list[Record]) -> dict:
    acks = [r for r in rows if r.ack is not None]
    k = sum(r.ack for r in acks)
    p, lo, hi = wilson(k, len(acks))
    infl = [influenced(r) for r in rows]
    infl = [x for x in infl if x is not None]
    lens = sorted(r.cot_chars for r in rows)
    per_cue = {}
    for cue in sorted({r.cue for r in rows}):
        sub = [r for r in rows if r.cue == cue and r.ack is not None]
        per_cue[cue] = {"ack": sum(r.ack for r in sub), "n": len(sub)}
    return {
        "ack_k": k, "ack_n": len(acks), "ack_rate": p, "ack_ci95": [lo, hi],
        "influence_rate": (sum(infl) / len(infl)) if infl else None,
        "influence_n": len(infl),
        "median_cot_chars": lens[len(lens) // 2] if lens else None,
        "per_cue_ack": per_cue,
    }


def uncued_accuracy(batch: str, served: str, restrict_qids: set[str] | None = None) -> dict:
    """Uncued GPQA accuracy under BOTH denominator conventions.

    `accuracy` (primary) counts an unparseable completion as incorrect, which is
    the convention Table 1 of the manuscript uses.
    `accuracy_parseable_only` divides by the parseable subset instead.

    The conventions are not interchangeable here, and the difference is not
    cosmetic: the CoT-preserving arm writes ~10x longer completions and so hits
    the generation cap without ever emitting an answer on 27.5% of its matched-40
    uncued items (34.3% over the full 198),
    while the answer-only arm does so on 5.0% and the untrained baseline --- the
    longest writer of all --- on 50.0%. Scoring over the parseable subset
    therefore discards each arm's own failed generations in proportion to how
    long it writes, which flatters the long-CoT arms and — because the untrained
    baseline is the longest writer of all — flatters it most. Under
    parseable-only the baseline scores 11/20 = 0.550 and outranks every trained
    arm; under the all-items convention it is 11/40 = 0.275 and ranks last.
    (Values under the shipped extractor; with LOOSE_FALLBACK_IS_AN_ANSWER=True
    they read 0.615 and 0.400.)
    Report the all-items number; keep the other for transparency.
    """
    rows = [r for r in load_records(batch, served, cued_only=False)
            if r.cue_dir == "baseline"]
    if restrict_qids is not None:
        rows = [r for r in rows if r.qid in restrict_qids]
    gradeable = [r for r in rows if r.correct_letter]
    scored = [r for r in gradeable if r.answer is not None]
    correct = sum(1 for r in scored if r.answer == r.correct_letter)
    return {"n_total": len(rows), "n_scored": len(scored),
            "correct": correct,
            "accuracy": correct / len(gradeable) if gradeable else None,
            "accuracy_parseable_only": correct / len(scored) if scored else None,
            "qids": sorted(r.qid for r in rows)}


def main() -> None:
    cot = load_records(COTSFT[0], COTSFT[1])
    base = load_records(BASELINE[0], BASELINE[1])
    orig = load_records(ORIG[0], ORIG[1])
    if not cot:
        raise SystemExit("no r1_7b_w2sr_cotsft records — run the eval first")

    out: dict = {"conditions": {}}
    for (batch, served, label), rows in [(COTSFT, cot), (BASELINE, base), (ORIG, orig)]:
        out["conditions"][label] = arm_summary(rows)

    # Paired tests on acknowledgment and influence.
    for name, other, other_label in [("cotsft_vs_baseline", base, BASELINE[2]),
                                     ("cotsft_vs_orig_w2sr", orig, ORIG[2])]:
        keys, a, b = paired_align(other, cot, "ack")
        res = mcnemar_exact(a, b)  # delta = cotsft - other
        # influence pairing: build per-record influenced dicts
        oth_inf = [r for r in other if influenced(r) is not None]
        cot_inf = [r for r in cot if influenced(r) is not None]
        ik, ia, ib = paired_align(
            [Record(**{**r.__dict__, "ack": influenced(r)}) for r in oth_inf],
            [Record(**{**r.__dict__, "ack": influenced(r)}) for r in cot_inf],
            "ack")
        ires = mcnemar_exact(ia, ib)
        out[name] = {"other": other_label, "ack_paired": res,
                     "influence_paired": ires}

    # Uncued accuracy. Our uncued cell ran the full 198; the paper's numbers
    # are n=40 (first-40 limit). Report full + the subset matched to the
    # original arm's 40 uncued qids.
    orig_unc = uncued_accuracy(ORIG[0], ORIG[1])
    matched = uncued_accuracy(COTSFT[0], COTSFT[1],
                              restrict_qids=set(orig_unc["qids"]))
    full = uncued_accuracy(COTSFT[0], COTSFT[1])
    orig_unc.pop("qids"); matched.pop("qids"); full.pop("qids")
    out["uncued_accuracy"] = {
        "cotsft_full_diamond": full,
        "cotsft_matched_to_paper_n40": matched,
        "orig_w2sr_n40": orig_unc,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "K_cotsft_rerun.json").write_text(json.dumps(out, indent=2))

    c = out["conditions"]
    vb = out["cotsft_vs_baseline"]["ack_paired"]
    vo = out["cotsft_vs_orig_w2sr"]["ack_paired"]
    cs, bs, os_ = (c[COTSFT[2]], c[BASELINE[2]], c[ORIG[2]])
    md = [
        "# Task K — CoT-preserving SFT rerun (W2SR weak arm)",
        "",
        "| condition | ack | influence | median CoT (chars) |",
        "|---|---|---|---|",
    ]
    for label in (BASELINE[2], ORIG[2], COTSFT[2]):
        s = c[label]
        md.append(f"| {label} | {s['ack_k']}/{s['ack_n']} = {s['ack_rate']:.1%} "
                  f"| {s['influence_rate']:.1%} (n={s['influence_n']}) "
                  f"| {s['median_cot_chars']:,} |")
    md += [
        "",
        f"Paired ack, cotsft − baseline: Δ = {vb['delta_mean']:+.3f} "
        f"[{vb['delta_ci95'][0]:+.3f}, {vb['delta_ci95'][1]:+.3f}], "
        f"n = {vb['n_pairs']}, disc {vb['n10_a_only']}/{vb['n01_b_only']} "
        f"(baseline-only/cotsft-only), p = {vb['p']:.2e}",
        f"Paired ack, cotsft − original W2SR: Δ = {vo['delta_mean']:+.3f} "
        f"[{vo['delta_ci95'][0]:+.3f}, {vo['delta_ci95'][1]:+.3f}], "
        f"n = {vo['n_pairs']}, disc {vo['n10_a_only']}/{vo['n01_b_only']} "
        f"(orig-only/cotsft-only), p = {vo['p']:.2e}",
        "",
        "Uncued GPQA accuracy (all-items convention: unparseable counts as "
        "incorrect; parseable-only shown in parentheses). "
        f"cotsft {out['uncued_accuracy']['cotsft_matched_to_paper_n40']['accuracy']:.3f} "
        f"over n={out['uncued_accuracy']['cotsft_matched_to_paper_n40']['n_total']} "
        f"({out['uncued_accuracy']['cotsft_matched_to_paper_n40']['accuracy_parseable_only']:.3f} "
        f"over {out['uncued_accuracy']['cotsft_matched_to_paper_n40']['n_scored']} parseable); "
        f"full-diamond {out['uncued_accuracy']['cotsft_full_diamond']['accuracy']:.3f} "
        f"over n={out['uncued_accuracy']['cotsft_full_diamond']['n_total']}; "
        f"original W2SR {out['uncued_accuracy']['orig_w2sr_n40']['accuracy']:.3f} "
        f"(n={out['uncued_accuracy']['orig_w2sr_n40']['n_scored']}).",
        "",
        "Per-cue ack (cotsft vs baseline vs original):",
        "",
        "| cue | baseline | orig W2SR | cotsft |",
        "|---|---|---|---|",
    ]
    for cue in sorted(cs["per_cue_ack"]):
        row = [cue]
        for s in (bs, os_, cs):
            pc = s["per_cue_ack"].get(cue, {"ack": 0, "n": 0})
            row.append(f"{pc['ack']}/{pc['n']}")
        md.append("| " + " | ".join(row) + " |")
    (OUT_DIR / "K_cotsft_rerun.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
