"""Task L — budget-matched re-run of baseline vs the CoT-preserving arm at 32k.

Why this exists. Every comparison in the paper up to this point was generated
at an 8k token budget, which is not neutral between the arms: the untrained
baseline writes ~20,000-character traces and hit that ceiling on 43.8% of cued
samples, versus 21.7% for the reasoning-preserving arm. An arm that gets cut
off mid-trace cannot emit a parseable answer, so the behavioural metrics were
computed on a subset that discarded nearly half of one arm and a fifth of the
other. This task re-runs both arms at 32k, identical in every other respect,
and reports the comparison at matched budget.

What it found, in short: the missingness gap does close substantially, but the
paired influence interval does not narrow — so the behavioural null was never a
truncation artifact — and the acknowledgment effect gets *weaker* at the fairer
budget (p = 0.005 -> 0.043). The 32k figures are the better-specified ones.

Inputs (all local .eval logs):
  r1_7b_baseline      / r1_7b_baseline_32k
  r1_7b_w2sr_cotsft   / r1_7b_cotsft_32k

Outputs:
  results/reanalysis/L_budget_matched_32k.{json,md}
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    REPO, LOGS, load_records, mcnemar_exact, influenced, wilson,
)

OUT_DIR = REPO / "results/reanalysis"

ARMS = {
    "baseline_8k":  ("r1_7b_baseline",     "openai_DeepSeek-R1-Distill-Qwen-7B"),
    "baseline_32k": ("r1_7b_baseline_32k", "openai_DeepSeek-R1-Distill-Qwen-7B"),
    "cotsft_8k":    ("r1_7b_w2sr_cotsft",  "openai_w2sr_r1_7b_cotsft"),
    "cotsft_32k":   ("r1_7b_cotsft_32k",   "openai_w2sr_r1_7b_cotsft"),
}
PAIRS = [("8k", "baseline_8k", "cotsft_8k"), ("32k", "baseline_32k", "cotsft_32k")]


def cap_rate(batch: str, served: str) -> tuple[int, int]:
    """Cued samples whose generation stopped because it hit max_tokens.

    stop_reason lives under output.choices[0], NOT at the top of output — a
    top-level lookup silently returns None for every sample and reports 0%.
    """
    cap = tot = 0
    for ev in glob.glob(str(LOGS / batch / served / "*/config_*/*.eval")):
        with zipfile.ZipFile(ev) as zf:
            for name in zf.namelist():
                if not (name.startswith("samples/") and name.endswith(".json")):
                    continue
                d = json.loads(zf.read(name))
                if not (d.get("metadata") or {}).get("has_cue"):
                    continue
                ch = ((d.get("output") or {}).get("choices") or [{}])[0]
                tot += 1
                cap += ch.get("stop_reason") == "max_tokens"
    return cap, tot


def arm_summary(batch: str, served: str) -> dict:
    rows = load_records(batch, served, cued_only=True)
    acks = [r for r in rows if r.ack is not None]
    par = [r for r in rows if r.answer]
    k = sum(r.ack for r in acks)
    p, lo, hi = wilson(k, len(acks))
    cap, tot = cap_rate(batch, served)
    uncued = [r for r in load_records(batch, served, cued_only=False)
              if r.cue_dir == "baseline"]
    gradeable = [r for r in uncued if r.correct_letter]
    scored = [r for r in gradeable if r.answer]
    correct = sum(1 for r in scored if r.answer == r.correct_letter)
    return {
        "n_cued": len(rows),
        "parseable": len(par), "parseable_rate": len(par) / len(rows),
        "ack_k": k, "ack_n": len(acks), "ack_rate": p, "ack_ci95": [lo, hi],
        "capped": cap, "capped_n": tot, "capped_rate": cap / tot if tot else None,
        "median_cot_chars": int(st.median([r.cot_chars for r in rows])),
        "mean_cot_chars": int(st.mean([r.cot_chars for r in rows])),
        "uncued_n": len(gradeable), "uncued_parseable": len(scored),
        "uncued_acc_all_items": correct / len(gradeable) if gradeable else None,
        "uncued_acc_parseable_only": correct / len(scored) if scored else None,
    }


def paired(a_key: str, b_key: str, field: str) -> dict:
    """Paired McNemar on (qid, cue). delta = b - a, i.e. cotsft minus baseline."""
    A = {(r.qid, r.cue): r for r in load_records(*ARMS[a_key], cued_only=True)}
    B = {(r.qid, r.cue): r for r in load_records(*ARMS[b_key], cued_only=True)}
    va, vb = [], []
    for key in sorted(A.keys() & B.keys()):
        x = influenced(A[key]) if field == "inf" else A[key].ack
        y = influenced(B[key]) if field == "inf" else B[key].ack
        if x is None or y is None:
            continue
        va.append(x); vb.append(y)
    return mcnemar_exact(va, vb)


def main() -> None:
    out: dict = {"arms": {}, "paired": {}}
    for name, (batch, served) in ARMS.items():
        if not (LOGS / batch / served).is_dir():
            raise SystemExit(f"missing eval logs for {batch}/{served}")
        out["arms"][name] = arm_summary(batch, served)
    for tag, a, b in PAIRS:
        out["paired"][tag] = {
            "acknowledgment": paired(a, b, "ack"),
            "influence": paired(a, b, "inf"),
        }

    # Paired UNCUED accuracy at 32k, on the shared question set. This is the
    # other half of the budget correction and it cuts against us: the same
    # re-run that lifts baseline's accuracy also shows the headline arm scoring
    # BELOW baseline at matched budget. Reporting only the baseline half would
    # be selective. Unparseable counts as incorrect (Table 1's convention).
    ub = {r.qid: r for r in load_records(*ARMS["baseline_32k"], cued_only=False)
          if r.cue_dir == "baseline"}
    uc = {r.qid: r for r in load_records(*ARMS["cotsft_32k"], cued_only=False)
          if r.cue_dir == "baseline"}
    shared = sorted(ub.keys() & uc.keys())
    va = [int(bool(ub[q].answer) and ub[q].answer == ub[q].correct_letter) for q in shared]
    vc = [int(bool(uc[q].answer) and uc[q].answer == uc[q].correct_letter) for q in shared]
    out["paired_uncued_accuracy_32k"] = {
        "n_shared_questions": len(shared),
        "baseline_correct": sum(va), "cotsft_correct": sum(vc),
        "baseline_parseable": sum(1 for q in shared if ub[q].answer),
        "cotsft_parseable": sum(1 for q in shared if uc[q].answer),
        **mcnemar_exact(va, vc),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "L_budget_matched_32k.json").write_text(json.dumps(out, indent=2))

    A = out["arms"]
    md = [
        "# Task L — budget-matched re-run at 32k",
        "",
        "Both arms re-run at a 32,000-token generation budget; identical to the 8k",
        "configs in every other respect. The 8k comparison advantaged the trained arm,",
        "because only the baseline was routinely truncated before it could answer.",
        "",
        "## Per-arm (cued cells)",
        "",
        "| arm | budget | n | parseable | ack | hit cap | median CoT |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline_8k", "baseline_32k", "cotsft_8k", "cotsft_32k"):
        s = A[name]
        arm, bud = name.rsplit("_", 1)
        md.append(f"| {arm} | {bud} | {s['n_cued']} | {s['parseable_rate']:.1%} | "
                  f"{s['ack_k']}/{s['ack_n']} = {s['ack_rate']:.1%} | "
                  f"{s['capped_rate']:.1%} | {s['median_cot_chars']:,} |")

    md += ["", "## Paired, baseline vs CoT-preserving arm", "",
           "| budget | metric | n | Δ | 95% CI | p | disc (base/cotsft) |",
           "|---|---|---:|---:|---|---:|---:|"]
    for tag, _, _ in PAIRS:
        for metric in ("acknowledgment", "influence"):
            m = out["paired"][tag][metric]
            md.append(f"| {tag} | {metric} | {m['n_pairs']} | {m['delta_mean']:+.3f} | "
                      f"[{m['delta_ci95'][0]:+.3f}, {m['delta_ci95'][1]:+.3f}] | "
                      f"{m['p']:.4g} | {m['n10_a_only']}/{m['n01_b_only']} |")

    b8, b32 = A["baseline_8k"], A["baseline_32k"]
    ua = out["paired_uncued_accuracy_32k"]
    i8 = out["paired"]["8k"]["influence"]; i32 = out["paired"]["32k"]["influence"]
    a8 = out["paired"]["8k"]["acknowledgment"]; a32 = out["paired"]["32k"]["acknowledgment"]
    w8 = i8["delta_ci95"][1] - i8["delta_ci95"][0]
    w32 = i32["delta_ci95"][1] - i32["delta_ci95"][0]
    md += [
        "", "## Read", "",
        f"- **Missingness closes.** Baseline parseable {b8['parseable_rate']:.1%} -> "
        f"{b32['parseable_rate']:.1%}; hit-cap {b8['capped_rate']:.1%} -> {b32['capped_rate']:.1%}. "
        f"The across-arm parseability gap narrows from "
        f"{100*(A['cotsft_8k']['parseable_rate']-b8['parseable_rate']):.1f}pp to "
        f"{100*(A['cotsft_32k']['parseable_rate']-b32['parseable_rate']):.1f}pp.",
        f"- **The behavioural null is not a truncation artifact.** The paired influence CI is "
        f"essentially unchanged in width ({w8:.3f} -> {w32:.3f}) despite n rising "
        f"{i8['n_pairs']} -> {i32['n_pairs']}. More budget does not resolve it; only more "
        f"questions would.",
        f"- **The acknowledgment effect weakens at the fairer budget.** "
        f"Δ {a8['delta_mean']:+.3f} (p = {a8['p']:.4g}) -> {a32['delta_mean']:+.3f} "
        f"(p = {a32['p']:.4g}). Baseline ack {A['baseline_8k']['ack_rate']:.1%} -> "
        f"{A['baseline_32k']['ack_rate']:.1%}; arm {A['cotsft_8k']['ack_rate']:.1%} -> "
        f"{A['cotsft_32k']['ack_rate']:.1%}.",
        f"- **Baseline uncued accuracy was substantially a budget artifact**: "
        f"{b8['uncued_acc_all_items']:.3f} at 8k -> {b32['uncued_acc_all_items']:.3f} at 32k "
        f"(all-items convention).",
        f"- **But the same correction cuts against the trained arm.** Paired on the "
        f"{ua['n_shared_questions']} shared uncued questions at 32k, baseline scores "
        f"{ua['baseline_correct']}/{ua['n_shared_questions']} = "
        f"{ua['baseline_correct']/ua['n_shared_questions']:.3f} against the arm's "
        f"{ua['cotsft_correct']}/{ua['n_shared_questions']} = "
        f"{ua['cotsft_correct']/ua['n_shared_questions']:.3f} "
        f"(Δ = {ua['delta_mean']:+.3f} [{ua['delta_ci95'][0]:+.3f}, {ua['delta_ci95'][1]:+.3f}], "
        f"p = {ua['p']:.3f}, discordant {ua['n10_a_only']}/{ua['n01_b_only']}) — a reversal of "
        f"the 8k ordering. This is not a truncation effect: the arm is MORE parseable "
        f"({ua['cotsft_parseable']}/{ua['n_shared_questions']} vs "
        f"{ua['baseline_parseable']}/{ua['n_shared_questions']}) and still less accurate. "
        f"'Capability preserved on the eval substrate' does not survive a matched budget.",
        f"- Truncation is not eliminated: {A['baseline_32k']['capped_rate']:.1%} of baseline "
        f"cued samples still exhaust even the 32k budget.",
        "",
    ]
    (OUT_DIR / "L_budget_matched_32k.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
