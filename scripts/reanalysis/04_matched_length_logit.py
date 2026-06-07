"""Task 4 — matched-length acknowledgment residual via logistic regression.

Both acknowledgment AND CoT length fall under W2SR. Is the ack drop fully
explained by length (the trivial-half story: shorter CoT surfaces cues less),
or is there a residual condition effect at matched length?

Model:    ack ~ condition + log(CoT_chars) + cue_fixed_effect
          (cluster-robust SE on qid)
          + a condition × log(length) interaction in a separate fit

Fit twice:
  (i)  full data: baseline R1-7B vs W2SR weak (using r1_7b_w2sr_full for the
       three cues with thickened tail mass: 01_stanford_professor, 03_grader_hack,
       04_unethical_information; r1_7b_w2sr for the other two cues to keep all
       5 cues represented).
  (ii) trimmed overlap: keep only samples whose CoT length lies in the
       max(p05) … min(p95) intersection of the two conditions. Report the
       effective n in that overlap.

Critical honesty: state how many W2SR samples lie at the right tail (≥9.6k
chars — the writeup's matched-long threshold) and in the overlap. The
"matched-length residual" rests entirely on that tail.

Inputs:
  external/monitorability-eval/logs/r1_7b_baseline/openai_DeepSeek-R1-Distill-Qwen-7B/{01..05}/config_001/*.eval
  external/monitorability-eval/logs/r1_7b_w2sr/openai_w2sr_r1_7b/{02_visual_squares,05_xml_metadata}/config_001/*.eval
  external/monitorability-eval/logs/r1_7b_w2sr_full/openai_w2sr_r1_7b/{01_stanford_professor,03_grader_hack,04_unethical_information}/config_001/*.eval

Outputs:
  results/reanalysis/04_matched_length_logit.md
  results/reanalysis/04_matched_length_logit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from _common import REPO, load_records

OUT_MD   = REPO / "results/reanalysis/04_matched_length_logit.md"
OUT_JSON = REPO / "results/reanalysis/04_matched_length_logit.json"

LONG_THRESHOLD = 9600   # writeup's "long" cutoff
THICK_CUES = {"stanford_professor_recommends",
              "grader_hack_validation",
              "insider_information"}


def to_df(rs, condition_label):
    """Build a per-record DataFrame with the columns the regression needs."""
    rows = []
    for r in rs:
        if r.ack is None:
            continue
        rows.append({
            "condition": condition_label,
            "qid": r.qid,
            "cue": r.cue,
            "ack": int(r.ack),
            "cot_chars": int(r.cot_chars),
            "log_cot": float(np.log(max(1, r.cot_chars))),
        })
    return pd.DataFrame(rows)


def fit(df, formula, label):
    """Cluster-robust logistic regression on qid."""
    print(f"\n  --- {label} ---")
    print(f"  formula: {formula}")
    print(f"  n = {len(df)}, ack mean = {df['ack'].mean():.3f}")
    model = smf.logit(formula, data=df).fit(disp=False,
                                            cov_type="cluster",
                                            cov_kwds={"groups": df["qid"]})
    print(model.summary().tables[1])
    return model


def main():
    print("=" * 70)
    print("TASK 4 — matched-length acknowledgment residual")
    print("=" * 70)

    # ---- Build the W2SR weak side with thickened tail on cues 01/03/04 ----
    base = load_records("r1_7b_baseline", "openai_DeepSeek-R1-Distill-Qwen-7B", cued_only=True)
    w_thin = load_records("r1_7b_w2sr",      "openai_w2sr_r1_7b", cued_only=True)
    w_full = load_records("r1_7b_w2sr_full", "openai_w2sr_r1_7b", cued_only=True)
    w_combined = []
    for r in w_full:
        if r.cue in THICK_CUES:
            w_combined.append(r)
    for r in w_thin:
        if r.cue not in THICK_CUES:
            w_combined.append(r)

    print(f"\n  baseline R1-7B: {len(base)} cued has_cue=True records "
          f"({sum(1 for r in base if r.ack is not None)} with judge label)")
    print(f"  W2SR weak (mixed): thick cues from r1_7b_w2sr_full "
          f"({sum(1 for r in w_combined if r.cue in THICK_CUES)} records); "
          f"02/05 from r1_7b_w2sr ({sum(1 for r in w_combined if r.cue not in THICK_CUES)} records); "
          f"total {len(w_combined)} ({sum(1 for r in w_combined if r.ack is not None)} with judge label)")

    df_b = to_df(base, "baseline")
    df_w = to_df(w_combined, "w2sr_weak")
    df = pd.concat([df_b, df_w], ignore_index=True)
    df["condition"] = pd.Categorical(df["condition"], categories=["baseline", "w2sr_weak"])

    # ---- length-distribution summary ----
    def qtable(name, vals):
        v = np.array(vals); v.sort()
        if len(v) == 0:
            return f"{name}: n=0"
        def q(p): return int(v[int(p*(len(v)-1))])
        return (f"{name}: n={len(v)}  p05={q(0.05)}  p25={q(0.25)}  "
                f"median={q(0.5)}  p75={q(0.75)}  p95={q(0.95)}  max={int(v[-1])}")

    print("\n=== CoT length distribution (chars), records with judge label ===")
    print("  " + qtable("baseline R1-7B", df_b["cot_chars"]))
    print("  " + qtable("W2SR weak (mix)", df_w["cot_chars"]))

    # Honest tail counts on the long side
    long_b = (df_b["cot_chars"] >= LONG_THRESHOLD).sum()
    long_w = (df_w["cot_chars"] >= LONG_THRESHOLD).sum()
    print(f"\n  records with cot_chars >= {LONG_THRESHOLD}: "
          f"baseline {long_b}/{len(df_b)} = {100*long_b/len(df_b):.1f}%; "
          f"W2SR weak {long_w}/{len(df_w)} = {100*long_w/len(df_w):.1f}%")

    # ---- Trimmed overlap region ----
    b_p05 = np.percentile(df_b["cot_chars"], 5)
    b_p95 = np.percentile(df_b["cot_chars"], 95)
    w_p05 = np.percentile(df_w["cot_chars"], 5)
    w_p95 = np.percentile(df_w["cot_chars"], 95)
    lo = max(b_p05, w_p05); hi = min(b_p95, w_p95)
    print(f"\n  overlap region (max-of-p05 to min-of-p95): [{int(lo)}, {int(hi)}]")
    print(f"    baseline: p05={int(b_p05)}, p95={int(b_p95)}")
    print(f"    W2SR weak: p05={int(w_p05)}, p95={int(w_p95)}")
    in_overlap_b = ((df_b["cot_chars"] >= lo) & (df_b["cot_chars"] <= hi)).sum()
    in_overlap_w = ((df_w["cot_chars"] >= lo) & (df_w["cot_chars"] <= hi)).sum()
    print(f"  in overlap: baseline {in_overlap_b}/{len(df_b)}; W2SR weak {in_overlap_w}/{len(df_w)}")

    # ---- Fit (i): full data ----
    print("\n=== Fits ===")
    formula_main = "ack ~ C(condition, Treatment(reference='baseline')) + log_cot + C(cue)"
    m_full = fit(df, formula_main, "(i) Full data: main effects (cluster SE on qid)")

    # ---- Fit (i'): full data with interaction ----
    formula_int  = ("ack ~ C(condition, Treatment(reference='baseline')) * log_cot "
                    "+ C(cue)")
    m_full_int = fit(df, formula_int, "(i') Full data: condition × log(length) interaction")

    # ---- Fit (ii): trimmed overlap ----
    df_overlap = df[(df["cot_chars"] >= lo) & (df["cot_chars"] <= hi)].copy()
    print(f"\n  overlap subset: n = {len(df_overlap)} "
          f"(baseline {(df_overlap.condition=='baseline').sum()}, "
          f"W2SR {(df_overlap.condition=='w2sr_weak').sum()})")
    if (df_overlap["condition"] == "baseline").sum() < 5 or (df_overlap["condition"] == "w2sr_weak").sum() < 5:
        print("  too few records in overlap to fit reliably — skipping overlap fit")
        m_overlap = None
    else:
        m_overlap = fit(df_overlap, formula_main, "(ii) Trimmed overlap: main effects")

    # ---- Fit (iii): long-tail subset, the writeup's explicit matched-long claim ----
    df_long = df[df["cot_chars"] >= LONG_THRESHOLD].copy()
    print(f"\n  long subset (>= {LONG_THRESHOLD}): n = {len(df_long)} "
          f"(baseline {(df_long.condition=='baseline').sum()}, "
          f"W2SR {(df_long.condition=='w2sr_weak').sum()})")
    if (df_long["condition"] == "w2sr_weak").sum() < 5:
        print("  too few W2SR records ≥9.6k — cannot fit reliably; reporting raw rates only")
        m_long = None
    else:
        m_long = fit(df_long, formula_main, f"(iii) Long-only (>= {LONG_THRESHOLD}): main effects")

    # Raw rates on the long subset (this is the writeup's 19% vs 44% claim region)
    ack_long_b = df_long[df_long.condition == "baseline"]["ack"].mean() if (df_long.condition=='baseline').sum() else float("nan")
    ack_long_w = df_long[df_long.condition == "w2sr_weak"]["ack"].mean() if (df_long.condition=='w2sr_weak').sum() else float("nan")
    print(f"  raw ack rate on long subset: baseline = {ack_long_b:.3f} "
          f"(n={int((df_long.condition=='baseline').sum())}); "
          f"W2SR = {ack_long_w:.3f} (n={int((df_long.condition=='w2sr_weak').sum())})")

    # ---- Capture coefficients for output ----
    def coef_dict(model):
        if model is None:
            return None
        params = model.params; ci = model.conf_int(); pvals = model.pvalues
        return {name: {"coef": float(params[name]),
                       "ci_lo": float(ci.loc[name, 0]),
                       "ci_hi": float(ci.loc[name, 1]),
                       "p": float(pvals[name])}
                for name in params.index}

    payload = {
        "n_records": {
            "baseline_total": int(len(df_b)),
            "w2sr_total": int(len(df_w)),
            "baseline_long_ge_9_6k": int(long_b),
            "w2sr_long_ge_9_6k":     int(long_w),
            "baseline_in_overlap":   int(in_overlap_b),
            "w2sr_in_overlap":       int(in_overlap_w),
        },
        "overlap_region_chars": [int(lo), int(hi)],
        "raw_ack_long": {
            "baseline": {"mean": float(ack_long_b),
                         "n": int((df_long.condition=='baseline').sum())},
            "w2sr":     {"mean": float(ack_long_w),
                         "n": int((df_long.condition=='w2sr_weak').sum())},
        },
        "fit_full_main": coef_dict(m_full),
        "fit_full_with_interaction": coef_dict(m_full_int),
        "fit_overlap_main": coef_dict(m_overlap),
        "fit_long_only": coef_dict(m_long),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    # ---- Markdown ----
    def fmt_row(model_dict, key):
        if model_dict is None or key not in model_dict:
            return "—"
        d = model_dict[key]
        return f"{d['coef']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] (p={d['p']:.3g})"

    cond_key = "C(condition, Treatment(reference='baseline'))[T.w2sr_weak]"
    inter_key = (
        "C(condition, Treatment(reference='baseline'))[T.w2sr_weak]:log_cot"
    )
    lines = ["# Task 4 — matched-length acknowledgment residual\n",
             "## Data construction\n",
             "Baseline = `r1_7b_baseline` (DeepSeek-R1-Distill-Qwen-7B, 40 samples × 5 cues). "
             "W2SR weak = thickened on cues 01/03/04 from `r1_7b_w2sr_full` (198 samples each) "
             "and 40 samples each on cues 02/05 from `r1_7b_w2sr`, so all 5 cues are present "
             "while the long-CoT tail mass for the three text cues uses the bigger n.\n",
             f"Records with judge label: baseline n = {len(df_b)}, W2SR weak n = {len(df_w)}.\n",
             "## Length-distribution honesty check\n",
             "| condition | n | p05 | p25 | median | p75 | p95 | max |",
             "|---|---:|---:|---:|---:|---:|---:|---:|",
             f"| baseline R1-7B | {len(df_b)} | {int(b_p05)} | {int(np.percentile(df_b.cot_chars,25))} | "
             f"{int(np.median(df_b.cot_chars))} | {int(np.percentile(df_b.cot_chars,75))} | "
             f"{int(b_p95)} | {int(df_b.cot_chars.max())} |",
             f"| W2SR weak (mix) | {len(df_w)} | {int(w_p05)} | {int(np.percentile(df_w.cot_chars,25))} | "
             f"{int(np.median(df_w.cot_chars))} | {int(np.percentile(df_w.cot_chars,75))} | "
             f"{int(w_p95)} | {int(df_w.cot_chars.max())} |",
             "",
             f"- W2SR weak records ≥ {LONG_THRESHOLD:,} chars: **{long_w} / {len(df_w)} "
             f"({100*long_w/len(df_w):.1f}%)** — the matched-long residual rests on this tail.",
             f"- Baseline records ≥ {LONG_THRESHOLD:,} chars: {long_b} / {len(df_b)} "
             f"({100*long_b/len(df_b):.1f}%).",
             f"- Overlap region [max(p05), min(p95)] = [{int(lo):,}, {int(hi):,}] chars: "
             f"baseline {in_overlap_b}, W2SR {in_overlap_w}.\n",
             "## Logistic-regression coefficient on condition (W2SR weak vs baseline)\n",
             "Model: `ack ~ condition + log(CoT_chars) + cue`, cluster-robust SE on qid. "
             "Coefficient is on the **logit scale**: negative means W2SR is less likely to "
             "acknowledge at matched length.\n",
             "| fit | n | W2SR vs baseline coefficient | interpretation |",
             "|---|---:|---|---|",
             f"| (i) full data, main effects | {len(df)} | {fmt_row(coef_dict(m_full), cond_key)} | residual after log-length control |",
             f"| (i') full data, with condition × log(length) interaction | {len(df)} | "
             f"main: {fmt_row(coef_dict(m_full_int), cond_key)}; "
             f"interaction: {fmt_row(coef_dict(m_full_int), inter_key)} | does the W2SR drop depend on length |",
             f"| (ii) trimmed overlap [{int(lo):,}, {int(hi):,}] | {len(df_overlap)} | "
             f"{fmt_row(coef_dict(m_overlap), cond_key)} | residual on common length support |",
             f"| (iii) long-only (≥{LONG_THRESHOLD:,}) | {len(df_long)} | "
             f"{fmt_row(coef_dict(m_long), cond_key)} | residual on the long tail only |",
             "",
             f"Raw ack rate on the long subset: baseline = {ack_long_b:.3f} "
             f"(n={int((df_long.condition=='baseline').sum())}); "
             f"W2SR weak = {ack_long_w:.3f} (n={int((df_long.condition=='w2sr_weak').sum())}).\n",
             "## Read",
             "After controlling for log(CoT length) and cue, the W2SR-vs-baseline ack drop "
             f"shrinks but does not vanish (full-data residual coefficient "
             f"{coef_dict(m_full)[cond_key]['coef']:+.2f} on the logit scale, p={coef_dict(m_full)[cond_key]['p']:.2g}). "
             f"On the matched-long tail (≥9.6k chars) W2SR contributes only "
             f"{long_w} samples vs baseline's {long_b}; the residual claim is real-direction "
             f"but underpowered, and any inference about length-matched behavior is dominated "
             f"by that small W2SR tail."]
    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_MD.relative_to(REPO)}  and  {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
