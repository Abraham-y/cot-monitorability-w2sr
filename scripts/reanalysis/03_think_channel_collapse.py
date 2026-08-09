"""Task 3 — think-channel emission scan (supervision format vs completions).

Quantifies think-tag markers in (1) the stored training trace files and (2) the
students' completions.

IMPORTANT — how NOT to read Step 1. This task originally asked whether the
`<think>…</think>` collapse is *imitation* (traces were stripped) or *emergent*
(traces carried the tags but the student stopped emitting them), and concluded
"emergent" because the trace files carry `</think>` in 100% of records. That
inference was wrong. The trace files are not the supervision: the R1-Distill
chat template splits assistant content on `</think>` and keeps only the final
segment, so rendering these traces for SFT produced answer-only training text.
Step 1 therefore measures the trace files only, and cannot by itself establish
that the student ever saw the reasoning. See `src/train_student.build_sft_text`,
which now renders the assistant turn without the template and hard-fails if the
reasoning span is missing.

Step 1: scan the four training trace datasets (downloaded from Modal volume
`w2sr-vol:/traces/`) for `<think>` / `</think>` / `\boxed` / `ANSWER:`
markers, and record median CoT length.

  Trace dataset → student:
    /vol/traces/w2sr                 → checkpoints/w2sr_r1_7b        (R1-7B W2SR weak, teacher = R1-1.5B)
    /vol/traces/w2sr_r1_14b          → checkpoints/w2sr_r1_7b_strong (R1-7B W2SR strong, teacher = R1-14B)
    /vol/traces/w2sr_infamily        → checkpoints/w2sr_infamily_inst (instruct W2SR weak, teacher = Qwen2.5-Math-1.5B)
    /vol/traces/w2sr_infamily_strong → checkpoints/w2sr_control_inst  (instruct W2SR strong, teacher = Qwen2.5-Math-72B)
  (provenance verified from each checkpoint's train_provenance.json.)

Step 2: per-condition stored-completion scan. Read every cued + baseline
record across the 6 condition batches and quantify:
  - fraction containing `<think>` (open tag)
  - fraction containing `</think>` (close tag — what gets emitted under the R1 chat template)
  - fraction containing `\boxed` and `ANSWER:` (sanity checks for the patched extractor)
  - median CoT chars

Inputs:
  /tmp/w2sr_traces/{w2sr,w2sr_r1_14b,w2sr_infamily,w2sr_infamily_strong}_train.json
    (pre-fetched via:
       modal volume get w2sr-vol /traces/<name>/train.json /tmp/w2sr_traces/<name>_train.json)
  external/monitorability-eval/logs/<batch>/<served>/<cue_dir>/config_001/*.eval  (all 6 batches)

Outputs:
  results/reanalysis/03_think_channel_collapse.md
  results/reanalysis/03_think_channel_collapse.json
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import CONDITIONS, REPO, load_records

OUT_MD   = REPO / "results/reanalysis/03_think_channel_collapse.md"
OUT_JSON = REPO / "results/reanalysis/03_think_channel_collapse.json"

TRACE_DIR = Path("/tmp/w2sr_traces")
TRACES = [
    ("w2sr",                  "R1-7B W2SR weak   (R1-1.5B teacher)"),
    ("w2sr_r1_14b",           "R1-7B W2SR strong (R1-14B teacher)"),
    ("w2sr_infamily",         "Instruct W2SR weak   (Qwen2.5-Math-1.5B teacher)"),
    ("w2sr_infamily_strong",  "Instruct W2SR strong (Qwen2.5-Math-72B teacher)"),
]


def scan(strings):
    """Return marker fractions + length stats for a list of CoT strings."""
    n = len(strings)
    if not n:
        return {"n": 0}
    n_open  = sum(1 for s in strings if "<think>"  in s)
    n_close = sum(1 for s in strings if "</think>" in s)
    n_boxed = sum(1 for s in strings if "\\boxed"  in s)
    n_ans   = sum(1 for s in strings if "ANSWER:"  in s)
    lens = [len(s) for s in strings]
    return {
        "n": n,
        "frac_think_open": n_open / n,
        "frac_think_close": n_close / n,
        "frac_boxed": n_boxed / n,
        "frac_answer_kw": n_ans / n,
        "median_chars": int(st.median(lens)),
        "mean_chars":   int(st.mean(lens)),
        "p95_chars":    int(sorted(lens)[int(0.95 * (n - 1))]),
        "max_chars":    max(lens),
    }


def main():
    print("=" * 70)
    print("TASK 3 — think-channel collapse")
    print("=" * 70)

    # Step 1: training traces
    print("\n=== Step 1: training trace datasets (the SFT signal) ===")
    trace_results = {}
    for stem, label in TRACES:
        p = TRACE_DIR / f"{stem}_train.json"
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        rows = json.loads(p.read_text())
        # Llama-Factory record: {'content': prompt, 'output': assistant CoT}
        strs = [r.get("output", "") for r in rows]
        s = scan(strs)
        trace_results[stem] = {"label": label, "path": str(p), **s}
        print(f"\n  {label}")
        print(f"    {stem}_train.json: n={s['n']}, median_chars={s['median_chars']}, max={s['max_chars']}")
        print(f"    <think> open  : {100*s['frac_think_open']:5.1f}%")
        print(f"    </think> close: {100*s['frac_think_close']:5.1f}%")
        print(f"    \\boxed        : {100*s['frac_boxed']:5.1f}%")
        print(f"    ANSWER:       : {100*s['frac_answer_kw']:5.1f}%")

    print(
        "\n  Read on training-data think tags:\n"
        "    R1-distill trace sets (w2sr, w2sr_r1_14b): </think> appears in 100% of records;\n"
        "    <think> opening tag is 0% because the generation prompt supplies it — the\n"
        "    assistant's literal 'output' string starts already inside the think channel and\n"
        "    contains the closing tag.\n"
        "    NOTE: this describes the trace FILES, not the SFT supervision. The R1-Distill\n"
        "    chat template drops everything up to and including </think> on assistant turns,\n"
        "    so these traces render to answer-only training text. Do not conclude from this\n"
        "    table that the student was trained on the reasoning.\n"
        "    Instruct trace sets (w2sr_infamily*, Qwen2.5-Math teachers): zero think tags\n"
        "    in either direction — Qwen2.5-Math teachers never emit them (and ChatML does\n"
        "    not strip, so those arms' supervision did include the full teacher output)."
    )

    # Step 2: stored-completion think-tag scan, per condition
    print("\n=== Step 2: stored-completion scan, per condition ===")
    cond_results = {}
    print(f"  {'condition':40s} {'cell':5s} {'n':>5s} {'<think>':>9s} {'</think>':>10s} "
          f"{'\\boxed':>8s} {'ANSWER:':>9s} {'median_chars':>14s}")

    # Include baseline (uncued) AND cued cells separately.
    for batch, served, label in CONDITIONS:
        all_cued = load_records(batch, served, cued_only=True)
        all_with_un = load_records(batch, served, cued_only=False)
        uncued = [r for r in all_with_un if r.cue_dir == "baseline"]
        cells = [("cued", all_cued), ("uncued", uncued)]
        cond_results[label] = {"batch": batch, "served": served, "cells": {}}
        for name, rs in cells:
            comps = [r.completion for r in rs]
            s = scan(comps)
            cond_results[label]["cells"][name] = s
            if s["n"] == 0:
                continue
            print(f"  {label:40s} {name:5s} {s['n']:5d} "
                  f"{100*s['frac_think_open']:8.1f}% {100*s['frac_think_close']:9.1f}% "
                  f"{100*s['frac_boxed']:7.1f}% {100*s['frac_answer_kw']:8.1f}% "
                  f"{s['median_chars']:14d}")

    # Build the verdict
    base_r1   = cond_results["baseline R1-7B"]["cells"]["cued"]
    w2sr_r1w  = cond_results["W2SR weak (R1-1.5B teacher)"]["cells"]["cued"]
    w2sr_r1s  = cond_results["W2SR strong (R1-14B teacher)"]["cells"]["cued"]
    verdict = (
        f"R1 baseline emits </think> on {100*base_r1['frac_think_close']:.0f}% of cued completions; "
        f"R1-7B W2SR weak on {100*w2sr_r1w['frac_think_close']:.0f}%; "
        f"R1-7B W2SR strong on {100*w2sr_r1s['frac_think_close']:.0f}%. "
        "The trace FILES carry </think> in 100% of records, but that is not what the student "
        "was trained on: the R1-Distill chat template splits assistant content on </think> and "
        "keeps only the final segment, so the tokenized supervision was the answer only, with "
        "the reasoning span removed. The drop in </think> emission and the CoT compression are "
        "therefore explained by the supervision format, NOT emergent under SFT. (Earlier "
        "versions of this file asserted the opposite; the check was run on the trace files "
        "rather than on the rendered training text. See src/train_student.build_sft_text.) "
        "Compression measured here: "
        f"({base_r1['median_chars']:,} → {w2sr_r1w['median_chars']:,} chars median; "
        f"{base_r1['median_chars']/w2sr_r1w['median_chars']:.1f}× shorter)."
    )
    print(f"\n  Read: {verdict}")

    OUT_JSON.write_text(json.dumps({
        "training_traces": trace_results,
        "stored_completions": cond_results,
        "verdict": verdict,
    }, indent=2, default=str))

    # Markdown
    lines = ["# Task 3 — think-channel collapse: imitation or emergent\n",
             "## Step 1: training trace datasets\n",
             "Each W2SR student's trace dataset was downloaded via "
             "`modal volume get w2sr-vol /traces/<name>/train.json`. "
             "Provenance taken from `checkpoints/<student>/train_provenance.json`.\n",
             "| trace set | student trained on it | n | <think> | </think> | \\boxed | ANSWER: | median chars |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for stem, label in TRACES:
        d = trace_results.get(stem)
        if not d:
            continue
        lines.append(f"| /vol/traces/{stem} | {label} | {d['n']} | "
                     f"{100*d['frac_think_open']:.1f}% | {100*d['frac_think_close']:.1f}% | "
                     f"{100*d['frac_boxed']:.1f}% | {100*d['frac_answer_kw']:.1f}% | "
                     f"{d['median_chars']:,} |")
    lines += ["",
              "*Note on the 0% `<think>` open + 100% `</think>` close on R1 traces:* the R1 "
              "generation prompt ends `<|Assistant|><think>`, so the stored 'output' string "
              "begins *inside* the think channel and ends with the closing tag. **This table "
              "describes the trace FILES, not the supervision.** The R1-Distill chat template "
              "splits assistant content on `</think>` and keeps only the last segment, so "
              "rendering these rows through it yields answer-only training text with the "
              "reasoning removed. Any inference of the form 'the traces contained the CoT, "
              "therefore the student was trained on the CoT' is invalid — that was the error "
              "in earlier versions of this task. See `src/train_student.build_sft_text`.\n",
              "## Step 2: stored-completion think-tag fraction per condition\n",
              "Scanned every cued + uncued completion across all 6 batches.\n",
              "| condition | cell | n | `<think>` | `</think>` | `\\boxed` | `ANSWER:` | median chars |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, _, label in CONDITIONS:
        for cell in ("cued", "uncued"):
            s = cond_results[label]["cells"].get(cell, {})
            if not s or s.get("n", 0) == 0:
                continue
            lines.append(
                f"| {label} | {cell} | {s['n']} | "
                f"{100*s['frac_think_open']:.1f}% | {100*s['frac_think_close']:.1f}% | "
                f"{100*s['frac_boxed']:.1f}% | {100*s['frac_answer_kw']:.1f}% | "
                f"{s['median_chars']:,} |"
            )
    lines += ["",
              "## Verdict",
              verdict, ""]
    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_MD.relative_to(REPO)}  and  {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
