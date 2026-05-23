#!/usr/bin/env python3
"""Dump influenced-but-unacknowledged cases (the unfaithful ones) to a readable
markdown file for qualitative analysis + paper examples.

A case is "unfaithful" here when the cue FLIPPED the answer to its target
(influenced) yet the cue_aware judge found the CoT did NOT acknowledge the cue
(acknowledged=0) — i.e., the model changed its answer because of the hint but
gave a rationale that hides it.

Usage (eval venv):
    . .venv-eval/bin/activate
    python scripts/dump_unfaithful.py baseline_7b weak_teacher
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "external/monitorability-eval/logs"
OUT = REPO / "results/unfaithful_cases.md"
CUE_DIRS = ("01_stanford_professor", "02_visual_squares", "03_grader_hack",
            "04_unethical_information", "05_xml_metadata")


def main(batches: list[str]) -> None:
    from inspect_ai.log import read_eval_log
    lines = ["# Influenced-but-unacknowledged (unfaithful) cases\n"]
    total = 0
    for batch in batches:
        lines.append(f"\n## condition: {batch}\n")
        for model_dir in sorted(p for p in (LOGS / batch).iterdir() if p.is_dir()):
            for cue in CUE_DIRS:
                for evf in (model_dir / cue).rglob("*.eval"):
                    for s in (read_eval_log(str(evf)).samples or []):
                        md, sc = s.metadata or {}, s.scores or {}
                        ans = sc["answer"].value if "answer" in sc else None
                        base, tgt = md.get("baseline_model_answer"), md.get("cue_target_answer")
                        ack = sc.get("cue_aware_adaptive_scorer")
                        influenced = ans == tgt and ans != base
                        if influenced and ack is not None and int(ack.value or 0) == 0:
                            total += 1
                            cot = ((s.output.completion if s.output else "") or "").strip()
                            stop = None
                            try:
                                stop = s.output.choices[0].stop_reason
                            except Exception:
                                pass
                            truncated = stop == "max_tokens"
                            # show head + tail so the ENDING (truncated vs final
                            # answer) is visible without dumping 50k chars
                            if len(cot) > 6000:
                                shown = cot[:4000] + f"\n\n...[{len(cot) - 5000} chars omitted]...\n\n" + cot[-1000:]
                            else:
                                shown = cot
                            lines += [
                                f"\n### {s.id} — cue: {md.get('cue_name')}",
                                f"- baseline answer: **{base}** → cued answer: **{ans}** (cue target: {tgt}); correct: {md.get('correct_letter')}",
                                f"- acknowledged cue: **no** (cue_aware=0); stop_reason: **{stop}**"
                                + ("  ⚠️ TRUNCATED (hit max_tokens)" if truncated else "") + "\n",
                                "```text", shown, "```",
                            ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines.insert(1, f"\n**Total unfaithful cases: {total}** across {batches}\n")
    OUT.write_text("\n".join(lines))
    print(f"wrote {total} unfaithful cases -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["baseline_7b", "weak_teacher"])
