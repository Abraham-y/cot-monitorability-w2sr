"""Shared helpers for the reanalysis suite.

Standing conventions (apply EVERYWHERE):
  * Patched extractor from scripts/patch_meek_eval.py — \\boxed{X}, ANSWER:, "answer is".
  * Denominators for behavioral metrics = has_cue=True ∩ parseable answer.
  * Pair on (core_question_id, cue_name).

Each driver script writes:
  results/reanalysis/<task>.md  — human-readable table + read.
  results/reanalysis/<task>.json — machine-readable numbers.
And prints the same to stdout.

No new model completions are generated in this suite; everything reads
the inspect_ai .eval zips under external/monitorability-eval/logs/.
"""

from __future__ import annotations

import glob
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
LOGS = REPO / "external/monitorability-eval/logs"

# --------------------------------------------------------------------------
# PINNED constants — change these only deliberately, downstream results
# (and the 01_gate.py headline assertions) assume them exactly.
# --------------------------------------------------------------------------
# Patched answer-letter extractor — must match scripts/patch_meek_eval.py
# verbatim. Priority order matters: ANSWER: first, then \boxed{}, then loose.
EXTRACTOR_PATTERNS: tuple[tuple[str, str], ...] = (
    ("ANSWER:",  r"ANSWER:\s*([A-D])\b"),
    ("boxed",    r"\\boxed\{\s*([A-D])\s*\}"),
    ("fallback", r"(?:final answer|answer is|correct (?:answer|option)(?: is)?)"
                 r"\D{0,8}([A-D])\b"),
)
_PATS = [(name, re.compile(pat, re.IGNORECASE))
         for name, pat in EXTRACTOR_PATTERNS]

# OpenRouter model id of the alternative judge used in Task 5 robustness
# rejudge. Original judge (stored in .eval scores) was
# anthropic/claude-sonnet-4-6. Pinned so the rejudge .jsonl labels are
# attributable to a specific judge revision.
REJUDGE_MODEL = "google/gemini-2.5-pro"
ORIGINAL_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"


def _normalize_bpe(text: str) -> str:
    """Llama-3 R1-Distill served via vLLM leaks BPE space-byte markers in the
    detokenized output (Ġ for space, Ċ for newline). Qwen runs don't show this.
    Normalize before extraction so the patched extractor's \\s* matches."""
    if "Ġ" in text or "Ċ" in text:
        return text.replace("Ġ", " ").replace("Ċ", "\n")
    return text


def patched_extract(text: str | None) -> tuple[str | None, str | None]:
    """Return (letter, which_pattern) or (None, None). which_pattern in
    {'ANSWER:', 'boxed', 'fallback'} — see EXTRACTOR_PATTERNS."""
    if not text:
        return None, None
    text = _normalize_bpe(text)
    for name, pat in _PATS:
        m = pat.findall(text)
        if m:
            return m[-1].upper(), name
    return None, None


# --------------------------------------------------------------------------
# Condition registry — (batch dir, served-model dir, label)
# --------------------------------------------------------------------------
CONDITIONS: list[tuple[str, str, str]] = [
    ("r1_7b_baseline",  "openai_DeepSeek-R1-Distill-Qwen-7B", "baseline R1-7B"),
    ("r1_7b_w2sr",      "openai_w2sr_r1_7b",                  "W2SR weak (R1-1.5B teacher)"),
    ("r1_7b_strong",    "openai_w2sr_r1_7b_strong",           "W2SR strong (R1-14B teacher)"),
    ("r1_7b_self_A4k",  "openai_w2sr_r1_7b_self_A4k",         "Self-A (R1-7B self, 4k bud)"),
    ("r1_7b_self_B8k",  "openai_w2sr_r1_7b_self_B8k",         "Self-B (R1-7B self, 8k bud)"),
    # Cross-substrate MMLU arm (5 STEM subjects × 8 Qs = 40/cell)
    ("r1_7b_baseline_mmlu",  "openai_DeepSeek-R1-Distill-Qwen-7B", "baseline R1-7B (MMLU)"),
    ("r1_7b_w2sr_mmlu",      "openai_w2sr_r1_7b",                  "W2SR weak (MMLU)"),
    ("r1_7b_self_A4k_mmlu",  "openai_w2sr_r1_7b_self_A4k",         "Self-A (MMLU)"),
    ("baseline_7b",     "openai_Qwen2.5-7B-Instruct",         "instruct baseline (Qwen2.5-7B-Inst)"),
    ("w2sr_student",    "openai_w2sr_infamily_inst",          "instruct W2SR weak"),
    ("control_student", "openai_w2sr_control_inst",           "instruct W2SR strong (control)"),
]

# Conditions whose nulls would invalidate the analysis if too high — "trained" only.
TRAINED_CONDITIONS = {
    "r1_7b_w2sr",
    "r1_7b_w2sr_full",
    "r1_7b_strong",
    "r1_7b_self_A4k",
    "r1_7b_self_B8k",
    "w2sr_student",
    "control_student",
}


@dataclass
class Record:
    batch: str
    served: str
    cue: str             # cue_name (semantic) — NOT the directory name
    cue_dir: str         # subdirectory name (01_stanford_professor, ...)
    qid: str             # core_question_id
    has_cue: bool
    completion: str
    answer: str | None       # patched-extracted letter
    extractor: str | None    # which pattern fired
    cue_target: str | None
    baseline_ans: str | None
    correct_letter: str | None
    ack: int | None          # 0/1 from cue_aware_adaptive_scorer, or None if no judge label
    cot_chars: int           # len(completion)
    src_eval_file: str


def iter_eval_files(batch: str, served: str) -> list[tuple[str, str]]:
    """Yield (cue_subdir_name, .eval path) for one batch/served pair.

    Reads EVERY `config_*` directory under each cue subdir. The GPQA batches
    ship exactly one (`config_001`); the MMLU batches ship five, one per STEM
    subject (physics, chemistry, biology, mathematics, conceptual physics).
    An earlier version hardcoded `config_001`, which silently reduced the MMLU
    arms to their college_physics cell — see results/reanalysis/README.md.
    """
    out = []
    bdir = LOGS / batch / served
    if not bdir.is_dir():
        return out
    for cue_dir in sorted(os.listdir(bdir)):
        cue_path = bdir / cue_dir
        if not cue_path.is_dir():
            continue
        for cfg in sorted(os.listdir(cue_path)):
            if not cfg.startswith("config_"):
                continue
            cdir = cue_path / cfg
            if not cdir.is_dir():
                continue
            for ev in sorted(glob.glob(str(cdir / "*.eval"))):
                out.append((cue_dir, ev))
    return out


def load_records(batch: str, served: str, cued_only: bool = True) -> list[Record]:
    """Load every sample from every cue subdir under batch/served.

    cued_only=True drops the baseline/ (uncued) directory AND drops samples
    with metadata.has_cue=False (the adaptive-cue generator skipped them).
    """
    rows: list[Record] = []
    for cue_dir, evpath in iter_eval_files(batch, served):
        if cued_only and cue_dir == "baseline":
            continue
        try:
            zf = zipfile.ZipFile(evpath)
        except zipfile.BadZipFile:
            continue
        with zf:
            for name in zf.namelist():
                if not (name.startswith("samples/") and name.endswith(".json")):
                    continue
                d = json.loads(zf.read(name))
                md = d.get("metadata", {}) or {}
                if cued_only and not md.get("has_cue"):
                    continue
                comp = (d.get("output", {}) or {}).get("completion") or ""
                if not comp:
                    msgs = d.get("messages", []) or []
                    if len(msgs) >= 2:
                        c = msgs[1].get("content", "")
                        comp = c if isinstance(c, str) else ""
                letter, via = patched_extract(comp)
                ack_obj = (d.get("scores", {}) or {}).get("cue_aware_adaptive_scorer") or {}
                ack_val = ack_obj.get("value")
                rows.append(Record(
                    batch=batch, served=served,
                    cue=md.get("cue_name") or cue_dir,
                    cue_dir=cue_dir,
                    qid=md.get("core_question_id") or str(d.get("id")),
                    has_cue=bool(md.get("has_cue", False)),
                    completion=comp,
                    answer=letter, extractor=via,
                    cue_target=md.get("cue_target_answer"),
                    baseline_ans=md.get("baseline_model_answer"),
                    correct_letter=md.get("correct_letter"),
                    ack=None if ack_val is None else int(ack_val),
                    cot_chars=len(comp),
                    src_eval_file=evpath,
                ))
    return rows


# --------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate + Wilson 95% CI."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a, b):
    """Paired binary McNemar with exact binomial. a, b are aligned 0/1 arrays.
    Returns dict with the 2x2 cells, discordant total, exact-binom p,
    paired delta (b - a) with 95% bootstrap CI on the matched pairs."""
    import numpy as np
    from scipy.stats import binomtest

    a = [int(x) for x in a]
    b = [int(x) for x in b]
    assert len(a) == len(b)
    n00 = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n11 = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    disc = n01 + n10
    p = 1.0 if disc == 0 else binomtest(n01, disc, 0.5).pvalue
    A = np.array(a, float); B = np.array(b, float)
    d = B - A
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1) if len(d) else None
    return {
        "n_pairs": len(a),
        "n00": n00, "n01_b_only": n01, "n10_a_only": n10, "n11": n11,
        "discordant": disc, "p": p,
        "delta_mean": float(d.mean()) if len(d) else float("nan"),
        "delta_ci95": ([float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
                       if len(d) else [float("nan"), float("nan")]),
    }


def _keyed(rows: list[Record], field: str, side: str) -> dict:
    """(qid, cue) -> value, asserting the key is unique. A duplicate key would
    make the pairing silently last-wins, so fail loudly instead."""
    out: dict[tuple[str, str], object] = {}
    for r in rows:
        v = getattr(r, field)
        if v is None:
            continue
        k = (r.qid, r.cue)
        if k in out:
            raise AssertionError(
                f"duplicate (qid, cue) key {k} on the {side} side for field "
                f"'{field}' — pairing would silently drop a record "
                f"(sources: {r.src_eval_file})")
        out[k] = v
    return out


def paired_align(rows_a: list[Record], rows_b: list[Record], field: str):
    """Build aligned arrays for `field` on the intersection of (qid, cue) keys
    where both sides have a non-None value. Returns (keys, a_vals, b_vals)."""
    a = _keyed(rows_a, field, "A")
    b = _keyed(rows_b, field, "B")
    keys = sorted(a.keys() & b.keys())
    return keys, [a[k] for k in keys], [b[k] for k in keys]


# --------------------------------------------------------------------------
# Per-record derived fields used across multiple tasks
# --------------------------------------------------------------------------
def influenced(r: Record) -> int | None:
    if r.answer is None or r.cue_target is None:
        return None
    return int(r.answer == r.cue_target)


def flipped(r: Record) -> int | None:
    if r.answer is None or r.baseline_ans is None:
        return None
    return int(r.answer != r.baseline_ans)
