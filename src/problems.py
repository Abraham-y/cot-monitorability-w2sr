"""MATH problem loading for trace generation + the capability gate (spec 8.1).

Source: EleutherAI/hendrycks_math (per-subject configs; problem/level/solution).
gt_answer = the \\boxed{} answer in the reference solution, via the W2SR parser.
Fixed split + seed; reserve a DISJOINT held-out set for the Pass@1 gate (spec 9).
"""

from __future__ import annotations

import random

MATH_SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)


def extract_boxed(s: str) -> str | None:
    """Last \\boxed{...} content with balanced braces — the MATH gt answer.
    Self-contained (no latex2sympy2/antlr4): the w2sr parser's heavy deps break
    on py3.13 and aren't needed just to read the reference answer. The robust
    w2sr math_equal grader is still used for trace/gate grading (on Modal)."""
    key = "\\boxed{"
    idx = s.rfind(key)
    if idx == -1:
        return None
    i = idx + len(key)
    depth, out = 1, []
    while i < len(s) and depth > 0:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if depth > 0:
            out.append(c)
        i += 1
    return "".join(out).strip() or None


def load_math_problems(
    n_train: int = 1000,
    n_held_out: int = 200,
    levels: tuple[int, ...] = (3, 4, 5),
    split: str = "train",
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Return (train_problems, held_out_problems), each [{"problem","gt_answer"}].
    The two sets are disjoint (held-out is sliced off first). Levels 3-5 only
    (spec 8.1: harder problems surface the W2SR effect)."""
    from datasets import concatenate_datasets, load_dataset

    keep = {f"Level {l}" for l in levels}
    parts = [load_dataset("EleutherAI/hendrycks_math", s, split=split) for s in MATH_SUBJECTS]
    ds = concatenate_datasets(parts).filter(lambda e: e["level"] in keep)

    problems = []
    for e in ds:
        gt = extract_boxed(e["solution"])
        if gt:
            problems.append({"problem": e["problem"], "gt_answer": str(gt)})

    random.Random(seed).shuffle(problems)
    held_out = problems[:n_held_out]
    train = problems[n_held_out:n_held_out + n_train]
    return train, held_out
