"""Self-contained MATH answer grader (no latex2sympy2).

Why not the W2SR grader: external/w2sr/infer/utils/parser.py hard-imports
latex2sympy2, whose published build has a broken relative import + antlr4
version mismatch on our stack (dependency hell). Rather than pin a fragile
combo, we use a sympy-based grader here. It's used ONLY for the MATH Pass@1
capability gate (the reproduction signal) and trace is_correct — NOT for the
monitorability measurement (that uses the Meek eval's own scorers). The Pass@1
GAIN is what the gate tests, and the SAME grader scores baseline and trained,
so the relative comparison is robust to grader noise. Validated on MATH-style
answers in tests.
"""

from __future__ import annotations

import re


def _extract_boxed(s: str) -> str | None:
    key = "\\boxed{"
    idx = s.rfind(key)
    if idx == -1:
        return None
    i, depth, out = idx + len(key), 1, []
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


def _normalize(a: str) -> str:
    a = a.strip()
    a = re.sub(r"\\(left|right|!|,|;|:|\s|displaystyle|text|mathrm|mathbf)", "", a)
    a = a.replace("$", "").replace("\\%", "").replace("%", "")
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    a = a.replace("{", "").replace("}", "")
    a = a.replace("^\\circ", "").replace("\\circ", "")
    a = a.rstrip(".").replace(",", "").strip()
    return a


def _to_sympy(a: str):
    """Best-effort latex->sympy-parseable string, then sympify. None on failure."""
    from sympy import sympify
    s = a
    s = re.sub(r"\\frac(\d)(\d)", r"(\1)/(\2)", s)        # \frac12
    s = re.sub(r"\\frac\(?([^)]+?)\)?\(?([^)]+?)\)?$", r"(\1)/(\2)", s)
    s = s.replace("\\frac", "").replace("\\times", "*").replace("\\cdot", "*")
    s = s.replace("\\pi", "pi").replace("^", "**").replace("\\sqrt", "sqrt")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    try:
        return sympify(s)
    except Exception:
        return None


def grade(response: str, gt_answer: str) -> bool:
    """True if the response's final (boxed) answer matches gt_answer."""
    pred = _extract_boxed(response)
    if pred is None:
        # fallback: last number-ish token
        nums = re.findall(r"-?\d+\.?\d*", response)
        pred = nums[-1] if nums else None
    if pred is None:
        return False

    np_, ng = _normalize(pred), _normalize(str(gt_answer))
    if np_ == ng:
        return True
    # numeric compare
    try:
        if abs(float(np_) - float(ng)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    # symbolic compare
    sp_, sg = _to_sympy(np_), _to_sympy(ng)
    if sp_ is not None and sg is not None:
        try:
            from sympy import simplify
            if simplify(sp_ - sg) == 0:
                return True
        except Exception:
            pass
    return False
