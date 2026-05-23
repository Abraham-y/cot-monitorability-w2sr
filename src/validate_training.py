"""The training-validation gate (spec 9). Run after EVERY SFT run.

No monitorability number is trusted until the corresponding training run passes
this gate; a failed gate invalidates that condition (spec 9, CLAUDE.md). The
heavy model-generation step is injected as `generate_fn` so the gate LOGIC is
unit-testable locally; on Modal `generate_fn` serves the checkpoint via vLLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from src import config

_BOXED = re.compile(r"\\boxed\{")


@dataclass
class GateReport:
    loss_converged: bool
    format_valid_rate: float
    no_degenerate_generation: bool
    pass1: float
    pass1_baseline: float
    passed: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def pass1_gain(self) -> float:
        return self.pass1 - self.pass1_baseline


def loss_converged(loss_log: list[dict]) -> bool:
    """True if training loss decreased and plateaued — not diverged, not flat
    from step 0 (spec 9.1). Compares the first vs last quartile mean loss."""
    losses = [e["loss"] for e in loss_log if "loss" in e]
    if len(losses) < 4:
        return False
    q = max(1, len(losses) // 4)
    first, last = sum(losses[:q]) / q, sum(losses[-q:]) / q
    return last < first and all(l == l for l in losses)  # decreased; no NaN


def format_valid_rate(responses: list[str]) -> float:
    """Fraction of held-out responses with a parseable boxed final answer
    (spec 9.2). Catches format collapse."""
    if not responses:
        return 0.0
    return sum(1 for r in responses if _BOXED.search(r)) / len(responses)


def degenerate_flags(responses: list[str], max_chars: int = 60000) -> list[str]:
    """Flag repetition loops, empty CoTs, runaway length (spec 9.3)."""
    flags = []
    if any(not r.strip() for r in responses):
        flags.append("empty_response")
    if any(len(r) > max_chars for r in responses):
        flags.append("runaway_length")
    for r in responses:                       # crude repetition-loop check
        toks = r.split()
        if len(toks) > 200 and len(set(toks)) / len(toks) < 0.15:
            flags.append("repetition_loop")
            break
    return flags


def pass1(responses: list[str], gts: list[str], grade_fn: Callable[[str, str], bool]) -> float:
    if not responses:
        return 0.0
    return sum(grade_fn(r, g) for r, g in zip(responses, gts)) / len(responses)


def validate(
    loss_log: list[dict],
    held_out_responses: list[str],
    held_out_gts: list[str],
    baseline_pass1: float,
    grade_fn: Callable[[str, str], bool],
    cfg: config.GateConfig | None = None,
    *,
    is_control: bool = False,
) -> GateReport:
    """Combine the four checks (spec 9). The W2SR student must show a capability
    gain over baseline (>= cfg.min_pass1_gain_abs); the CONTROL student must pass
    1-3 but its capability is only recorded (spec 9, 6.2), so set is_control=True
    to exempt it from the gain requirement.
    """
    cfg = cfg or config.GateConfig()
    conv = loss_converged(loss_log)
    fmt = format_valid_rate(held_out_responses)
    degen = degenerate_flags(held_out_responses)
    p1 = pass1(held_out_responses, held_out_gts, grade_fn)

    reasons = []
    if not conv:
        reasons.append("loss did not converge")
    if fmt < cfg.format_valid_threshold:
        reasons.append(f"format valid {fmt:.2f} < {cfg.format_valid_threshold}")
    if degen:
        reasons.append("degenerate: " + ",".join(sorted(set(degen))))
    gain_ok = is_control or (p1 - baseline_pass1) >= cfg.min_pass1_gain_abs
    if not gain_ok:
        reasons.append(f"no capability gain: pass1 {p1:.3f} vs baseline {baseline_pass1:.3f}")

    passed = conv and fmt >= cfg.format_valid_threshold and not degen and gain_ok
    return GateReport(conv, fmt, not degen, p1, baseline_pass1, passed, reasons)
