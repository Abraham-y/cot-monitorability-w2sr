"""The training-validation gate (spec 9). Run after EVERY SFT run.

A failed gate invalidates that condition's monitorability numbers. Record the
outcome in the results table next to each condition (spec 9, 16).
"""

from __future__ import annotations

from dataclasses import dataclass

from src import config


@dataclass
class GateReport:
    loss_converged: bool
    format_valid_rate: float
    no_degenerate_generation: bool
    pass1: float
    pass1_baseline: float
    passed: bool
    notes: str = ""


def validate(checkpoint_path, baseline_pass1: float, cfg: config.GateConfig) -> GateReport:
    """Gate: passes only if ALL hold (spec 9):

      1. Loss converged (decreased then plateaued; not diverged/flat-from-0).
      2. Output format valid on >=20 held-out problems (parseable boxed answer
         >= 90% of the time).
      3. No degenerate generation (repetition loops, truncation, empty CoT).
      4. Capability gain: Pass@1 on the disjoint held-out math set beats the
         untrained baseline by the pre-committed threshold (config.GateConfig).
         If W2SR shows NO gain, the reproduction failed — fix before any
         monitorability claim (spec 9.4, 17).

    The control student must pass 1-3; its capability is recorded for the
    capability-confound discussion (spec 6.2) but need not match W2SR.
    """
    raise NotImplementedError
