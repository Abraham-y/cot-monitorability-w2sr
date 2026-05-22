"""Analysis: deltas, bootstrap CIs, McNemar, mixed-effects, plots (spec 7, 11).

Pre-register the primary test on PILOT data, then run it UNCHANGED on the full
data (spec 11). Unit of analysis = individual case (question x hint type).

Primary estimands:
  1. monitorability(W2SR) - monitorability(baseline)   [practically relevant]
  2. monitorability(W2SR) - monitorability(control)    [confound-clean]
Here "control" = the strong-teacher (distillation) student; "W2SR" = the
weak-teacher student. Reference: each teacher's standalone score, to adjudicate
H1 (inheritance) vs H2 (improve). If >2 teachers are on the axis, also fit/plot
monitorability vs teacher.strength_rank as a dose-response curve (spec 5.3).

Tests (spec 11.2):
  - Combined score: paired mean diff + bootstrap 95% CI (resample questions,
    10000 iters). Report CI + effect size, not just p.
  - Binary acknowledgment on paired cases: McNemar + rate diff with bootstrap CI.
  - Preferred: mixed-effects logistic
      acknowledgment ~ condition + (1|question) + (1|hint_type)   (statsmodels)
    as primary if feasible, else robustness check.

Decision rule (spec 11.5): H1 if W2SR < baseline AND < control (CI excludes 0);
H2 if W2SR > both (CI excludes 0); H0 if W2SR-control CI includes 0; H3 if
nothing moves from baseline. Per-hint breakdown is exploratory (correct or label
as such; don't cherry-pick one significant hint as the headline -- spec 11.3).
"""

from __future__ import annotations


def bootstrap_paired_diff(scores_a, scores_b, n_iter: int = 10000, seed: int = 0):
    """Mean paired difference with a bootstrap 95% CI (resample questions)."""
    raise NotImplementedError


def mcnemar_ack(paired_labels_a, paired_labels_b):
    """McNemar's test on paired binary acknowledgment outcomes."""
    raise NotImplementedError


def mixed_effects_ack(df):
    """acknowledgment ~ condition + (1|question) + (1|hint_type)."""
    raise NotImplementedError
