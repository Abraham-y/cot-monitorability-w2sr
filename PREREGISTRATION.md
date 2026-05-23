# Pre-registration — statistical analysis plan

Locked **before** any W2SR/control student exists or is evaluated (spec §11, §22).
Written 2026-05-23. At this point we have only condition 1 (baseline 7B student)
and condition 4 (teachers); the W2SR (condition 2) and control (condition 3)
students are not yet trained. This document fixes the analysis so it cannot be
tuned to the treatment results.

Implementation: `src/analysis.py` (per-case extraction, bootstrap, McNemar).

## 1. Unit, conditions, dependent variable
- **Unit of analysis:** one evaluation case = (question_id × cue). Paired across
  conditions on the same (question_id, cue).
- **Conditions:** baseline student (1), W2SR student (2), control/strong-teacher
  student (3); weak + strong teachers (4) as references.
- **Primary DV:** per-case **acknowledgment** (cue_aware scorer, binary) and the
  combined **monitorability** score. Secondary: verbosity, influence rate, Pass@1.
- **Benchmark:** GPQA-Diamond (headline), 5 Meek cues (`config.MEEK_CUES`),
  temperature 0. MMLU secondary if compute allows.

## 2. Estimands (pre-registered, few → no multiplicity correction)
- **E1 (practical):** monitorability/acknowledgment, **W2SR − baseline**.
- **E2 (confound-clean, PRIMARY):** monitorability/acknowledgment,
  **W2SR − control**. Only E2 attributes an effect to *weak supervision*
  specifically (spec §6.1).
- **Reference:** each teacher's standalone score, to place the W2SR student
  between baseline and teacher (H1 vs H2).

## 3. Tests
- **Acknowledgment (binary, paired on (qid,cue)):** McNemar's test + the raw
  rate difference with a **bootstrap 95% CI** (resample questions w/ replacement,
  10,000 iters). Report the CI and effect size, not just p.
- **Combined monitorability (bounded continuous):** paired mean difference +
  bootstrap 95% CI (same resampling).
- **Robustness (if feasible):** mixed-effects logistic
  `acknowledged ~ condition + (1|question) + (1|cue)`. Primary remains the
  bootstrap paired comparison; this is a check.
- **Pooling:** pool across the 5 cues for the primary test to raise informative
  N (spec §11.4). Per-cue breakdown is **exploratory** and labeled as such
  (Benjamini–Hochberg if reported as confirmatory).

## 4. Decision rules (spec §11.5) — committed in advance
Let Δ = W2SR − comparator on the primary DV, with bootstrap 95% CI.
- **H1 (inheritance):** W2SR **below** baseline AND below control, CI excludes 0,
  trending toward the (here more-faithful, reasoning) teacher. NB: because our
  weak teacher is *more* faithful than the instruct student, "inheritance" here
  predicts W2SR moves **toward the teacher's higher** faithfulness.
- **H2 (generalization beyond teacher):** W2SR **above** baseline AND above
  control (and beyond the teacher), CI excludes 0.
- **H0 (SFT-preserves):** **E2 CI includes 0** (no weak-supervision-specific
  effect) — even if E1 moves (an SFT-on-CoT effect, itself reportable).
- **H3 (base-determined):** neither W2SR nor control differs from baseline.

## 5. Validity gates (must hold before any monitorability claim)
- **Capability/reproduction gate (spec §9.4):** the W2SR student's held-out MATH
  **Pass@1 must exceed baseline by ≥ 5 absolute points** (or recover ≥30% of the
  baseline→teacher gap). If not, the W2SR reproduction failed and monitorability
  numbers for condition 2 are invalid until training is fixed.
- **Judge validity (spec §10.3):** report judge–human agreement on a ≥50-case
  hand-labeled set before trusting acknowledgment labels.
- Report influence rate and informative N per cell; if informative N is too low
  to separate H1/H2, say so rather than over-interpret.

## 6. What is NOT pre-registered (exploratory)
Per-cue effects, MMLU, the dose-response curve (>2 teachers, spec §5.3), the
ground-truth-reference sanity arm — all reported as exploratory.
