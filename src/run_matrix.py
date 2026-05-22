"""Orchestrator over the teacher-strength axis x size pairs (spec 7, 5.1).

For a size pair with a fixed student and a `teacher_axis` of TeacherSpecs:
  - Baseline student (untrained)                 -- eval only; starting point
  - One SFT'd student PER teacher on the axis:
        weak teacher  -> the W2SR student (treatment)
        strong teacher-> the control / distillation student (CRITICAL control)
        interior teachers -> dose-response points (spec 5.3)
  - Each teacher evaluated standalone            -- eval only; H1-vs-H2 reference
  - RL upper bound                               -- OPTIONAL (cited, not trained)

Every SFT'd student shares identical hyperparameters (config.SFTConfig); the
ONLY difference across them is teacher strength (spec 6.1). Build order
(spec 15): eval half first (baseline + teachers, no training), then traces,
then the weak+strong students through the gate, then the end-to-end slice.
Collects everything into one results table (spec 16 schema), with
teacher.strength_rank as the dose-response x-axis.
"""

from __future__ import annotations

from src import config


def run_matrix(size_pairs=None) -> None:
    size_pairs = size_pairs or config.ACTIVE_SIZE_PAIRS
    for pair in size_pairs:
        # TODO: eval baseline student; eval each teacher standalone; for each
        # teacher in pair.teacher_axis, generate/load matched traces, SFT the
        # student, run the gate, eval monitorability; (+RL bound if enabled).
        # Append one row per (pair, teacher/condition, dataset, hint_type).
        _ = pair
    raise NotImplementedError
