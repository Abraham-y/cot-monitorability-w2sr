"""Stage 1: generate CoT traces from a teacher model (spec 7, 8.1).

Output: dataset of (problem, prompt, cot, answer, correct?) records to disk.
Ground truth for prompt templates + grading lives in external/w2sr/infer/.

Reuse, don't reinvent (spec 13.4):
  - external/w2sr/infer/generate.py     vLLM sampling loop
  - external/w2sr/infer/split_true_false.py + judge_correct.py
  - external/w2sr/infer/utils/grader.py answer extraction / grading
"""

from __future__ import annotations

from src import config


def generate_traces(
    teacher_model: str,
    n_problems: int,
    cfg: config.TraceGenConfig,
    out_path,
) -> None:
    """Sample CoT trajectories from `teacher_model` over `n_problems`.

    TODO:
      1. Load source problems (MATH / GSM8K), fixed split; reserve a disjoint
         held-out set for the capability gate (spec 9).
      2. vLLM generate at temp=0.6, top_p=0.95, max_new_tokens=4096; log seed.
      3. Grade each trace (correct?) via the w2sr grader. Keep both correct and
         incorrect for the main W2SR run; correct-only is the W2SR-P ablation.
      4. Write records + hash the dataset (spec 16) to `out_path`.
    Teacher axis (spec 5.1): call once per TeacherSpec. The weak teacher gives
    the W2SR dataset; the strong teacher gives the control/distillation dataset.
    All datasets MUST match on problem set, trace count, and token budget so the
    only variable is teacher strength (spec 6.1, 8.1).

    Strong-teacher traces (spec 8.1): PREFER pulling from a public R1 trace
    dataset (TeacherSpec.trace_dataset; OpenR1-Math / OpenThoughts) over
    generating from a 32B model. When loading a public dataset, filter to the
    weak condition's exact problem set and subsample to match its trace count
    and total token volume; see `load_matched_traces` below.
    """
    raise NotImplementedError("see references/recipe_w2sr.md for format")


def load_matched_traces(teacher, weak_problem_ids, target_token_volume, out_path) -> None:
    """Ingest an external/public CoT trace dataset for a strong teacher and
    match it to the weak condition (spec 8.1, 16): keep only `weak_problem_ids`,
    subsample to the weak trace count, and trim/select so total tokens match
    within tolerance. Hash the resulting dataset. Avoids costly 32B generation.
    """
    raise NotImplementedError
