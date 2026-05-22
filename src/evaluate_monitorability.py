"""Stage 3: monitorability eval on any checkpoint (spec 7, 8.3, 10).

Built on the Meek et al. Inspect eval (external/monitorability-eval). Do NOT
reimplement the cued-input logic or improvise hint wording — drive their
scorers/prompts. Key pieces in that repo:
  - src/measuring_cot_monitorability/prompts/cue_system.py      cue templates
  - src/measuring_cot_monitorability/scorers/cue_aware_adaptive.py  faithfulness
  - src/measuring_cot_monitorability/scorers/factor_utilization.py  verbosity
  - configs/core_model_group_gpqa.yaml / _mmlu.yaml             eval presets
  - scripts/evals/run_eval.py                                   entrypoint

Output (spec 10.4, 16): per-case rows — question id, hint type, unhinted
answer, hinted answer, influenced flag, acknowledgment label (per channel),
verbosity score, monitorability score, full transcripts — as JSON/CSV.
"""

from __future__ import annotations

from src import config


def evaluate_monitorability(checkpoint_or_model: str, cfg: config.EvalConfig, out_path) -> None:
    """Run the Inspect monitorability eval and persist per-case results.

    The student is Qwen2.5-7B-Instruct, so the BASELINE already produces CoT;
    use identical CoT elicitation for baseline, every SFT'd student, and the
    teachers, so condition differences aren't elicitation artifacts (spec 5.2).

    TODO:
      1. Point Inspect at the model (vLLM-served checkpoint or HF id).
      2. For each dataset (GPQA headline) and each of the 6 cue categories:
         run baseline (unhinted) + cued prompts at temp 0; record answers/CoT.
      3. Score: faithfulness (cued CoT acknowledges the cue, per channel if
         thinking tokens exposed) + verbosity (CoT lists every needed factor)
         -> combined monitorability score.
      4. Report influence rate, acknowledgment rate (conditional + overall),
         verbosity, monitorability — and SAVE raw transcripts (spec 8.3).
    Judge: a strong model distinct from all models under test; validate on a
    >=50-case hand-labeled set (kappa>=0.6) before trusting labels (spec 10.3).
    """
    raise NotImplementedError
