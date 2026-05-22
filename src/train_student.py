"""Stage 2: SFT a base student on a trace dataset (spec 7, 8.2).

Handles BOTH the W2SR dataset and the matched control dataset, selected by
config. Same method/hyperparameters across the two so they are comparable
(spec 6.1). LoRA for pilot, full SFT for final.

Ground truth recipe: references/recipe_w2sr.md (extracted from the Yuan repo).
The repo trains via LLaMA-Factory; we use TRL/peft for a lighter single-GPU
LoRA path on Modal. Match total training tokens between W2SR and control.
"""

from __future__ import annotations

from src import config


def train_student(
    base_student: str,
    trace_dataset_path,
    cfg: config.SFTConfig,
    out_dir,
) -> str:
    """Run SFT; return the checkpoint path.

    TODO:
      1. Load + format traces into the chat/template the student expects
         (Qwen template). Confirm record schema vs LLaMA-Factory dataset_info.
      2. SFT (LoRA rank 16 / alpha 32, or full). Log loss curve, LR schedule,
         seed, exact data hash (spec 8.2, 16).
      3. Save checkpoint + training-validation report; then call
         validate_training.validate() — do NOT trust eval until the gate passes.
    """
    raise NotImplementedError
