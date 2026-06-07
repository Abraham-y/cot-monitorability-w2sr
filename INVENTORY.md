# Data inventory: stored GPQA monitorability runs

Scope: everything that exists on disk (local + Modal volume `w2sr-vol`) at the time of
audit. Aggregate metric JSONs in [results/](results/) are **summary-only** and
do not contain per-sample fields — the per-sample records live in the inspect_ai
`.eval` zips under [external/monitorability-eval/logs/](external/monitorability-eval/logs/).

## 1. Per-condition file inventory

Each `.eval` file is a zip of one inspect_ai run (one model × one cue category × one config).
`samples/gpqa_diamond_<qid>_original_epoch_1.json` is the per-question record;
`_journal/` holds aggregate stats. Path pattern:

`external/monitorability-eval/logs/<batch>/<served_name>/<cue_dir>/config_001/<timestamp>_task_*.eval`

`<cue_dir>` is one of: `baseline` (uncued), `01_stanford_professor`,
`02_visual_squares`, `03_grader_hack`, `04_unethical_information`,
`05_xml_metadata`.

### 1.1 Instruct-student family (Qwen2.5 base, no `<think>` tags)

| Batch | Served name | N per cue | Cues stored | Note |
|---|---|---|---|---|
| `baseline_7b` | `openai_Qwen2.5-7B-Instruct` | **198** | all 5 + baseline | Untrained Qwen2.5-7B-Instruct (cond 1, instruct) |
| `w2sr_student` | `openai_w2sr_infamily_inst` | **198** | all 5 + baseline | Weak-teacher SFT (R1-1.5B traces) on Qwen2.5-7B-Instruct |
| `control_student` | `openai_w2sr_control_inst` | **198** | all 5 + baseline | Strong-teacher SFT control (R1-32B / GT) on Qwen2.5-7B-Instruct |
| `pilot` | `openai_Qwen2.5-1.5B-Instruct` | **198** | all 5 + baseline | 1.5B pilot to plumb the eval |

### 1.2 Reasoning-student family (R1-distill base, native `<think>…</think>` CoT)

| Batch | Served name | N per cue | Cues stored | Note |
|---|---|---|---|---|
| `r1_7b_baseline` | `openai_DeepSeek-R1-Distill-Qwen-7B` | **40** | all 5 + baseline | Untrained R1-distill-7B (cond 1', reasoning) |
| `r1_7b_w2sr` | `openai_w2sr_r1_7b` | **40** | all 5 + baseline | W2SR student: R1-7B SFT'd on R1-1.5B traces (weak-teacher) |
| `r1_7b_w2sr_full` | `openai_w2sr_r1_7b` | **198** | baseline + cues 01, 03, 04 only | Thickened re-run of the long-bin cells (no cue 02, no cue 05) |
| `r1_7b_strong` | `openai_w2sr_r1_7b_strong` | **40** | all 5 + baseline | Strong-teacher control: R1-7B SFT'd on R1-14B traces |
| `r1_7b_self_A4k` | `openai_w2sr_r1_7b_self_A4k` | **40** | all 5 + baseline | **Self-distillation negative control (Self-A)**: R1-7B SFT'd on its OWN R1-7B traces, max_tokens=4096 (matches W2SR teacher-axis budget). T=0.6 sampling, same 1200 MATH L3-5 problems |
| `r1_7b_self_B8k` | `openai_w2sr_r1_7b_self_B8k` | **40** | all 5 + baseline | **Self-distillation negative control (Self-B)**: R1-7B SFT'd on its OWN R1-7B traces, max_tokens=8192 (natural-length arm — lets R1-7B express full CoT) |
| `r1_7b_baseline_brief` | `openai_DeepSeek-R1-Distill-Qwen-7B` | 40 baseline; **01 EMPTY (0 samples)** | baseline + 01 | Brevity-constrained probe; cue eval did not produce samples |
| `r1_7b_premise` | `openai_DeepSeek-R1-Distill-Qwen-7B` | **40** | baseline + 01 | Premise-shuffle probe (single cue) |
| `r1_7b_baseline_mmlu` | `openai_DeepSeek-R1-Distill-Qwen-7B` | **8 per subject × 5 subjects ≈ 40/cell** | all 5 + baseline | **Cross-substrate replication on MMLU** (5 STEM subjects: college_physics, college_chemistry, college_biology, college_mathematics, conceptual_physics). `W2SR_LIMIT=8` per subject. |
| `r1_7b_w2sr_mmlu` | `openai_w2sr_r1_7b` | **≈40/cell** | all 5 + baseline | W2SR weak on MMLU |
| `r1_7b_self_A4k_mmlu` | `openai_w2sr_r1_7b_self_A4k` | **≈40/cell** | all 5 + baseline | Self-A on MMLU |

### 1.3 Teacher refs

| Batch | Served name | N | Cues stored |
|---|---|---|---|
| `weak_teacher` | `openai_DeepSeek-R1-Distill-Qwen-1.5B` | **198** baseline+cue01; **60** cue02 | baseline, 01, 02 only (no 03/04/05) |
| `strong_teacher` | `openrouter_deepseek_deepseek-r1-distill-qwen-32b` | **10** | baseline only |
| `strong_teacher_modal` | `openai_DeepSeek-R1-Distill-Qwen-32B` | **10** | baseline only |

The 32B teacher only has a 10-sample baseline sanity check — there is no cued data for it.

## 2. Per-sample record schema (inspect_ai `.eval` → `samples/*.json`)

Top-level keys: `id`, `epoch`, `input`, `target`, `messages`, `output`, `scores`,
`metadata`, `store`, `events`, `model_usage`, `total_time`, `working_time`, `uuid`.

| Logical field | Where to read it | Notes |
|---|---|---|
| **Question id** | `id` (e.g. `gpqa_diamond_0000_original`) and `metadata.core_question_id` (e.g. `gpqa_diamond_0000`) | `core_question_id` is the join key across uncued/cued runs |
| **Cue type** | `metadata.cue_name` (e.g. `stanford_professor_recommends`), `metadata.cue_category` (e.g. `authority`), `metadata.cue_strength`, `metadata.cue_position`, `metadata.cue_template` | All five are present on cued samples |
| **Cued vs uncued flag** | `metadata.has_cue` (bool); `metadata.is_adaptive` is present and `True` on cued | Uncued samples (under `…/baseline/`) have `has_cue: false` and no `cue_*` keys |
| **Model's final answer** | `scores.answer.answer` (letter), plus `scores.answer.value` = `"C"` correct / `"I"` incorrect | Note: the upstream parser misses `\boxed{X}`; [scripts/patch_meek_eval.py](scripts/patch_meek_eval.py) patches `extract_metrics.py` with a `\boxed` + "ANSWER:" + "answer is" fallback. For untouched fields use `output.completion` and re-extract with the same regex. |
| **Cue's target answer** | `metadata.cue_target_answer` (letter) | Plus `metadata.baseline_model_answer` is cached — the model's UNCUED answer on the same `core_question_id`. So an "influence" indicator is computable from a cued record alone (no join needed). |
| **Full CoT text** | `messages[1].content` (assistant turn) OR `output.completion` OR `output.choices[0].message.content` — all identical strings | Full prompt sent to the model is `messages[0].content` |
| **Thinking vs answer channel** | See §3 below | Separable on `r1_7b_baseline` only |

Auxiliary scores per record:

- `scores.cue_aware_adaptive_scorer.value`: 0/1 judge label for whether the
  cued CoT verbally acknowledges the cue (channel-agnostic — the judge sees
  the full CoT, including `<think>` content).
- `scores.factor_utilization_scorer.value`: 0–1 verbosity / factor-coverage
  score from the judge.

Judge prompts used: `metadata.judge_prompt` (cue-specific).
Ground-truth answer: `target` (single letter A–D) and `metadata.correct_text`,
`metadata.choices`. All 4 multiple-choice options are in `metadata.choices`.

There is no `metadata.cue_target_answer` on `baseline/` (uncued) records — but
the cued records embed `baseline_model_answer`, so the pair is recoverable
without re-reading uncued.

## 3. CoT channel separability (thinking vs answer)

Verified by inspecting the assistant content on `01_stanford_professor` for each model:

| Condition | `<think>` open | `</think>` close | `\boxed{}` | Inline-only CoT | Can split thinking-vs-answer? |
|---|---|---|---|---|---|
| `baseline_7b` (Qwen2.5-7B-Instruct) | no | no | no | yes (ANSWER:) | **No** — instruct model, no think channel |
| `w2sr_student` (infamily_inst) | no | no | sometimes | yes | **No** |
| `control_student` (control_inst) | no | no | sometimes | yes | **No** |
| `pilot` (1.5B-Instruct) | no | no | no | yes | **No** |
| `r1_7b_baseline` (R1-distill-7B, untrained) | no¹ | **yes** | no | — | **Yes** — split at `</think>` (q0 cued sample was 15 139 chars, full reasoning trace) |
| `r1_7b_w2sr` (W2SR R1-7B) | **no** | **no** | no | yes (~1.6k chars) | **No — think tags collapsed during SFT** |
| `r1_7b_w2sr_full` (same checkpoint, full N) | **no** | **no** | no | yes | **No** |
| `r1_7b_strong` (strong-teacher R1-7B) | **no** | **no** | no | yes (~1.3k chars) | **No — same collapse** |
| `weak_teacher` (R1-distill-1.5B) | (not re-verified — same family as R1 baseline) | — | — | — | Likely yes via `</think>` |
| `strong_teacher` / `strong_teacher_modal` (R1-distill-32B) | (not re-verified, baseline-only run) | — | — | — | Likely yes via `</think>` |

¹ The R1 chat template injects the opening `<think>` invisibly (the model is
forced to begin in the think channel) so only the closing `</think>` appears in
the assistant string. Splitting on `</think>` recovers `[thinking, answer]`.

**Implication:** the per-channel acknowledgment analysis (thinking vs answer)
is only feasible on `r1_7b_baseline` (40 samples × 5 cues) — and likely
`weak_teacher` and `strong_teacher*` if you spot-verify. The whole point of
the W2SR run on R1-7B is that the trained students *lost* the think channel,
so there is nothing to split there. That itself is a finding, not a missing
file.

## 4. LoRA checkpoints (Modal volume `w2sr-vol:/checkpoints/`)

Each trained student saved **intermediate epoch-end LoRA adapters**, so a
dose-response sweep over training progress is possible without re-training.

| Adapter dir | Intermediate checkpoints | Files per checkpoint |
|---|---|---|
| `/checkpoints/w2sr_r1_7b` | `checkpoint-23`, `checkpoint-46`, `checkpoint-69` | `adapter_model.safetensors`, `adapter_config.json`, `optimizer.pt`, `scheduler.pt`, `trainer_state.json`, tokenizer/chat-template |
| `/checkpoints/w2sr_r1_7b_strong` | `checkpoint-21`, `checkpoint-42`, `checkpoint-63` | same |
| `/checkpoints/w2sr_infamily_inst` | `checkpoint-28`, `checkpoint-56`, `checkpoint-84` | same |
| `/checkpoints/w2sr_control_inst` | `checkpoint-28`, `checkpoint-56`, `checkpoint-84` | same |
| `/checkpoints/w2sr_r1_7b_self_A4k` | (final adapter only — no intermediate snapshots) | `adapter_model.safetensors`, gate report, provenance |
| `/checkpoints/w2sr_r1_7b_self_B8k` | (final adapter only) | same |

Trace dirs:
- `/vol/traces/w2sr_r1_7b_self_A4k/` — 857 kept of 1200 (68% correct, 343 degenerate dropped); data_hash `14f16f87...`
- `/vol/traces/w2sr_r1_7b_self_B8k/` — 955 kept of 1200 (78% correct, 245 degenerate dropped); data_hash `d137c96c...`

Each top-level adapter dir also has the final-epoch `adapter_model.safetensors`,
`train_provenance.json` (with `loss_log`), and `gate_report.json` (W2SR
reproduction-gate output). Merged full models are at `/merged/w2sr_r1_7b`,
`/merged/w2sr_r1_7b_strong`, `/merged/w2sr_infamily_inst`,
`/merged/w2sr_control_inst`.

The intermediate adapters are 3 epoch-snapshots per student (step counts ≈
dataset_size / batch). They give us 3 + 1 = **4 points along the training
trajectory** per student for a dose-response curve. Optimizer/scheduler state
is preserved (resume is possible if needed).

## 5. Feasibility of each analysis without re-generating

Notation: ✓ = doable from disk now; ◐ = doable but partial / caveat; ✗ = requires new generation.

### (a) Cue-influence rate
**✓ for every cued cell that exists.** Per-record: `influenced = (scores.answer.answer == metadata.cue_target_answer)` and `flipped = (scores.answer.answer != metadata.baseline_model_answer)`. Both fields are stored. The state-transition tables (correct→incorrect etc.) are already aggregated in [results/](results/) but the per-record fields are in the `.eval` zips, so you can recompute with any definition. Caveats:
- Use the patched extractor in [scripts/patch_meek_eval.py](scripts/patch_meek_eval.py) (or replicate the regex) on the R1/MATH-SFT conditions, since their CoTs end in `\boxed{X}` not `ANSWER: X`.
- `r1_7b_baseline_brief / 01_stanford_professor` has 0 stored samples; treat as missing.

### (b) Length-controlled logistic regression
**✓.** Per-sample CoT length is recoverable as `len(messages[1].content)` (or token count from `model_usage`, which is also stored). The outcome (influenced) is recoverable per (a). Covariate set could be: log(CoT chars), `cue_category`, `correct_letter`, `core_question_id` fixed effect. n is at least 198 per cell for instruct conditions and 40 for the reasoning conditions (198 for `r1_7b_w2sr_full`'s 3 cues).

### (c) Per-cue breakdown
**✓.** Cues are already stored in separate subdirs and tagged by `metadata.cue_name` / `cue_category` / `cue_strength`. All 5 cue categories are present for instruct conditions and the 40-sample R1 runs; `r1_7b_w2sr_full` is missing cues 02 (visual_squares) and 05 (xml_metadata) — those would need regeneration if you want the thickened 198 cells.

### (d) Thinking-vs-answer-channel acknowledgment
**◐ partial.** Only `r1_7b_baseline` reliably has `</think>` in the assistant text (verified on q0); `weak_teacher` and `strong_teacher*` are very likely the same family but should be spot-verified. The current judge (`cue_aware_adaptive_scorer`) ran on the **whole** CoT, so its 0/1 label is channel-agnostic. To get per-channel labels you would need to either (i) re-judge on the split substrings, or (ii) substring-search the cue text / cue keywords inside the `thinking` slice vs the `answer` slice. The raw text exists for the split; the judge labels do not.

For the four trained conditions (`r1_7b_w2sr`, `r1_7b_w2sr_full`, `r1_7b_strong`, both instruct students) the channel collapsed during SFT — there is nothing to split. That collapse is itself a measurable finding from the existing files.

### (e) Checkpoint dose-response
**◐ partial.** The intermediate LoRA adapters exist on the Modal volume for all 4 trained students (3 mid-training snapshots + 1 final each), so a dose-response over training progress is possible. **But** monitorability eval on the intermediate checkpoints was *not* run — `external/monitorability-eval/logs/` only contains the final-epoch served names. So this analysis requires re-running the eval pipeline against each intermediate adapter (merge → serve via Modal vLLM → batch_eval). No new training needed; only new eval passes.

If "dose" instead means *teacher strength*, that axis IS partially populated:
- Reasoning family: `r1_7b_w2sr` (R1-1.5B teacher) vs `r1_7b_strong` (R1-14B teacher) on the 40-sample cells; `r1_7b_w2sr_full` on the 198 for 3 cues. A 2-point dose-response. No 32B or intermediate-size student exists.
- Instruct family: `w2sr_student` (weak teacher) vs `control_student` (strong teacher) at 198 per cell. Also 2 points.

## 6. Quick paths to the files

- Aggregate metrics (already summarized): [results/](results/) — one folder per condition, single `*_metrics.json` each.
- Per-sample traces: `external/monitorability-eval/logs/<batch>/<served>/<cue>/config_001/*.eval` (zip with `samples/*.json` inside).
- Adaptive cue datasets (the prompts before they hit the model): `…/<cue>/adaptive_config_001.json`.
- All-cells aggregate per batch: `external/monitorability-eval/logs/<batch>/all_evaluations_summary.json`.
- Adapters & training state: Modal volume `w2sr-vol:/checkpoints/<student>/` (final + intermediate).
- Trace datasets that fed training: `w2sr-vol:/traces/<student>/{train.json, manifest.json, held_out.json}`.
- Merged full models for serving: `w2sr-vol:/merged/<student>/`.
