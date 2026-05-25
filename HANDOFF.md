# HANDOFF — full project state (consolidated, 2026-05-25)

Single resume doc. Chronological detail is in LOG.md; findings in
references/findings_repro.md; analysis plan in PREREGISTRATION.md. This file is
the "where are we / how to continue" summary.

## Project
"Does Weak-to-Strong Reasoning (W2SR) preserve CoT monitorability?" Spec:
project_spec.md. Two halves:
- **Reproduce** (course req): W2SR capability gain on MATH (weak teacher SFTs a
  stronger student → Pass@1 rises). STATUS: **does not reproduce** in our setup
  (details below). User directive: keep trying configs until SOMETHING shows a
  gain, then go to the extension.
- **Extend** (novel contribution): measure CoT **monitorability** (Meek et al.
  Inspect eval) on baseline vs W2SR vs control students + teacher refs, on GPQA.
  This is the real paper and is further along.

## ⚙️ INFRA / GOTCHAS (read before running anything)
- **Modal profile:** active profile may be `cs336-2026` (user's other class).
  ALL our resources are in workspace **`ayeung16`**. Prefix EVERY modal command:
  `MODAL_PROFILE=ayeung16`. Do not change the user's active profile.
- **modal binary:** `/opt/miniconda3/bin/modal` (not on the non-login PATH).
- **Eval venv:** `.venv-eval` (inspect_ai, anthropic, openai, datasets,
  transformers, sympy, regex). Activate for eval/analysis/loader scripts.
- **Modal images** (modal_app.py): heavy `image` (train/gen/gate; vllm+trl+peft+
  sympy) and `vllm_image` (serving). Both set `VLLM_USE_FLASHINFER_SAMPLER=0`
  (debian_slim has no nvcc) — needed or vLLM engine init fails.
- **Grader:** src/grading.py (self-contained sympy; latex2sympy2 is broken on
  our stack). Used for MATH Pass@1 gate + trace is_correct only.
- **Judge:** `anthropic/claude-sonnet-4-6` NATIVE (bills the ~$700 Anthropic
  key). scripts/patch_meek_eval.py patches the Meek scorers to omit top_p for
  anthropic/* (else Anthropic 400s) and run_eval.py to honor
  `W2SR_MAX_CONNECTIONS`. Judge max_tokens capped 4096 via model_configs. RE-RUN
  patch_meek_eval.py after re-cloning external/.
- **OpenRouter:** ~$10 left; used only for the strong-32B teacher (monitorability)
  + earlier probes. Student/judge no longer use it.
- **vLLM endpoint:** `https://ayeung16--w2sr-monitorability-vllmserver-serve.modal.run`
  serves VLLMServer's default model (flip default in modal_app.py + `modal deploy`).
  Trained checkpoints are gated OFFLINE (gate loads base+LoRA in-process), NOT via
  the endpoint.
- **Background-watcher fragility:** long-poll `nohup`/run_in_background bashes
  often die mid-poll. Use sparse polls (sleep≥240) or just check the volume
  manually. Detached Modal jobs (`modal run --detach`) survive regardless.
- **Modal connection blips:** occasional `AppHeartbeat Deadline exceeded` kills a
  non-detached `modal run`. Use `--detach` for anything long; poll the volume.

## Volume `w2sr-vol` layout
- `/traces/{w2sr (R1 L3-5, 729 kept), w2sr_simplerl (SimpleRL, noisy), w2sr_L5
  (R1 L5, 528 kept), w2sr_r32}` — each has train.json, held_out.json, manifest.json
- `/checkpoints/{w2sr_base, w2sr_base_r32, w2sr_simplerl, w2sr_L5, w2sr_math7b}`
  — each has adapter_model.safetensors + gate_report.json
- `/hf_cache` — HF weights

## REPRODUCTION — results (all vs UNELICITED baseline unless noted; ALL FLAT)
| student | teacher | task | rank | baseline | W2SR | gain | format |
|---|---|---|---|---|---|---|---|
| 7B-Instruct | R1-1.5B | L3-5 | 16 | 0.605* | 0.16 | −0.445 | 0.10 |
| 7B-base | R1-1.5B | L3-5 | 16 | 0.45 | 0.435 | −0.015 | 0.575 |
| 7B-base | R1-1.5B | L3-5 | 32 | 0.445 | 0.45 | +0.005 | 0.615 |
| 7B-base | SimpleRL-1.5B | L3-5 | 32 | 0.445 | 0.12 | −0.325 | 0.26 |
| 7B-base | R1-1.5B | **L5** | 32 | 0.355 | 0.335 | −0.02 | 0.475 |
| **Math-7B** | R1-1.5B | L5 | 32 | 0.39 | 0.395 | +0.005 | 0.505 |
\*CoT-prompted (inflated); all others unelicited (build_direct_prompt). See
PREREGISTRATION §4b: reproduction baseline = no-CoT; monitorability baseline = zero-shot CoT.

**Findings (references/findings_repro.md):** (1) no W2SR gain across the matrix;
(2) R1-distill's long over-thinking CoT degrades the student under LoRA (format
0.47-0.62, doesn't conclude) — bounded fixes (rep_penalty hurts math accuracy;
more tokens don't help; rank16→32 + filter only 0.55→0.615); gain stays flat as
format improves → cleaner gen doesn't hide a gain; (3) SimpleRL-Zoo-1.5B traces
are noisy (CJK/URLs/garbage, 18% correct) → collapse. Benchmark/headroom is NOT
the blocker (L5 had headroom, still flat).

## NEXT (per user "reproduce SOMETHING, then extension")
1. **Strong-teacher distillation control** (HIGH-confidence gain + needed by the
   extension). Source `open-r1/OpenR1-Math-220k` (fields: problem, generations
   [R1 traces], correctness_math_verify, answer). Build train.json from CORRECT,
   length-filtered traces → SFT Qwen2.5-7B (general; 32k ctx ok) → gate vs the
   unelicited baseline. Standard distillation from strong R1 reliably lifts the
   student → "reproduce something". Reuse gen_traces format helpers / write a
   small OpenR1 loader; train via modal_app::train; gate via modal_app::gate
   (max_tokens 4096, max_model_len 8192).
   - If gain → reproduced (distillation). Proceed to extension.
   - If still flat → likely LoRA-vs-full-SFT is the limiter; try full SFT
     (finetuning_type full) as a last reproduction lever.
2. **EXTENSION (the actual paper):** monitorability eval (configs/*.yaml drive
   external/monitorability-eval via batch_eval.py; see scripts/run_*_teacher.sh
   for the 6-pass pattern: baseline pass + 5 adaptive-cue passes + extract_metrics).
   - Condition 1 (baseline 7B-Instruct student): DONE — results/baseline_7b_metrics/
     (acc 0.369, verbosity 0.476, faithfulness 0–4.5%/cue).
   - Condition 4a (weak teacher R1-1.5B): DONE — results/weak_teacher_metrics/
     (verbosity 0.640, faith 0.217/0.015/0.061/0.136/0.0). NOTE: a 30k-token
     re-run was launched for clean (un-truncated) numbers — CHECK if it finished
     (6 .eval logs at 198 each in logs/weak_teacher); if not, restart it.
   - Condition 4b (strong 32B teacher, OpenRouter): INCOMPLETE (was ~40/198
     baseline). Re-run scripts/run_strong_teacher.sh (needs OpenRouter $).
   - Conditions 2 & 3 (W2SR + control student MONITORABILITY): serve each trained
     checkpoint (Modal vLLM endpoint or a served LoRA) and run the monitorability
     eval. NB: capability gate failed for W2SR, so per spec §9 a monitorability
     CLAIM for the W2SR student is caveated — but the descriptive comparison
     (baseline vs W2SR vs control faithfulness) is still the study; report the
     gate status alongside (spec 16 schema).
   - Analysis: src/analysis.py (load_cases → bootstrap_paired_diff → mcnemar →
     adjudicate per PREREGISTRATION H0-H3). Validated: weak_teacher−baseline
     acknowledgment Δ=+0.179 [95% CI .143,.218], McNemar p~6e-20.

## Run patterns (copy-paste)
- Gen traces (Modal GPU): `MODAL_PROFILE=ayeung16 /opt/miniconda3/bin/modal run --detach
  modal_app.py::gen_traces --teacher-model <hf> --out-dir /vol/traces/<x>
  --n-problems N [--levels "5"] [--teacher-system "You are a helpful assistant."]`
- Train: `... modal run --detach modal_app.py::train --base-student <hf>
  --train-json /vol/traces/<x>/train.json --out-dir /vol/checkpoints/<y>
  [--max-seq-len 4096]`
- Gate: `... modal run --detach modal_app.py::gate --base-student <hf>
  --adapter-dir /vol/checkpoints/<y> --held-out-json /vol/traces/<x>/held_out.json
  --rep-penalty 1.0 --max-tokens 4096 [--max-model-len 4096]`
  Then fetch /vol/checkpoints/<y>/gate_report.json.
- Monitorability eval: see scripts/run_weak_teacher.sh (deploy endpoint→warm→
  batch_eval baseline + 5 cues→extract_metrics). Judge=anthropic/claude-sonnet-4-6.
