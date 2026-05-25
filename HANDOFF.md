# HANDOFF — full project state (consolidated, 2026-05-25)

Single resume doc. Chronological detail is in LOG.md; findings in
references/findings_repro.md; analysis plan in PREREGISTRATION.md. This file is
the "where are we / how to continue" summary.

## Project
"Does Weak-to-Strong Reasoning (W2SR) preserve CoT monitorability?" Spec:
project_spec.md. Two halves:
- **Reproduce** (course req): W2SR capability gain on MATH (weak teacher SFTs a
  stronger student → Pass@1 rises). STATUS: **REPRODUCED** (2026-05-25) with an
  in-family native-Qwen weak teacher: Qwen2.5-Math-1.5B-Instruct → Qwen2.5-Math-7B,
  Pass@1 0.325→0.670 (**+0.345**, gate PASSED, format 1.0). Earlier cross-style
  teachers (R1-distill, SimpleRL) were flat/collapse — root cause was teacher↔student
  style mismatch, not the method (findings_repro.md Finding 4). Course req met.
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
  (R1 L5, 528 kept), w2sr_r32, w2sr_infamily (Math-1.5B-Inst L3-5, 891 kept ✅)}`
  — each has train.json, held_out.json, manifest.json
- `/checkpoints/{w2sr_base, w2sr_base_r32, w2sr_simplerl, w2sr_L5, w2sr_math7b,
  w2sr_infamily (✅ gate PASSED +0.345)}` — adapter_model.safetensors + gate_report.json
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
| **Math-7B** | **Math-1.5B-Inst** (in-family) | L3-5 | 32 | 0.325 | **0.67** | **+0.345** ✅ | **1.00** |
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
0. **✅ DONE — in-family native-Qwen weak teacher REPRODUCED W2SR** (2026-05-25).
   Qwen2.5-Math-1.5B-Instruct → Qwen2.5-Math-7B, L3-5: Pass@1 0.325→0.670
   (+0.345), format 1.0, no degeneration, gate PASSED. Artifacts:
   /traces/w2sr_infamily (891 traces, hash 7235870…), /checkpoints/w2sr_infamily.
   Course-req reproduction met → focus shifts to the EXTENSION (step 2). NOTE:
   this Math-7B student is REPRODUCTION-ONLY; the monitorability student stays the
   general 7B (GPQA is science, not math; cond-1 baseline already on 7B-Instruct).
   Fix applied: gen_traces gained a `teacher_max_model_len` param (4k teachers).

   --- original plan (for provenance) ---
   **★ in-family native-Qwen weak teacher (the real fix).** Root
   cause of no-reproduction is likely a TEACHER-FAMILY/STYLE mismatch: Yuan keeps
   teacher+student in-family (both Qwen2.5, native-Qwen reasoning) so the weak
   teacher's CoT is in the student's latent distribution → clean elicitation. Our
   R1-Distill teacher is Qwen-based but imparts a FOREIGN DeepSeek-R1 over-thinking
   style → poor elicitation + hard-to-distill long traces. We never tested a clean
   in-family teacher. DO: teacher `Qwen2.5-Math-1.5B-Instruct` (native-Qwen, clean,
   weak) → student `Qwen2.5-Math-7B` (4k ctx → gen max_tokens≈2048, train
   max_seq_len 4096, gate max_model_len 4096). teacher_system="You are a helpful
   assistant." (Qwen chat). MATH L3-5 or L5. gen_traces → train → gate vs
   unelicited baseline. This is the most faithful-to-Yuan + most likely to gain.
   ⏳ IN PROGRESS: gen_traces (teacher Qwen2.5-Math-1.5B-Instruct, max_tokens 2048,
   L3-5) → /vol/traces/w2sr_infamily. NOTE: first launch CRASHED — gen_traces
   hardcoded teacher max_model_len=8192 but Qwen2.5-Math-1.5B-Instruct is 4k-ctx
   (max_position_embeddings=4096). FIXED: added `teacher_max_model_len` param to
   gen_traces (default 8192); relaunched with --teacher-max-model-len 4096.
   ON RESUME: check manifest (degeneracy should be LOW — native-Qwen clean), then
   `train --base-student Qwen/Qwen2.5-Math-7B --train-json
   /vol/traces/w2sr_infamily/train.json --out-dir /vol/checkpoints/w2sr_infamily
   --max-seq-len 4096` → `gate ... --max-model-len 4096 --max-tokens 3500`.
0b. **★★ HEADROOM FINDING (findings_repro.md Finding 5) — drives the extension.**
   Added a headroom probe to the gate (zero-shot-CoT untrained baseline). Result:
   the W2SR gain is GENUINE on the BASE model (Math-7B: cot-prompt 0.24 → W2SR
   0.645, +0.405 beyond prompting) but an ARTIFACT on the Instruct model
   (7B-Instruct: cot-prompt 0.63 = W2SR 0.63, +0.0 beyond prompting). So:
   - Reproduction (course req) = GENUINE, on the base model. DONE.
   - Monitorability student = locked 7B-Instruct (GPQA-capable, matches cond-1),
     run as a CAPABILITY-CONTROLLED faithfulness study ("monitorability changes
     without capability gain", reviewer-authorized). cond-2 W2SR student already
     trained: /vol/checkpoints/w2sr_infamily_inst (7B-Instruct on Math-1.5B-Inst).
1. **Strong-teacher distillation control (cond-3)** — IN-FAMILY to keep E2
   (W2SR−control) confound-free: strong teacher `Qwen2.5-Math-7B-Instruct` on the
   SAME MATH L3-5 problems → SFT `Qwen2.5-7B-Instruct`. (NOT OpenR1/R1-32B: a
   foreign-family control would reintroduce the very style confound Finding 4/5
   identified.) ⏳ gen_traces → /vol/traces/w2sr_infamily_strong LAUNCHED. Then
   train --base-student Qwen/Qwen2.5-7B-Instruct --out-dir
   /vol/checkpoints/w2sr_control_inst → gate --headroom-probe. Larger in-family
   strong teacher (72B) = exploratory dose-response (spec 5.3).
2. **EXTENSION (the actual paper):** monitorability eval (configs/*.yaml drive
   external/monitorability-eval via batch_eval.py; see scripts/run_*_teacher.sh
   for the 6-pass pattern: baseline pass + 5 adaptive-cue passes + extract_metrics).
   - Condition 1 (baseline 7B-Instruct student): DONE — results/baseline_7b_metrics/
     (acc 0.369, verbosity 0.476, faithfulness 0–4.5%/cue).
   - Condition 4a (weak teacher R1-1.5B): DONE — results/weak_teacher_metrics/
     (verbosity 0.640, faith 0.217/0.015/0.061/0.136/0.0). Existing result valid;
     the 30k re-run never persisted (no logs/ dir) — optional cosmetic redo only.
   - Condition 4b (strong 32B teacher, OpenRouter): INCOMPLETE (was ~40/198
     baseline). Re-run scripts/run_strong_teacher.sh (needs OpenRouter $).
   - Conditions 2 & 3 (W2SR + control student MONITORABILITY): ✅ DONE.
     cond-2 = w2sr_infamily_inst (weak Math-1.5B-Inst teacher), cond-3 =
     w2sr_control_inst (strong Math-72B-Inst teacher), both 7B-Instruct, merged →
     served via Modal vLLM → 6-pass GPQA eval. Results in findings_extension.md +
     results/{w2sr_student,control_student}_metrics (gitignored). HEADLINE:
     faithfulness NULL (E2 W2SR−control Δ=−0.006 p=.13, all at floor, H0/H3);
     verbosity TRANSFERS large (E1 +0.264). Distillation transfers verbosity not
     cue-faithfulness; teacher's 0.172 faith not inherited; teacher strength
     irrelevant. Scoring fix: MATH-SFT students emit \boxed (patch_meek_eval.py).
   - Condition 4b (strong 32B teacher ref): OPTIONAL remaining — rounds out the
     teacher-strength reference axis (cond-4a weak done: faith 0.172). Core study
     complete without it. scripts/run_strong_teacher.sh (OpenRouter $).
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
