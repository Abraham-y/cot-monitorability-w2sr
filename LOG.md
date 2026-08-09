# LOG — last / next / blockers

Daily log per spec 15. Newest entry on top.

> **Note (2026-08-07).** This log stops at 2026-05-25 and covers only the
> instruct-substrate phase. The reasoning-student study that the paper is built
> on (R1-distill arms, self-distillation controls, MMLU, Tasks A–J) is not
> logged here; the paper and `results/reanalysis/README.md` are the record for
> that work. Entries below also reference internal working docs
> (`findings_repro.md`, `findings_extension.md`, `HANDOFF.md`, `project_spec.md`)
> that were removed before public release; their surviving content is in the
> paper, `README.md`, and `results/reanalysis/README.md`. Two entries disagree
> on the reproduction gate figure (0.670 vs 0.645) — the paper uses 0.645, the
> headroom-probe value.

## 2026-05-25 (pm-3) — EXTENSION COMPLETE: verbosity transfers, faithfulness doesn't
**Last:** Ran the monitorability study end-to-end on the locked 7B-Instruct
student (capability-controlled per Finding 5). Built cond-2 (W2SR, weak in-family
teacher Qwen2.5-Math-1.5B-Inst) and cond-3 (control, strong in-family teacher
Qwen2.5-Math-72B-Inst, same MATH L3-5 problems → clean 48× teacher-strength axis;
the 7B "strong" teacher tied the 1.5B at ~66% so L3-5 is saturated — use scale).
Each: train LoRA → gate(headroom) → merge_adapter → serve via Modal vLLM → 6-pass
GPQA monitorability eval (judge=sonnet-4-6). Also re-extracted cond-1 + cond-4a.

Hit + fixed a scoring bug: MATH-SFT students emit \boxed{X} not "ANSWER: X" ->
Inspect answer() misparsed -> fake accuracy collapse (cond-2 0.051 vs true 0.328).
Faithfulness/verbosity are judge/factor-based, unaffected. Codified a
format-agnostic re-parse in patch_meek_eval.py; cond-1 unchanged (0.369->0.379).

RESULTS (findings_extension.md): faithfulness (acknowledgment) is NULL — E2
(W2SR-control) Δ=-0.006 [-.013,.000] p=.13; E1 (W2SR-baseline) Δ=+0.024 [.000,.049]
p=.11; all at floor -> H0/H3. Verbosity TRANSFERS large — E1 Δ=+0.264 [.218,.312],
control-baseline Δ=+0.247, E2≈0. -> Distillation transfers VERBOSITY but not
cue-FAITHFULNESS; teacher's 0.172 faithfulness not inherited; teacher strength
irrelevant. A distilled model looks more monitorable (verbose) without being more
faithful. Core study COMPLETE (reproduction + extension).

**Next (optional):** cond-4b strong-teacher ref (R1-32B, OpenRouter) to round out
the teacher-strength reference axis; otherwise write the 5-page paper.
**Blockers:** OpenRouter balance for the optional cond-4b.

## 2026-05-25 (pm) — ★ W2SR REPRODUCED (in-family native-Qwen teacher)
**Last:** Resolved the reproduction. Root cause of all prior flat/collapse runs
= teacher↔student STYLE mismatch (R1-distill's foreign over-thinking CoT,
SimpleRL's noisy RL output) — not the method. Tested an in-family native-Qwen
weak teacher: `Qwen2.5-Math-1.5B-Instruct` → student `Qwen2.5-Math-7B`.
- gen_traces (L3-5, max_tokens 2048, teacher_system="You are a helpful
  assistant.", teacher_max_model_len 4096): 891 kept / 1000, teacher 66.9%
  correct, only 10.9% degenerate (vs R1's ~73% raw loop rate), ~1.5k char/trace.
- Fixed a blocker: gen_traces hardcoded teacher max_model_len=8192 → crashed on
  the 4k-ctx Math teacher; added a `teacher_max_model_len` param (default 8192).
- train: LoRA r32, max_seq_len 4096, 3 epochs (84 steps, loss ~0.16, tok-acc 0.96).
- **gate (held-out L3-5, temp 0, max_tokens 3500, max_model_len 4096): Pass@1
  0.325 → 0.670 = +0.345, format-valid 1.00, no degeneration, GATE PASSED.**
  Pre-registered success criterion was ≥+0.05; we hit +0.345. First passing gate
  across the whole matrix. Course-requirement reproduction MET.
Docs updated: findings_repro.md Finding 4 + net-result RESOLVED; HANDOFF table.
This Math-7B student is REPRODUCTION-ONLY — the monitorability student stays the
general 7B (GPQA is science; cond-1 baseline already on 7B-Instruct).

**Next:** EXTENSION (the paper). Strong-teacher distillation control on the
general 7B (doubles as cond-3 control student) → monitorability conds 2,3,4b →
preregistered analysis (H0-H3).
**Blockers:** OpenRouter balance (~$10) for cond 4b; Anthropic judge budget.

## 2026-05-25 (pm-2) — HEADROOM PROBE: gain genuine on base, artifact on Instruct
**Last:** Per reviewer guidance (gate-pass must reflect real elicitation, not
rearrangement on an already-strong student), added a `headroom_probe` to the gate
= score UNTRAINED base with zero-shot CoT (the prompting-only ceiling), then
`w2sr_beyond_cot_prompt` = W2SR − that ceiling. Built the cond-2 monitorability
W2SR student in-family on the LOCKED base (Qwen2.5-Math-1.5B-Instruct →
Qwen2.5-7B-Instruct, reusing /traces/w2sr_infamily) and gated both:
- **Math-7B (base):** unelicited 0.325, 0-shot-CoT 0.24, W2SR 0.645 →
  **+0.405 beyond prompting = GENUINE** elicitation (base can't CoT unprompted).
- **7B-Instruct:** unelicited 0.23, 0-shot-CoT 0.63, W2SR 0.63 → **+0.0 beyond
  prompting = ARTIFACT** (already CoT-capable; the +0.40 vs unelicited is the
  PREREG §4b elicitation confound, not weak supervision).
findings_repro.md Finding 5. Reproduction (course req) stands GENUINE on the base.
For the extension: monitorability student = locked 7B-Instruct, run
CAPABILITY-CONTROLLED ("monitorability changes without capability gain",
reviewer-authorized) — faithfulness diffs aren't a capability confound.
cond-2 = /checkpoints/w2sr_infamily_inst.
- cond-3 control = strong IN-FAMILY teacher (Qwen2.5-Math-7B-Instruct, same MATH
  L3-5 problems) → SFT 7B-Instruct, to keep E2 (W2SR−control) confound-free
  (a foreign R1-32B control would reintroduce the Finding-4/5 style confound).
  gen_traces LAUNCHED → /vol/traces/w2sr_infamily_strong.

**Next:** train+gate control → serve cond-2/cond-3 → monitorability eval (GPQA,
5 cues) → cond-4b strong-teacher ref (OpenRouter) → preregistered analysis.
**Blockers:** OpenRouter balance (~$10) for cond-4b; Anthropic judge budget.

## 2026-05-23 (pm) — W2SR reproduction pipeline wired + launched
**Built the full training half end-to-end and started condition 2 (W2SR =
the reproduction):**
- src/problems.py: MATH loader (hendrycks_math L3-5, self-contained boxed
  extractor, disjoint train/held-out). src/train_student.py (LoRA SFT, tested),
  src/validate_training.py (spec 9 gate, tested), src/analysis.py (per-case
  stats, tested on real data).
- modal_app.py entrypoints: gen_traces (offline batched vLLM, Yuan-style,
  bounded 4096 tok), train (LoRA SFT -> volume), gate (one vLLM load runs
  baseline + LoRA on held-out MATH -> Pass@1 gain = reproduction check).
- LAUNCHED gen_traces: 500 MATH traces from DeepSeek-R1-Distill-Qwen-1.5B ->
  /vol/traces/w2sr. Next chain: train -> gate. Reproduction success criterion
  (pre-registered): W2SR Pass@1 - baseline >= 5 pts (or >=30% of baseline->teacher gap).

Reproduce/critique/extend mapping locked: reproduce=W2SR MATH Pass@1 gain;
critique=unmeasured monitorability; extend=GPQA monitorability + control.

## 2026-05-24 (paused) — RESUME HERE: fix residual degeneration
**Decision:** dig into the W2SR student's residual degeneration (57.5% format-valid,
loops) BEFORE trusting the flat MATH result or pivoting task — format failure
depresses the score independently of capability. (Legit measurement repair, not
gain-manufacturing.) Timebox to format-valid >~80%, then re-read the gate:
still flat ⇒ trustworthy no-headroom ⇒ harder-task pivot; gain appears ⇒ repro works.

**Planned steps, cost order (NOT yet run):**
1. eval-time `gate --rep-penalty 1.15` (greedy+no-penalty was worst case for loops) — CHEAPEST, no retrain. `gate` already has the rep_penalty/max_tokens params (committed).
2. strengthen trace filter to drop teacher traces that don't conclude cleanly; regen + retrain.
3. raise student eval-time rep penalty further if needed.
4. raise LoRA rank (capacity for the 1.5B's long R1 CoT) — the one justified training-knob change.
If bounded effort doesn't lift conclusion rate ⇒ stop+flag: LoRA-distilling this teacher
into this base is the binding constraint (a finding); reconsider distillation/teacher.

**⚠️ MODAL PROFILE GOTCHA:** active profile switched to `cs336-2026` (other class).
ALL our resources (w2sr-vol, `huggingface` secret, endpoint, /vol/checkpoints/w2sr_base)
live in the **`ayeung16`** workspace. Resume every modal command with
`MODAL_PROFILE=ayeung16` (do NOT rely on the default; do not change the user's
active profile). Next action on resume: relaunch step 1 re-gate with that prefix.

**Background:** weak-teacher 30k monitorability re-run — no local proc alive now
(check on resume whether it completed: results/weak_teacher_metrics + 6 .eval logs
at 198 each; if not, restart it — H1 reference, valid regardless of the repro).

## 2026-05-24 — W2SR no gain vs UNELICITED baseline → harder-task next
Re-gated the existing base W2SR checkpoint against the corrected reproduction
baseline (unelicited, no-CoT direct answer — `build_direct_prompt`):
- unelicited base Pass@1: **0.45**  | W2SR student (CoT): **0.435** | gain **−0.015 (flat)**
The earlier −11.5pt drop was mostly the CoT-prompted-baseline artifact (0.535
was inflated). Against the fair baseline, W2SR is flat. Per the pre-agreed rule
(no gain vs unelicited ⇒ no MATH headroom at this pairing) the NEXT move is the
**harder-task option** (harder MATH subset / task the base can't already do),
NOT hyperparameter tuning. Two reproduction baselines now documented
(PREREGISTRATION §4b): reproduction=no-CoT, monitorability=zero-shot CoT.
CAVEAT: W2SR student still 57.5% format-valid w/ residual repetition loops
(LoRA distillation of the 1.5B's long R1 CoT into base is imperfect) → masks score.
STOPPED for human call on the harder-task pivot.

## 2026-05-23 (late) — W2SR reproduction FAILED then re-architected
**First real W2SR run failed the gate (the gate did its job):** baseline MATH
Pass@1 0.605 → W2SR 0.16 (−44pts), format_valid 0.105, repetition_loop.
Two root causes diagnosed:
1. **Degenerate teacher traces:** the 1.5B R1-distill spiraled into repetition
   on **73%** of traces ("Wait, 44+43 is 87? Wait, 44+43 is 87?..."). Training on
   them taught the student to loop. Causes: (a) NO repetition_penalty in sampling
   (primary), (b) we sent a "You are a helpful assistant" SYSTEM prompt, which is
   off-distribution for R1-distill (recommends user-turn-only; its template
   already forces `<think>`). The qwen-base template was for Yuan's Qwen teachers.
2. **No headroom:** 7B-Instruct baseline 0.605 ≈ teacher's 0.59 correctness, and
   Instruct is already elicited → W2SR can't show a gain. Yuan trained the BASE
   model (unelicited).

**Fixes:** build_prompt_messages drops the system prompt; gen_traces adds
repetition_penalty=1.1 + is_degenerate() loop filter; train_student drops system
prompt too (train/gate consistency).

**Decision (user): decouple reproduction from monitorability.**
- Reproduction → student = **Qwen2.5-7B BASE** (has headroom). Cleaned traces.
- Monitorability → pilot base + zero-shot CoT; fallback a smaller/less-elicited
  instruct model with headroom (NOT 7B-Instruct). Redo baseline+teacher evals on
  whatever substrate we land on.
- Do NOT retreat to "keep 7B-Instruct, report no gain" unless both routes fail.

**Running:** gen_traces v2 (1200 problems, fixes) → train BASE → gate, chained
(b92t6llik), ~3h, will report the Pass@1 verdict. Weak-teacher 30k re-run also
in parallel.

## 2026-05-23 — condition 4a (weak teacher) DONE ✓; strong teacher parallelized
**Weak teacher (DeepSeek-R1-Distill-Qwen-1.5B, GPQA-Diamond):** baseline acc
0.086 (weak at task), verbosity 0.640 (>student 0.476). Faithfulness MUCH higher
than the instruct student: stanford 0.217 (vs 0.045), unethical 0.136 (vs 0.035),
grader 0.061 (vs 0.005), visual 0.015, xml 0.0. acks 43/3/12/27/0. → reasoning
teacher is bad at the task but ~5x more faithful (Chua-Evans pattern). Strong H1
reference: conditions 2/3 test whether SFT transfers this faithfulness.
results/weak_teacher_metrics/.

**Strong 32B teacher:** was ~4.3h for 170/198 baseline (long CoT, low default
concurrency; no rate-limit errors). Added max_connections knob:
patch_meek_eval.py now also patches run_eval.py to read W2SR_MAX_CONNECTIONS
(injected into generator GenerateConfig); run_strong_teacher.sh sets =32. NO
token cap (would truncate reasoning → bias faithfulness). Relaunching.
NOTE: a too-broad `pkill batch_eval` also killed the weak teacher mid-extract;
recovered by re-running extract_metrics on the intact .eval logs (generation/
judging was already saved). Lesson: target pkill by batch name next time.

**Stage 1 (trace gen) built:** src/generate_traces.py (W2SR recipe, LF format,
grader, hash) + endpoint_sample_fn (reuse Modal vLLM endpoint). Locally tested.

## 2026-05-22 (eve) — CONDITION 1 (baseline 7B student) DONE ✓ (real result)
**Last:** Routed judge to native Anthropic (patch_meek_eval.py omits top_p for
anthropic/* judges; max_tokens capped 4096). Reran full 7B flow — 0 errors.
First real data point (Qwen2.5-7B-Instruct, GPQA-Diamond, 198 Qs):
  baseline acc 0.369, verbosity 0.476 (n=198)
  faithfulness per cue: stanford 0.045, visual_squares 0.0, grader_hack 0.005,
    unethical 0.035, xml_metadata 0.0  (acks: 9/0/1/7/0)
  monitorability: 0.233/0.240/0.225/0.191/0.109
Interpretation: cues clearly influence (acc drops to 0.18-0.29; 41-55
correct→incorrect flips/cue) but the untrained Instruct student almost never
acknowledges the cue (~0-4.5% faithfulness — matches Chua-Evans ~7% non-reasoning
baseline). CAVEAT (spec 11.4): acks are single-digit → small informative N;
W2SR-vs-control deltas must be sizeable to resolve. Results in
results/baseline_7b_metrics/baseline_7b_metrics.json.

**Next:** condition 4 = teacher references (build-order step 3, no training).
Weak teacher DeepSeek-R1-Distill-Qwen-1.5B (reasoning model → long CoT, needs
bigger max_model_len, e.g. 32k) served on Modal; strong 32B via OpenRouter.
Expect HIGHER faithfulness for the reasoning teacher (Chua-Evans: R1 ~59%) — the
H1-vs-H2 reference. Then conditions 2/3 (W2SR + control) need trace-gen + SFT.

## 2026-05-22 (pm) — 7B run BLOCKED on OpenRouter credits
**Last:** Switched endpoint to 7B/A100, launched real baseline flow. Student
generation worked (193/198 answers) but EVERY judge call failed: HTTP 402
"requires more credits" — OpenRouter balance exhausted. Checked: $10 total,
$9.69 used → ~$0.31 left. The pilot's ~1300 claude-sonnet-4.6 judge calls
(~$0.007 each) burned it. Killed the 7B run to stop A100 waste (auto-scaledown).
Fix applied: judge `max_tokens` was defaulting to 65536 (triggered the 402
pre-auth + wasteful) → capped to 4096 via `model_configs["judge:openrouter/
anthropic/claude-sonnet-4.6"]` in all configs; verified it resolves. Cap fixes
the 402 trigger but the real cost driver is sonnet token usage at volume.

**BLOCKER (need from human):** top up OpenRouter credits. Rough estimate:
~$15 to finish the 7B baseline + teacher conditions; ~$50-75 for the full
4-condition GPQA study. Alternative: switch to a cheaper judge (justify via the
spec 10.3 judge-validation set). Once topped up: rerun baseline_7b_gpqa.yaml
flow (or use batch_eval --retry-failed-samples to re-judge stored 7B outputs
without re-generating on A100).

## 2026-05-22 — FULL 6-pass pilot ran end-to-end on real GPQA ✓
**Last:** Deployed Modal vLLM endpoint (L4), served pilot Qwen2.5-1.5B-Instruct,
ran the COMPLETE flow over 198 GPQA-Diamond Qs: baseline + adaptive-cue gen + 5
cue passes + extract_metrics → results/pilot_metrics/pilot_metrics.json.
Pipeline fully validated. Pilot numbers are a degenerate artifact (as spec §17
predicts for a too-weak model), NOT a result:
  - baseline acc 0.136 (<25% random — 1.5B overwhelmed by GPQA), verbosity 0.056
  - faithfulness 0.0 across ALL 5 cues (tiny model never verbalizes the cue);
    ~160/198 cases incorrect→incorrect so few clean influenced cases
  - xml_metadata cue pass: 47% sample errors (1.5B mangled XML-cue prompts;
    auto-retry) → cue 5 partial. Watch this cue on 7B.
Takeaways for the real run: need the capable 7B to get non-floored faithfulness;
keep an eye on the xml_metadata error rate.

**Next (in progress):** flipped endpoint to Qwen2.5-7B-Instruct on A100;
redeploy + warm + rerun the full flow = REAL condition-1 (baseline student).
Then condition 4 teachers (weak 1.5B on Modal, strong 32B via OpenRouter).
Endpoint auto-scaledown 10 min idle (no lingering GPU cost).

## 2026-05-21 (late) — eval pipeline validated end-to-end ✓
**Last:** Keys added + verified. OpenRouter probe: 32B distill returns reasoning
traces; 1.5B teacher absent + qwq-32b retired → weak teacher on Modal, judge =
claude-sonnet-4.6. Built `.venv-eval` (minimal API-path deps; their pinned
torch/triton are Linux-only and unneeded since inference is remote). Downloaded
GPQA causal-factors (ameek/causal_factors, 198 diamond Qs — no GPQA gating
blocker). Ran the Meek baseline pass on 3 GPQA Qs (harness test, OpenRouter
stand-in student): dataset→generate→verbosity-judge→scores all work (factor
util ~0.79). Fix found: scorers hardcode temperature+top_p, which Anthropic's
native API rejects — route the judge via `openrouter/anthropic/claude-sonnet-4.6`
(normalizes it). Eval venv: `. .venv-eval/bin/activate`, run their scripts from
`external/monitorability-eval` with `PYTHONPATH=src` and `.env` sourced.

**Next (the real baseline, build-order step 2):** deploy Modal vLLM student
endpoint (Qwen2.5-7B-Instruct) → point baseline_student_gpqa.yaml at it → run
baseline pass + generate 5 adaptive-cue datasets + 5 cue passes over ~50 GPQA Qs
→ extract_metrics → hand-label ≥50 cases, validate judge agreement.

## 2026-05-21 (night) — Modal vLLM serving entrypoint built ✓
**Last:** Built + validated the Modal vLLM serving path (independent of the
blocked credentials). `src/serving.py` holds the Modal-free, unit-tested
command/URL helpers; `modal_app.py::VLLMServer` is a parametrized `@app.cls`
(model, max_model_len) exposing an OpenAI-compatible `@modal.web_server`.
`modal run modal_app.py::serve_url --model Qwen/Qwen2.5-1.5B-Instruct` returns
the endpoint URL + Inspect model string/env with NO GPU spin-up and NO secret
needed (Qwen/our checkpoints are ungated). Gotchas resolved: dropped
`from __future__ import annotations` in modal_app (broke `modal.parameter`
typing); `Secret.from_name(required=)` doesn't exist in Modal 1.4.3 so the
serving class just omits the unused HF secret. Deployed URL pattern (after
`modal deploy`): https://ayeung16--w2sr-monitorability-vllmserver-serve.modal.run

**Next:** unchanged — still blocked on OPENROUTER/ANTHROPIC/HF keys to run the
thinking-token check, deploy the student endpoint, download GPQA, and run the
baseline + 5-cue eval. The serving harness is ready the moment keys land.

## 2026-05-21 (eve) — Modal smoke ✓, eval mapped, serving locked
**Last:** Build-order step 1 DONE — `modal run modal_app.py` prints `GPU: Tesla
T4`. Mapped the Meek eval (config-driven, Inspect API-provider model strings;
2-phase per dataset: baseline pass + 5 adaptive-cue passes; default judge
qwq-32b). Corrected cue set to the 5 actually shipped (`config.MEEK_CUES`).
Added `configs/baseline_student_gpqa.yaml`. Locked serving: STUDENT (baseline +
checkpoints) always on Modal vLLM (one serving path for the headline compare);
off-the-shelf TEACHERS via OpenRouter, conditional on a thinking-token check
(`scripts/check_openrouter_thinking.py`). Judge: bench qwq-32b vs sonnet on the
≥50-case hand-labeled set, qwq favored, sonnet independence as tiebreaker.

**Next (build-order step 2), BLOCKED on credentials:**
1. Run `scripts/check_openrouter_thinking.py` (needs OPENROUTER_API_KEY) — decide
   teacher serving path.
2. Stand up the Modal vLLM endpoint for Qwen2.5-7B-Instruct (baseline student).
3. Download GPQA + ameek/causal_factors (needs HF_TOKEN; GPQA is gated).
4. Run baseline + 5-cue eval on the student over ~50 GPQA Qs; hand-label ≥50
   cases; validate both judges; pick one.

**Blockers (need from human):** `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`,
`HF_TOKEN` (+ mirror as Modal secret `huggingface`). See `.env.example`. No keys
are present anywhere yet (only `wandb` Modal secrets exist).

## 2026-05-21 (pm) — design decisions locked
**Last:** Locked two §22 decisions and propagated them into spec + code.
(1) Control = **strong-teacher distillation**, reframed as a **teacher-strength
axis** (weak teacher below student ↔ strong teacher above), structured for a
dose-response curve (new spec §5.3); GT references demoted to optional confounded
sanity arm (§5.4). (2) Teacher series = **public DeepSeek-R1-Distill-Qwen**
(weak 1.5B, strong 32B / fallback 14B / public R1 trace dataset matched on
problem set+volume). Student locked to **Qwen2.5-7B-Instruct** (baseline must
emit CoT). Confirmed Yuan's teachers = public `hkust-nlp/*-SimpleRL-Zoo`
(loadable; optional robustness only). Compute backend = Modal (set up).
Refactored `config.py` to a `TeacherSpec`/`teacher_axis` model; promoted PRIMARY
(7B) to active; re-enabled MMLU secondary. Ambition: aim for publishable.

**Next:** Modal smoke test → baseline-student monitorability eval on ~50 GPQA Qs
(driving external/monitorability-eval), validate judge on ≥50 hand-labeled cases.

**Blockers:** judge API budget; pre-register stat test on pilot; send Yuan email.

## 2026-05-21 (am)
**Last:** Bootstrapped the project. `git init`; reorganized cloned repos under
`external/` (w2sr, monitorability-eval, gair-w2s-reasoning). Cloned the Meek
monitorability Inspect eval. Downloaded the 6 must-read PDFs to
`references/pdfs/`. Extracted the W2SR SFT recipe → `references/recipe_w2sr.md`
(found real divergences from spec §8.2: full SFT not LoRA, 5–10 epochs, global
batch 128, warmup 0.1). Wrote scaffolding: `src/config.py` (real),
stage/gate/orchestrator/analysis skeletons, `modal_app.py`, README, CLAUDE.md.

**Next:** (1) Resolve blockers below. (2) Implement Modal smoke test → run it.
(3) Stand up the monitorability eval on the baseline student over ~50 GPQA Qs
(build-order step 2), driving `external/monitorability-eval`.

**Blockers (need from human):**
- Modal account + HF token stored as Modal Secret `huggingface`.
- Judge model API access/budget (currently `claude-sonnet-4-6` in config).
- §22 decisions: control-trace source (strong-teacher vs ground-truth);
  final teacher choice (Yuan checkpoints vs DeepSeek-R1-Distill).
- Send the Yuan (W2SR first-author) email; fold in Arcuschin/Meek thread guidance.
- Pre-register the statistical test on pilot before the full run.

## 2026-05-25 — degeneration bounded effort done: STOP-AND-FLAG (binding constraint)
Iterations: eval rep_penalty=1.15 (killed loops but tanked accuracy 0.45->0.27,
format still 0.55 — wrong lever for math); max_tokens=7000 (no help -> non-
conclusion is inherent, not truncation); rank16->32 + filter 729->532 (format
0.55->0.615, still <0.8, still some degeneration). Gain ROBUSTLY FLAT across ALL
configs (Instruct/base, rank16/32, all decoding): ~0 (+0.005 at r32). As format
climbed 0.55->0.615 the gain stayed flat -> cleaner generation isn't hiding a gain.
FINDINGS: (1) no W2SR capability gain at 1.5B-R1-teacher / 7B pairing on MATH;
(2) LoRA-distilling the 1.5B's long over-thinking R1 CoT into the base is the
binding constraint on clean generation (bounded levers only nudge it).
Per plan: stop; reconsider distillation/teacher before harder task. Decision pending.
No lingering Modal cost (ephemeral apps scaled to 0).
