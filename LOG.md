# LOG — last / next / blockers

Daily log per spec 15. Newest entry on top.

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
