# LOG — last / next / blockers

Daily log per spec 15. Newest entry on top.

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
