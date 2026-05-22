# LOG — last / next / blockers

Daily log per spec 15. Newest entry on top.

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
