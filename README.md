# Does Weak-to-Strong Reasoning Preserve Chain-of-Thought Monitorability?

CS 338 class project. We measure whether a strong student SFT'd on a weak
reasoner's chain-of-thought (W2SR; Yuan et al. 2505.20072) keeps a faithful,
monitorable CoT — the property CoT-monitoring safety schemes depend on. Yuan
et al. measured task accuracy only; we add the monitorability layer (Meek et
al. 2510.27378) plus a control condition that isolates *weak supervision* from
*SFT-on-CoT in general*.

Full spec: [project_spec.md](project_spec.md). Solo, but aimed at eventual
publishability. The core object is a **teacher-strength axis**: a fixed student
(`Qwen2.5-7B-Instruct`) is SFT'd on CoT from teachers spanning capability —
weak `DeepSeek-R1-Distill-Qwen-1.5B` (W2SR) up to strong `...-32B`
(distillation/control). Plotting monitorability vs teacher strength gives a
dose-response curve (spec 5.3); the control is its strong-teacher anchor.

## Layout
```
src/            config + 3 pipeline stages + gate + orchestrator + analysis
modal_app.py    Modal image/volume/secrets + GPU entrypoints (spec 12)
references/     notes.md, recipe_w2sr.md, pdfs/ (gitignored)
external/       cloned repos, read-only refs (gitignored)
data/ results/  gitignored; live on the Modal Volume in practice
```

## Pipeline (spec 7)
1. `generate_traces` — teacher CoT traces → dataset
2. `train_student` — SFT (W2SR + control via config), then the gate
3. `evaluate_monitorability` — Meek Inspect eval (cues + judge)
+ `validate_training` (gate), `run_matrix` (orchestrate), `analysis` (stats/plots)

## Status
Scaffolding only — stages are documented skeletons (`NotImplementedError`).
See [LOG.md](LOG.md) for last/next/blockers. Recipe ground truth in
[references/recipe_w2sr.md](references/recipe_w2sr.md).

## Setup (once compute is decided)
```
python -m venv .venv && source .venv/bin/activate
pip install -e .
modal run modal_app.py::smoke   # spec 15.1 plumbing test (needs Modal acct)
```
Requires: Modal account, HF token as Modal Secret `huggingface`, judge API access.
