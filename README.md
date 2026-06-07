# Looking Monitorable Without Being Faithful

> Empirical study of CoT monitorability under weak-to-strong reasoning (W2SR)
> fine-tuning on DeepSeek-R1-Distill students. Documents a
> **behavior–verbalization dissociation**: behavior preserved while CoT
> acknowledgment of cues collapses (25% → 3%, paired McNemar p ≈ 2×10⁻⁹).

**Paper:** [writeup_workshop.pdf](writeup_workshop.pdf)

## TL;DR

Fine-tuning a reasoning model (R1-Distill-Qwen-7B) on math-CoT — whether the
teacher is weaker (W2SR), stronger, or the *same* student under
self-distillation — drops cue-acknowledgment from 25% to 2.4–7.4% while leaving
the cue's *behavioral* pull on the answer preserved or slightly stronger.
The model keeps being moved by the cue and stops saying so. The effect is
teacher-strength-invariant, survives an out-of-family judge (Gemini) and a
within-judge rubric swap, and replicates on MMLU. An accuracy-only
certification — or one that rewards elaborate-looking CoT — can pass a model
whose reasoning has gone behaviorally opaque while remaining superficially
legible.

## Headline numbers (all reproduce exactly from disk)

- Baseline R1-7B acknowledgment: **40/160 = 25.0%**
- W2SR weak acknowledgment: **6/190 = 3.2%**
- Paired McNemar ack drop: **Δ = −0.220, n = 150, 34/1 discordant, p ≈ 2×10⁻⁹**
- Paired influence (behavior): **Δ = +0.157, n = 108, p = 0.021**
- Cross-judge robustness (Gemini, same rubric): **κ ∈ [0.65, 0.68], paired Δ = −0.153, p = 2.4×10⁻⁷**
- Cross-rubric robustness (alt prompt): **Δ = −0.10, p = 6×10⁻⁵, 15/0 discordant**

These four headline numbers (and the eight robustness numbers behind them) are
enforced by hard-fail `assert` statements in
[`scripts/reanalysis/01_gate.py`](scripts/reanalysis/01_gate.py) — the script
exits non-zero if any of them drift.

## Layout

```
writeup_workshop.tex         — the paper (6 pages)
writeup_workshop.pdf         — compiled paper
results/figs/                — figures
results/reanalysis/          — per-task result tables (.md) + numbers (.json)
                               + per-sample judge labels (.jsonl)
scripts/reanalysis/          — the reproducibility pipeline
  _common.py                 — pinned extractor / judge model / record loader
  01_gate.py                 — headline reproduction + extraction audit (hard-fail asserts)
  02_directional_influence.py — directional pull confound
  03_think_channel_collapse.py — </think> emission scan
  04_matched_length_logit.py — matched-length residual logit
  05_robustness_rejudge.py   — out-of-family judge (Gemini, same rubric)
  G_robustness_rubric.py     — within-judge rubric swap (NEW)
  A_cue_correct_confound.py  — cue-points-at-wrong stratum
  B_per_cue_effects.py       — per-cue paired tests
  C_decode_config_check.py   — greedy single-decode audit
  D_self_distillation_negcontrol.py — teacher-strength invariance
  E_cross_substrate_mmlu.py  — MMLU replication
  F_llama_capability_gate.py — cross-family Llama gate (failed, reported honestly)
  run_all.sh                 — one command runs the whole suite
configs/                     — eval configs per condition
modal_app.py                 — Modal pipeline (trace gen, train, gate, merge, serve)
src/                         — training/eval/analysis helpers
```

## Reproducibility

```bash
# Install
python3 -m venv .venv-eval && source .venv-eval/bin/activate
pip install scipy numpy pandas statsmodels matplotlib openai inspect-ai

# Run the entire reanalysis suite (free; reads stored .eval files)
bash scripts/reanalysis/run_all.sh

# To also re-judge the 525 R1-family CoTs under Gemini (~$2 via OpenRouter):
REJUDGE=1 bash scripts/reanalysis/run_all.sh
```

The `.eval` log files (multi-MB each) are NOT in this repo — they live on the
Modal Volume `w2sr-vol:/` plus `external/monitorability-eval/logs/` locally.
For inspection without re-running, see the per-task `.md`/`.json` files in
[`results/reanalysis/`](results/reanalysis/) and the per-sample judge labels
in the `.jsonl` files.

## What's NOT in the repo

- `.env` (API keys for Anthropic, OpenRouter — never committed)
- `*.eval` files (multi-MB; on Modal Volume + can be regenerated)
- `external/` (the cloned [Meek et al. monitorability-eval](https://github.com/anthropics/measuring-cot-monitorability) framework)
- LoRA checkpoints (~50MB each; on Modal Volume `w2sr-vol:/checkpoints/`)

## Citation

> Yeung, A. "Looking Monitorable Without Being Faithful: Math-CoT Fine-Tuning
> Degrades Cue-Faithfulness Independent of Teacher Strength." Technical
> report, 2026. Originated as Stanford CS 338 course project.

## Status & limitations

Sole author. All errors are mine. Substantive limitations are stated in §7 of
the paper: (a) single model family — a same-recipe cross-family attempt on
R1-Distill-Llama-8B fails the capability gate cleanly (Task F); (b)
matched-length residual is underpowered for magnitude; (c) LoRA only,
MATH-L3–5 SFT data only; (d) judge robustness checked with one alternative
family (Gemini), not a panel.
