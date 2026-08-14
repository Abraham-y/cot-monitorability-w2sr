# Math-CoT Distillation Suppresses Cue Acknowledgment Beyond What Compression Explains

> Empirical study of CoT monitorability under math-CoT SFT on
> DeepSeek-R1-Distill students. Headline is an **intervention**: with the
> reasoning span genuinely in the SFT loss, cue acknowledgment falls
> $25.0\% \to 14.4\%$ (paired McNemar $p = 0.005$) while the cue's behavioral
> pull is unchanged ($\Delta = +0.048$, $p = 0.52$) and held-out MATH accuracy
> rises ($0.650 \to 0.685$) — at traces only ~27% shorter than baseline, so
> compression cannot account for the gap. A second arm, supervised on
> answer segments only, compresses CoT 14× and drives acknowledgment to
> $3.2\%$, bounding how much of the effect trace length alone produces.

**Paper:** [writeup_workshop.pdf](writeup_workshop.pdf)  ·  **arXiv package:** [arxiv_submission.zip](arxiv_submission.zip)

## TL;DR

SFT on math traces (R1-Distill-Qwen-7B student) — whether the
teacher is weaker (W2SR), stronger (KD from 14B), or the *same* student under
self-distillation — drops cue-acknowledgment from 25% to 2.4–7.4% while
leaving the cue's *behavioral* pull on the answer preserved or slightly
stronger. The model keeps being moved by the cue and stops saying so. The
effect is teacher-strength-invariant, survives three judges from three model
families (Sonnet as primary, Gemini and Kimi as out-of-family cross-checks)
and a within-judge rubric swap, and replicates on
MMLU (both STEM and math-only subsets). An inference-time few-shot ICL
prompt partially recovers acknowledgment on nameable authority cues
(Stanford-professor: 5% → 40%; the pooled ICL-vs-none recovery is
$p=0.008$) but not on subtler framings, so
training-side preservation is still needed. An accuracy-only certification
— or one that rewards elaborate-looking CoT — can pass a model whose
reasoning has gone behaviorally opaque while remaining superficially
legible.

> **Important caveat (see §Setup and §Limitations of the paper).** The
> DeepSeek-R1-Distill chat template strips everything up to `</think>` from
> assistant turns, so the SFT examples the students actually saw were the
> **final-answer segments** of the math traces, not the reasoning. The
> intervention studied here is therefore answer-only SFT on math traces. This
> explains the CoT compression and weakens the teacher-invariance evidence; it
> does not affect the behavioral measurements, the paired acknowledgment
> collapse, or the matched-length residual. `src/train_student.py` now renders
> the assistant turn without the template and hard-fails if the reasoning span
> is missing from a rendered example. **Task K re-runs the W2SR-weak arm with
> the fixed renderer**: compression mostly disappears (median CoT 13,721 chars,
> still ~27% below baseline's 18,847) and acknowledgment recovers only partially
> (3.2% → 14.4%, still below baseline's 25.0%, paired p = 0.005) at preserved
> GPQA accuracy — so the format explains most, but not all, of the collapse
> (`results/reanalysis/K_cotsft_rerun.*`).

## Headline numbers (all reproduce exactly from disk)

- **Headline (Task K, reasoning-preserving SFT):** ack **26/180 = 14.4%** vs
  baseline **40/160 = 25.0%**; paired **Δ = −0.110, $n = 155$, disc. 25/8,
  $p = 0.005$**; paired influence null (**Δ = +0.048, $n = 83$, $p = 0.52$**);
  median CoT 13,721 vs baseline 18,847 chars; held-out MATH 0.650 → 0.685
- Answer-only arm acknowledgment: **6/190 = 3.2%**
- Paired McNemar ack drop (answer-only): **Δ = −0.220, 34/1 discordant, $p \approx 2\times 10^{-9}$**
- Paired influence (answer-only): **Δ = +0.244, $n = 82$, $p = 0.0017$**
- Matched-length, cue-stratified (text cues only): long bin **44.0% vs 18.8%,
  OR ≈ 3.40, Fisher $p = 0.016$**
- Three-judge robustness (κ vs Sonnet):
  - Gemini 2.5 Pro: **overall κ = 0.68**, Δ = −0.153, $p = 2.4 \times 10^{-7}$
  - Kimi K2: **overall κ = 0.556** (stricter), Δ = −0.114, $p = 1.4 \times 10^{-4}$
- Matched-length residual on safety-relevant subset (influenced=1):
  **baseline 70.0% vs W2SR 21.1%, OR ≈ 8.75, Fisher $p = 0.003$**
- MMLU replication, all 5 STEM subjects:
  **baseline 27.4% (52/190) vs W2SR 3.6% (7/195), paired $n=185$,
  $\Delta = -0.232$, McNemar $p = 2.4\times10^{-10}$**
- In-domain math replication (`mmlu_college_mathematics`):
  **baseline 36.7% (11/30) vs W2SR 2.5% (1/40), paired McNemar $p = 0.006$**
- Inference-time ICL recovery: **1% → 9%, 8/0 discordant, $p = 0.008$**
  (recovery concentrates on Stanford-professor cue; grader-hack + insider-info
  do not recover)

Judge labels are not bit-reproducible: re-running the Task G rubric swap with
the same judge at $T=0$ moved baseline ack 9.4% → 8.8% and its $p$ from
$6\times10^{-5}$ to $1.2\times10^{-4}$ (same conclusion). The committed label
files are the ones the paper cites; paid re-judging is opt-in
(`REJUDGE=1`, `RUBRIC=1`).

The four core headline numbers are enforced by hard-fail `assert`s in
[`scripts/reanalysis/01_gate.py`](scripts/reanalysis/01_gate.py) — the script
exits non-zero if any drift.

## Layout

```
writeup_workshop.tex          — the paper (13 pages, NeurIPS 2025 preprint style)
writeup_workshop.pdf          — compiled paper
arxiv_submission.zip          — arXiv-ready bundle (tex + sty + fig)
neurips_2025.sty              — NeurIPS style file

results/figs/                 — figures (dissociation_bars, mechanism_and_recovery)
results/reanalysis/           — per-task result tables (.md) + numbers (.json)
                                + per-sample judge labels (.jsonl)
                                See results/reanalysis/README.md for the pipeline map.

scripts/reanalysis/           — the reproducibility pipeline
  01_gate.py                  — headline reproduction + extraction audit (hard-fail asserts)
  02_directional_influence.py — behavioral pull / flip-to-cue rate
  03_think_channel_collapse.py — </think> emission scan
  04_matched_length_logit.py  — matched-length residual logit
  05_robustness_rejudge.py    — out-of-family judge cross-check (Gemini, Kimi)
  A_cue_correct_confound.py   — cue-points-at-wrong stratum
  B_per_cue_effects.py        — per-cue paired tests
  C_decode_config_check.py    — greedy single-decode audit
  D_self_distillation_negcontrol.py — teacher-strength invariance
  E_cross_substrate_mmlu.py   — MMLU cross-substrate replication (pooled)
  E2_mmlu_by_subject.py       — MMLU split by subject (incl. in-domain college_math)
  F_llama_capability_gate.py  — cross-family Llama gate (failed null)
  G_robustness_rubric.py      — within-judge rubric swap
  H_length_binned_ack.py      — length-binned Fisher unpaired
  H_length_binned_ack_by_influenced.py — H split by influenced=1
  I_cot_conclusion_judge.py   — CoT-conclusion judge, first pass (Task I, needs OpenRouter)
  I2_cot_conclusion_with_question.py — I2 with answer choices in context
  I3_cot_conclusion_baseline_control.py — baseline control for I2
  J_inference_time_recovery.py — inference-time system-prompt recovery (Task J)
  K_cotsft_rerun.py           — CoT-preserving SFT rerun of the W2SR-weak arm (Task K)
  make_length_binned_fig.py   — length-binned figure
  make_mechanism_fig.py       — mechanism + recovery 2-panel figure
  run_all.sh                  — one command runs the whole free (no-API) suite
                                (paid steps are opt-in: REJUDGE=1, RUBRIC=1)

configs/                      — eval configs per condition
modal_app.py                  — Modal pipeline (trace gen, train, gate, merge, serve)
src/                          — training/eval/analysis helpers
external/                     — cloned Meek et al. eval framework (gitignored, needed for reanalysis)

LOG.md                        — chronological progress log
PREREGISTRATION.md            — locked pre-treatment analysis plan (2026-05-23)
references/notes.md           — one-liner references for cited papers
```

## Reproducibility

```bash
# Install
python3 -m venv .venv-eval && source .venv-eval/bin/activate
pip install scipy numpy pandas statsmodels matplotlib openai inspect-ai

# Run the entire free reanalysis suite (reads stored .eval files, no API cost)
bash scripts/reanalysis/run_all.sh

# Re-judge with Gemini or Kimi (~$1-2 via OpenRouter):
export OPENROUTER_API_KEY=...
python scripts/reanalysis/05_robustness_rejudge.py --run-rejudge --judge google/gemini-2.5-pro
python scripts/reanalysis/05_robustness_rejudge.py --run-rejudge --judge moonshotai/kimi-k2-0905

# Task I2/I3 CoT-conclusion analysis (~$1 via OpenRouter):
python scripts/reanalysis/I2_cot_conclusion_with_question.py --all
python scripts/reanalysis/I3_cot_conclusion_baseline_control.py

# Task J inference-time recovery (needs W2SR model on Modal + judge; ~$5-10):
modal deploy modal_app.py
python scripts/reanalysis/J_inference_time_recovery.py --n 100
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
- Downloaded reference PDFs (`references/pdfs/`)

## Trained arms served via Modal (`w2sr-vol:/merged/`)

- `w2sr_r1_7b` — W2SR weak (R1-Distill-Qwen-1.5B → R1-Distill-Qwen-7B)
- `w2sr_r1_7b_strong` — strong-teacher KD (R1-Distill-Qwen-14B → 7B)
- `w2sr_r1_7b_self_A4k` — self-distillation, 4k budget
- `w2sr_r1_7b_self_B8k` — self-distillation, 8k budget
- `w2sr_llama_self_B8k` — cross-family Llama-8B self-distill (capability-gate fail)
- `w2sr_infamily_inst`, `w2sr_control_inst` — instruct-substrate arms (floored)

## Citation

> Yeung, A. "Math Distillation Decouples Chain-of-Thought from Behavior on a
> Reasoning Model." Technical report, 2026.

## History

- Started as Stanford CS 338 course project.
- Preregistered analysis plan on 2026-05-23 (see [PREREGISTRATION.md](PREREGISTRATION.md)).
- Rolling progress in [LOG.md](LOG.md).

## Status & limitations

Sole author. All errors are mine. Substantive limitations are stated in the
Limitations section of the paper: (a) **the supervision was answer-only** — the
student chat template stripped the reasoning span from the SFT examples (see the
caveat above), which explains the CoT compression and weakens the
teacher-invariance evidence; the Task K rerun with fixed supervision recovers
acknowledgment only partially (one arm, one seed, single judge), and the other
trained arms have not been rerun; (b) single model family — a same-recipe
cross-family attempt on R1-Distill-Llama-8B fails the capability gate cleanly;
(c) matched-length residual across all cued samples is marginal (the significant
residual is on the safety-relevant `influenced=1` subset, which conditions on a
post-treatment variable and is one uncorrected cell of a 2×5 grid); (d) LoRA
only, MATH-L3–5 SFT data only; (e) judge robustness checked with two
alternative families (Gemini, Kimi), not a full panel, and absolute rates are
judge-dependent; (f) inference-time mitigation (Task J) recovers only
on nameable authority cues, not on subtler framings; (g) training-side
mitigation not attempted; (h) the preregistered primary estimand
(W2SR − strong-teacher control) is null (p = 0.09) — the headline is the
preregistered secondary estimand, which is the prereg's H0 case; (i) the only
committed measurement of the preregistered held-out MATH capability gate for
the R1 W2SR-weak arm (re-gated alongside Task K,
`results/reanalysis/K_orig_w2sr_gate_report.json`) **fails** it (Pass@1 0.550
vs 0.695, repetition-loop flag), as does the Task K rerun's own gate report
(format-valid 0.80, gain +3.5pp) — capability-preservation claims are scoped
to the GPQA eval substrate (0.400 → 0.425), and this is disclosed as a
preregistration deviation in the paper.
