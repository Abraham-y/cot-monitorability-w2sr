# W2SR / CoT-monitorability reanalysis — outputs

Standalone reanalysis of the GPQA monitorability runs, working entirely from
the stored inspect_ai `.eval` records under
[`external/monitorability-eval/logs/`](../../external/monitorability-eval/logs/)
and (for the think-channel task) the Modal volume `w2sr-vol`. No new model
completions are generated. The only paid step is Task 5 Part B (judge
re-call), which is opt-in via `REJUDGE=1`.

## Standing conventions (apply everywhere)

- **Extractor:** the patched `\boxed{X}` / `ANSWER: X` / "answer is X" letter
  extractor pinned at [`scripts/reanalysis/_common.py::EXTRACTOR_PATTERNS`](../../scripts/reanalysis/_common.py).
- **Denominators:** behavioral metrics use `has_cue=True ∩ parseable-answer`.
  Judge-label metrics use `has_cue=True ∩ non-null judge label`.
- **Pairing:** always on `(core_question_id, cue_name)`.
- **CIs:** Wilson 95% for proportions; exact-binomial bootstrap on paired
  differences.
- **Paired tests:** exact McNemar (`scipy.stats.binomtest`).
- **Attrition note** repeated in every task: baseline R1-7B drops 23% of cued
  samples to no-answer; trained students drop ≤6%. Paired tests condition on
  baseline finishing — that selection biases the W2SR-vs-baseline Δ
  conservatively (toward zero / against W2SR).

## How to run

```bash
# Cheap path: everything except the paid rejudge step
bash scripts/reanalysis/run_all.sh

# Full path: also re-judge 525 records under google/gemini-2.5-pro (~$2)
REJUDGE=1 bash scripts/reanalysis/run_all.sh
```

[`run_all.sh`](../../scripts/reanalysis/run_all.sh) sources `.venv-eval/` and
`.env`, pulls the four training-trace files from Modal if missing, and runs
the eight scripts in order. `01_gate.py` hard-fails (Python `assert`) if any
of the manuscript headlines below disagree with what disk reproduces; the
suite stops there.

## Per-task manifest

| # | Script | Inputs | Outputs | Headline |
|---|---|---|---|---|
| 1 | [01_gate.py](../../scripts/reanalysis/01_gate.py) | Cued + baseline `.eval` zips for all 6 batches (`r1_7b_baseline`, `r1_7b_w2sr`, `r1_7b_strong`, `baseline_7b`, `w2sr_student`, `control_student`) | [`01_gate.md`](01_gate.md), [`01_gate.json`](01_gate.json) | **Hard-fail asserts:** baseline R1-7B ack **40/160 = 25.0%**; W2SR weak **6/190 = 3.2%**; paired ack n=150, discordant **34/1**, McNemar p ≤ 1e-8 (got 2.1e-9), Δ = **−0.220**; paired influence n=108, discordant **16/33**, p ≤ 0.05 (got 0.021), Δ = **+0.157** |
| 2 | [02_directional_influence.py](../../scripts/reanalysis/02_directional_influence.py) | Same 6 batches, cued cells, restricted to `baseline_ans ≠ cue_target` | [`02_directional_influence.md`](02_directional_influence.md), [`02_directional_influence.json`](02_directional_influence.json) | R1 family: flip→cue_target = baseline **64.7%** / W2SR weak **72.7%** / W2SR strong **73.9%** (all p < 1e-7 vs 1/3 chance). Paired switch-to-cue Δ = +0.157 (n=108, p=0.021) |
| A | [A_cue_correct_confound.py](../../scripts/reanalysis/A_cue_correct_confound.py) | Same as Task 2, additionally tagged by `cue_target == correct_letter` | [`A_cue_correct_confound.md`](A_cue_correct_confound.md), [`A_cue_correct_confound.json`](A_cue_correct_confound.json) | Cue-at-WRONG stratum: flip→cue = baseline 76.6% (n=47, p=1.6e-9); W2SR weak 73.6% (n=87, p=2.0e-14); W2SR strong 74.2% (n=97, p=2.1e-16). Paired clean Δ = **+0.172** [+0.016, +0.328], n=64, McNemar p = 0.061 |
| B | [B_per_cue_effects.py](../../scripts/reanalysis/B_per_cue_effects.py) | `r1_7b_baseline` and `r1_7b_w2sr` on `01_stanford_professor`, `03_grader_hack`, `04_unethical_information` | [`B_per_cue_effects.md`](B_per_cue_effects.md), [`B_per_cue_effects.json`](B_per_cue_effects.json) | Per-cue ack Δ: stanford **−0.40** (12/0, p=5e-4); grader_hack −0.20 (7/1, p=0.07); insider **−0.50** (15/0, p=6e-5). Per-cue switch-to-cue Δ: stanford −0.04; grader_hack **+0.33** (p=0.04); insider −0.10. **The +0.157 pooled switch gain comes from grader_hack alone**; ack collapses on all 3 |
| 3 | [03_think_channel_collapse.py](../../scripts/reanalysis/03_think_channel_collapse.py) | Cued + baseline `.eval` for all 6 batches; `/vol/traces/{w2sr,w2sr_r1_14b,w2sr_infamily,w2sr_infamily_strong}/train.json` from Modal (cached at `/tmp/w2sr_traces/`) | [`03_think_channel_collapse.md`](03_think_channel_collapse.md), [`03_think_channel_collapse.json`](03_think_channel_collapse.json) | Training traces carry `</think>` in **100%** of records; trained R1 students emit it on **22–38%** (down from baseline 57.5%); median CoT compression **13.7×** (18,692 → 1,362 chars). Collapse is emergent under SFT, partial, not stripping-driven |
| 4 | [04_matched_length_logit.py](../../scripts/reanalysis/04_matched_length_logit.py) | `r1_7b_baseline` + thickened mix from `r1_7b_w2sr_full` (cues 01/03/04) and `r1_7b_w2sr` (cues 02/05) | [`04_matched_length_logit.md`](04_matched_length_logit.md), [`04_matched_length_logit.json`](04_matched_length_logit.json) | `ack ~ condition + log(CoT_chars) + cue` (cluster SE on qid). W2SR coef: full **−1.29** [−2.15, −0.44] p=0.003; overlap **−1.95** [−3.25, −0.65] p=0.003; long-only ≥9.6k **−1.38** [−2.60, −0.15] p=0.028. **Long-tail W2SR effective n = 39** — direction-stable, underpowered for magnitude |
| C | [C_decode_config_check.py](../../scripts/reanalysis/C_decode_config_check.py) | `configs/{r1_7b_baseline,r1_7b_w2sr,r1_7b_strong,baseline_7b,w2sr_student,control_student}_gpqa.yaml` | [`C_decode_config_check.md`](C_decode_config_check.md), [`C_decode_config_check.json`](C_decode_config_check.json) | All 6 under-test conditions: **T=0, top_p=1.0, top_k=1, single decode.** Manuscript-ready line: *"all evaluation greedy, single decode"* |
| 5A | [05_robustness_rejudge.py](../../scripts/reanalysis/05_robustness_rejudge.py) | All 6 batches' cued cells with stored Claude-Sonnet judge labels | [`05_per_cue_ack_matrix.md`](05_per_cue_ack_matrix.md), [`05_per_cue_ack_matrix.json`](05_per_cue_ack_matrix.json) | Per-cue ack matrix (original judge): all 6 conditions × 5 cues. `stanford_professor`, `insider_information`, `grader_hack` carry all signal; `visual_squares` and `xml_metadata` floor at 0–3% everywhere |
| 5B | [05_robustness_rejudge.py --run-rejudge](../../scripts/reanalysis/05_robustness_rejudge.py) | R1-family cued records' `output.completion` + `metadata.judge_prompt`; OpenRouter | [`05_rejudge_summary.md`](05_rejudge_summary.md), [`05_rejudge_summary.json`](05_rejudge_summary.json), [`05_rejudge_labels.jsonl`](05_rejudge_labels.jsonl) (525 per-sample labels) | Judge swap to `google/gemini-2.5-pro`: overall Cohen's **κ = 0.68**; per-condition κ ∈ {0.645, 0.659, 0.684}. Paired ack drop survives: Δ = **−0.153** [−0.213, −0.100], McNemar p = **2.4e-7**, discordant **23/0** (zero W2SR-only acks under Gemini) |

## Pinned constants (change only deliberately)

[`scripts/reanalysis/_common.py`](../../scripts/reanalysis/_common.py):

- `EXTRACTOR_PATTERNS` — the three regexes used for letter extraction, in
  priority order. Must match [`scripts/patch_meek_eval.py`](../../scripts/patch_meek_eval.py) verbatim.
- `ORIGINAL_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"` — the run-time judge
  on every stored `.eval`.
- `REJUDGE_MODEL = "google/gemini-2.5-pro"` — the alternative judge used by
  Task 5 Part B, pinned so the rejudge `.jsonl` is attributable.

[`scripts/reanalysis/01_gate.py::HEADLINE`](../../scripts/reanalysis/01_gate.py) — the four manuscript headlines
and the p-value upper bounds the assertions enforce.

## Synthesis (what survived the audit)

1. The headline numbers reproduce exactly from disk; the patched extractor is
   doing real work and no trained-condition cell exceeds the 30% null gate.
2. The "behavior toward the cue, silence about it" dissociation holds: flips
   are directionally pulled toward `cue_target` at 2× chance for every R1
   condition (Task 2), and that survives stratifying out cases where the cue
   coincides with the correct answer (Task A) — i.e. the pull is real cue
   influence, not "the model is just right."
3. **Per-cue (Task B):** acknowledgment collapses on all 3 text cues
   (Δ from −0.20 to −0.50; McNemar discordant cells one-directional 12/0,
   7/1, 15/0). The pooled switch-to-cue gain is driven by `grader_hack`
   alone (Δ = +0.33); stanford and insider are flat. So "ack down, switch
   flat or up" is per-cue valid.
4. **Think-channel (Task 3):** training data carried `</think>` in 100% of
   records, trained students dropped to 22–38% — collapse is emergent under
   SFT, not imitation of stripped data, and partial rather than total. The
   dominant mechanism is the 13.7× CoT compression.
5. **Matched length (Task 4):** the W2SR-side ack residual is real and
   direction-stable across full / overlap / long-only fits, but rests on
   effective W2SR n = 39 at the matched-long threshold — direction confident,
   magnitude underpowered.
6. **Robustness (Task 5):** κ = 0.68 between Claude Sonnet and Gemini 2.5
   Pro; the ack drop survives the judge swap (Δ = −0.15, p < 1e-6,
   one-directional). The effect is not a single-judge artifact.
7. **Decode (Task C):** every under-test condition was greedy, single
   decode — no sampling confound.

## Self-corrections made during the audit

- My earlier [INVENTORY.md](../../INVENTORY.md) reported the trained R1
  students at **0% `</think>`**. The actual cued-cell rates are **21.6% and
  37.7%** (Task 3); the original claim was based on a single q0 sample.
- Earlier reads framed the matched-length residual as well-powered. Task 4
  makes the effective W2SR n = 39 explicit; the residual is direction-stable
  but not magnitude-precise.
- The pooled "switch-to-cue went up" finding is driven entirely by
  `grader_hack`; report it per-cue (Task B) rather than as a uniform effect.
