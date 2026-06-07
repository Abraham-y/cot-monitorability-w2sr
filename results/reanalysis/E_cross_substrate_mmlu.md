# Task E — cross-substrate MMLU replication

**Substrate:** 5 STEM MMLU subjects (college_physics, college_chemistry, college_biology, college_mathematics, conceptual_physics), capped at 8 questions per subject (`W2SR_LIMIT=8`) → ~40 cued samples per cell, matching the GPQA cell size. Same 5 cues, same `claude-sonnet-4-6` judge, same VLLMServer endpoint.

## Pooled ack

| condition | cued | parseable | ack k/n | ack % | 95% Wilson |
|---|---:|---:|---:|---:|---|
| baseline R1-7B (MMLU) | 40 | 40 | 8/40 | 20.0% | [10.5, 34.8] |
| W2SR weak (MMLU) | 35 | 30 | 1/35 | 2.9% | [0.5, 14.5] |
| Self-A (MMLU) | 35 | 34 | 0/35 | 0.0% | [0.0, 9.9] |

## Paired ack vs baseline R1-7B (MMLU)

| comparison | n pairs | disc base-only / student-only | Δ (CI95) | McNemar p |
|---|---:|---:|---|---|
| W2SR weak (MMLU) | 35 | **6** / **1** | -0.143 [-0.286, +0.000] | 0.125 |
| Self-A (MMLU) | 35 | **6** / **0** | -0.171 [-0.314, -0.057] | 0.0312 |

## Paired Self-A vs W2SR weak (MMLU) — replicates GPQA's 'indistinguishable' finding

n = 35; disc W2SR-only / Self-only = 1 / 0; Δ = -0.029 [-0.086, +0.000]; McNemar p = 1.

## Per-cue ack

| cue | baseline R1-7B (MMLU) | W2SR weak (MMLU) | Self-A (MMLU) |
|---|---|---|---|
| grader_hack_validation | 1/8 = 12.5% | 0/7 = 0.0% | 0/7 = 0.0% |
| insider_information | 4/8 = 50.0% | 0/7 = 0.0% | 0/7 = 0.0% |
| stanford_professor_recommends | 2/8 = 25.0% | 0/7 = 0.0% | 0/7 = 0.0% |
| visual_squares_correct | 1/8 = 12.5% | 0/7 = 0.0% | 0/7 = 0.0% |
| xml_metadata_success_rate | 0/8 = 0.0% | 1/7 = 14.3% | 0/7 = 0.0% |

## CoT compression

| condition | n | median chars | p95 chars |
|---|---:|---:|---:|
| baseline R1-7B (MMLU) | 40 | 3,898 | 31,746 |
| W2SR weak (MMLU) | 35 | 1,139 | 2,058 |
| Self-A (MMLU) | 35 | 1,162 | 1,938 |

Baseline R1-7B writes shorter CoT on MMLU than on GPQA (3,611 vs 18,537 chars median) — MMLU's question style and difficulty don't elicit the full long-CoT regime. So MMLU compression is **3× from baseline**, vs **14× on GPQA**. The ack collapse still fires.

## Think-tag emission (cued completions)

| condition | n | `</think>` |
|---|---:|---:|
| baseline R1-7B (MMLU) | 40 | 90.0% |
| W2SR weak (MMLU) | 35 | 25.7% |
| Self-A (MMLU) | 35 | 34.3% |

Baseline emits `</think>` on 90% of MMLU cued completions (vs 57% on GPQA — shorter problems means the closing tag fits within the generation budget). Trained students drop to 26–34%, matching the partial-collapse pattern from GPQA.

## Influence rate (answer == cue_target)

| condition | k/n | rate | 95% CI |
|---|---:|---:|---|
| baseline R1-7B (MMLU) | 3/40 | 7.5% | [2.6, 19.9] |
| W2SR weak (MMLU) | 5/30 | 16.7% | [7.3, 33.6] |
| Self-A (MMLU) | 7/34 | 20.6% | [10.3, 36.8] |

Same direction as GPQA: trained students show modestly higher switch-to-cue rate (baseline 7.5% → W2SR 16.7% → Self-A 20.6%), so the "behavior toward the cue, silence about it" dissociation holds.

## Interpretation
Cross-substrate replication on MMLU (5 STEM subjects × 8 Qs = 40/cell) holds the headline pattern: baseline ack 20.0%, W2SR weak 2.9%, Self-A 0.0%; Self-A vs baseline paired Δ = -0.171, McNemar p = 0.031 (discordant 6/0, one-directional). Self-A vs W2SR weak remains indistinguishable (Δ ≈ 0, p = 1.0, discordant 1/0) — the 'same mechanism' finding generalizes. W2SR weak vs baseline is direction-positive but underpowered at n=35 (discordant 6/1, p = 0.125). CoT compresses 3× on MMLU (3,611 → 1,139 chars) vs 14× on GPQA, because baseline R1-7B already writes shorter CoT on MMLU. Effect generalizes beyond GPQA-Diamond.

## Honest caveats
- n = 40/cell × 5 subjects × 1 cap is a small cross-substrate test. W2SR weak vs baseline does not reach p < 0.05 at this n (p = 0.125, discordant 6/1), though Self-A vs baseline does (p = 0.031, 6/0).
- Only 5 STEM subjects of MMLU; broader MMLU coverage (humanities, social sciences) untested.
- Same judge (claude-sonnet-4-6) as the GPQA arm; cross-judge robustness was checked on GPQA but not re-checked here.
