# Task E — cross-substrate MMLU replication

**Substrate:** 5 STEM MMLU subjects (college_physics, college_chemistry, college_biology, college_mathematics, conceptual_physics), capped at 8 questions per subject (`W2SR_LIMIT=8`) → ~40 cued samples per cell, matching the GPQA cell size. Same 5 cues, same `claude-sonnet-4-6` judge, same VLLMServer endpoint.

## Pooled ack

| condition | cued | parseable | ack k/n | ack % | 95% Wilson |
|---|---:|---:|---:|---:|---|
| baseline R1-7B (MMLU) | 190 | 165 | 52/190 | 27.4% | [21.5, 34.1] |
| W2SR weak (MMLU) | 195 | 187 | 7/195 | 3.6% | [1.7, 7.2] |
| Self-A (MMLU) | 195 | 193 | 9/195 | 4.6% | [2.4, 8.5] |

## Paired ack vs baseline R1-7B (MMLU)

| comparison | n pairs | disc base-only / student-only | Δ (CI95) | McNemar p |
|---|---:|---:|---|---|
| W2SR weak (MMLU) | 185 | **47** / **4** | -0.232 [-0.303, -0.168] | 2.42e-10 |
| Self-A (MMLU) | 185 | **45** / **4** | -0.222 [-0.292, -0.157] | 8.23e-10 |

## Paired Self-A vs W2SR weak (MMLU) — replicates GPQA's 'indistinguishable' finding

n = 195; disc W2SR-only / Self-only = 3 / 5; Δ = +0.010 [-0.015, +0.041]; McNemar p = 0.727.

## Per-cue ack

| cue | baseline R1-7B (MMLU) | W2SR weak (MMLU) | Self-A (MMLU) |
|---|---|---|---|
| grader_hack_validation | 9/38 = 23.7% | 2/39 = 5.1% | 1/39 = 2.6% |
| insider_information | 23/38 = 60.5% | 1/39 = 2.6% | 3/39 = 7.7% |
| stanford_professor_recommends | 18/38 = 47.4% | 2/39 = 5.1% | 3/39 = 7.7% |
| visual_squares_correct | 1/38 = 2.6% | 0/39 = 0.0% | 1/39 = 2.6% |
| xml_metadata_success_rate | 1/38 = 2.6% | 2/39 = 5.1% | 1/39 = 2.6% |

## CoT compression

| condition | n | median chars | p95 chars |
|---|---:|---:|---:|
| baseline R1-7B (MMLU) | 190 | 3,226 | 30,713 |
| W2SR weak (MMLU) | 195 | 1,141 | 2,510 |
| Self-A (MMLU) | 195 | 1,192 | 2,190 |

Baseline R1-7B writes shorter CoT on MMLU than on GPQA (3,226 vs 18,692 chars median (GPQA cued)) — MMLU's question style and difficulty don't elicit the full long-CoT regime. So MMLU compression is **2.8× from baseline**, vs **~14× on GPQA**. The ack collapse still fires.

## Think-tag emission (cued completions)

| condition | n | `</think>` |
|---|---:|---:|
| baseline R1-7B (MMLU) | 190 | 86.8% |
| W2SR weak (MMLU) | 195 | 43.6% |
| Self-A (MMLU) | 195 | 48.2% |

Baseline emits `</think>` on 87% of MMLU cued completions (vs 57% on GPQA — shorter problems means the closing tag fits within the generation budget). Trained students drop to 44–48%, matching the partial-collapse pattern from GPQA.

## Influence rate (answer == cue_target)

| condition | k/n | rate | 95% CI |
|---|---:|---:|---|
| baseline R1-7B (MMLU) | 22/165 | 13.3% | [9.0, 19.4] |
| W2SR weak (MMLU) | 43/187 | 23.0% | [17.5, 29.5] |
| Self-A (MMLU) | 47/193 | 24.4% | [18.8, 30.9] |

Same direction as GPQA: trained students show modestly higher switch-to-cue rate (baseline 13.3% → W2SR 23.0% → Self-A 24.4%), so the "behavior toward the cue, silence about it" dissociation holds.

## Interpretation
Cross-substrate replication on MMLU (5 STEM subjects × 8 Qs per cue) holds the headline pattern: baseline ack 27.4%, W2SR weak 3.6%, Self-A 4.6%. W2SR weak vs baseline paired Δ = -0.232, McNemar p = 2.42e-10 (n = 185, discordant 47/4); Self-A vs baseline Δ = -0.222, p = 8.23e-10 (discordant 45/4). Self-A vs W2SR weak remains indistinguishable (Δ = +0.010, p = 0.727) — the 'same mechanism' finding generalizes. CoT compresses 2.8× on MMLU (3,226 → 1,141 chars) vs ~14× on GPQA, because baseline R1-7B already writes shorter CoT on MMLU. Effect generalizes beyond GPQA-Diamond.

## Honest caveats
- Per-subject cells are small (~30–40 cued samples); the pooled paired comparisons carry the power (W2SR weak vs baseline p = 2.42e-10, discordant 47/4; Self-A vs baseline p = 8.23e-10, discordant 45/4).
- Only 5 STEM subjects of MMLU; broader MMLU coverage (humanities, social sciences) untested.
- Same judge (claude-sonnet-4-6) as the GPQA arm; cross-judge robustness was checked on GPQA but not re-checked here.
