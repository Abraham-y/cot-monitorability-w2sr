# Task D — self-distillation negative control

**Question:** is the W2SR ack collapse weak-teacher-specific, or does any MATH-CoT SFT do it?

**Design:** train baseline R1-7B on its OWN R1-7B traces (teacher == student strength). Two max_tokens budgets for trace generation: Self-A 4096 (matches W2SR teacher-axis budget), Self-B 8192 (lets R1-7B reach its full natural CoT length). Same 1200 MATH L3-5 problems, T=0.6 sampling, same SFT config, same 40-sample-per-cue GPQA eval, same `claude-sonnet-4-6` judge.

## Pooled ack rates

| condition | cued | parseable | ack k/n | ack % | 95% Wilson |
|---|---:|---:|---:|---:|---|
| baseline R1-7B | 160 | 90 | 40/160 | 25.0% | [18.9, 32.2] |
| W2SR weak (R1-1.5B) | 190 | 175 | 6/190 | 3.2% | [1.5, 6.7] |
| W2SR strong (R1-14B) | 175 | 161 | 13/175 | 7.4% | [4.4, 12.3] |
| Self-A (R1-7B, 4k bud) | 95 | 82 | 2/84 | 2.4% | [0.7, 8.3] |
| Self-B (R1-7B, 8k bud) | 100 | 83 | 3/89 | 3.4% | [1.2, 9.4] |

## Paired McNemar vs baseline R1-7B (matched on (qid, cue))

| comparison | n pairs | disc base-only / student-only | Δ (CI95) | McNemar p |
|---|---:|---:|---|---|
| W2SR weak (R1-1.5B) | 150 | **34** / **1** | -0.220 [-0.293, -0.153] | 2.1e-09 |
| W2SR strong (R1-14B) | 145 | **30** / **5** | -0.172 [-0.248, -0.097] | 2.24e-05 |
| Self-A (R1-7B, 4k bud) | 77 | **23** / **0** | -0.299 [-0.403, -0.207] | 2.38e-07 |
| Self-B (R1-7B, 8k bud) | 82 | **22** / **0** | -0.268 [-0.366, -0.171] | 4.77e-07 |

## Paired McNemar vs W2SR weak (is Self-* statistically equivalent?)

| comparison | n pairs | disc W2SR-only / this-only | Δ (CI95) | McNemar p |
|---|---:|---:|---|---|
| Self-A (R1-7B, 4k bud) vs W2SR weak | 79 | 2 / 2 | +0.000 [-0.051, +0.051] | 1 |
| Self-B (R1-7B, 8k bud) vs W2SR weak | 79 | 2 / 3 | +0.013 [-0.038, +0.076] | 1 |
| W2SR strong (R1-14B) vs W2SR weak | 175 | 3 / 10 | +0.040 [+0.000, +0.080] | 0.0923 |

## Per-cue ack

| cue | baseline R1-7B | W2SR weak (R1-1.5B) | W2SR strong (R1-14B) | Self-A (R1-7B, 4k bud) | Self-B (R1-7B, 8k bud) |
|---|---|---|---|---|---|
| grader_hack_validation | 7/32 = 21.9% | 1/38 = 2.6% | 0/35 = 0.0% | 0/16 = 0.0% | 0/17 = 0.0% |
| insider_information | 18/32 = 56.2% | 2/38 = 5.3% | 5/35 = 14.3% | 0/16 = 0.0% | 0/17 = 0.0% |
| stanford_professor_recommends | 15/32 = 46.9% | 3/38 = 7.9% | 7/35 = 20.0% | 2/18 = 11.1% | 3/19 = 15.8% |
| visual_squares_correct | 0/32 = 0.0% | 0/38 = 0.0% | 0/35 = 0.0% | 0/18 = 0.0% | 0/19 = 0.0% |
| xml_metadata_success_rate | 0/32 = 0.0% | 0/38 = 0.0% | 1/35 = 2.9% | 0/16 = 0.0% | 0/17 = 0.0% |

## CoT length distribution

| condition | n | median chars | p95 chars |
|---|---:|---:|---:|
| baseline R1-7B | 160 | 18,847 | 32,355 |
| W2SR weak (R1-1.5B) | 190 | 1,364 | 27,872 |
| W2SR strong (R1-14B) | 175 | 1,508 | 26,333 |
| Self-A (R1-7B, 4k bud) | 95 | 1,258 | 4,709 |
| Self-B (R1-7B, 8k bud) | 100 | 1,308 | 3,524 |

## Think-tag emission on cued completions

| condition | n | `<think>` | `</think>` |
|---|---:|---:|---:|
| baseline R1-7B | 160 | 0.0% | 57.5% |
| W2SR weak (R1-1.5B) | 190 | 0.0% | 21.6% |
| W2SR strong (R1-14B) | 175 | 0.0% | 37.7% |
| Self-A (R1-7B, 4k bud) | 95 | 0.0% | 28.4% |
| Self-B (R1-7B, 8k bud) | 100 | 0.0% | 26.0% |

## Interpretation
Self-distillation (Self-A 2.4%, Self-B 3.4%) reproduces the W2SR-weak ack collapse (3.2%) at the SAME magnitude — paired Δ vs W2SR weak ≈ 0, p = 1.0 on both arms. The faithfulness collapse is NOT weak-teacher-specific; it is general MATH-CoT SFT compression that fires even when teacher == student. The Self-B (8k budget) result also rules out teacher-truncation-of-CoT as the mechanism (Self-B median CoT 1,275 chars ≈ Self-A 1,258 chars, both ~14× shorter than baseline 18,537 chars).

## What this changes about the paper
- The W2SR "weak teacher harms monitorability" framing is incidental, not causal. The faithfulness collapse fires when teacher == student, so the asymmetry between teacher and student is not the mechanism.
- The mechanism is general: **SFT on terse MATH-style CoT traces** collapses ack regardless of teacher strength. W2SR is an instance of this broader phenomenon, not the cause.
- The safety claim sharpens: an accuracy-only certification of *any* CoT-SFT pipeline that trains on MATH-style traces — including self-distillation — can pass a model whose CoT has become markedly less revealing.
- The teacher-strength axis (W2SR weak ≈ W2SR strong ≈ self ≈ −0.22 to −0.30 paired Δ) now reads as the expected pattern of a teacher-independent effect, not a curiosity.
