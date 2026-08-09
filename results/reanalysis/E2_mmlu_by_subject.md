# Task E2 — MMLU replication by subject

Subjects: college_biology, college_chemistry, college_mathematics, college_physics, conceptual_physics

## Pooled (all subjects)

| condition | ack | rate | 95% CI |
|---|---:|---:|---|
| baseline R1-7B (MMLU) | 52/190 | 27.4% | [21.5, 34.1] |
| W2SR weak (MMLU) | 7/195 | 3.6% | [1.7, 7.2] |
| Self-A (MMLU) | 9/195 | 4.6% | [2.4, 8.5] |

## By subject

| subject | baseline R1-7B (MMLU) | W2SR weak (MMLU) | Self-A (MMLU) |
|---|---:|---:|---:|
| college_biology | 9/40 = 22.5% | 2/40 = 5.0% | 5/40 = 12.5% |
| college_chemistry | 8/40 = 20.0% | 1/40 = 2.5% | 1/40 = 2.5% |
| college_mathematics | 11/30 = 36.7% | 1/40 = 2.5% | 0/40 = 0.0% |
| college_physics | 8/40 = 20.0% | 1/35 = 2.9% | 0/35 = 0.0% |
| conceptual_physics | 16/40 = 40.0% | 2/40 = 5.0% | 3/40 = 7.5% |

## In-domain (college_mathematics) vs combined non-math STEM

| condition | college_mathematics | non-math STEM |
|---|---:|---:|
| baseline R1-7B (MMLU) | 11/30 = 36.7% | 41/160 = 25.6% |
| W2SR weak (MMLU) | 1/40 = 2.5% | 6/155 = 3.9% |
| Self-A (MMLU) | 0/40 = 0.0% | 9/155 = 5.8% |

## Paired McNemar vs baseline R1-7B (MMLU)

| condition | scope | n | disc (base/student) | Δ | p |
|---|---|---:|---:|---:|---:|
| W2SR weak (MMLU) | pooled | 185 | 47/4 | -0.232 | 2.42e-10 |
| W2SR weak (MMLU) | college_mathematics | 30 | 11/1 | -0.333 | 0.00635 |
| W2SR weak (MMLU) | non_math | 155 | 36/3 | -0.213 | 3.61e-08 |
| Self-A (MMLU) | pooled | 185 | 45/4 | -0.222 | 8.23e-10 |
| Self-A (MMLU) | college_mathematics | 30 | 11/0 | -0.367 | 0.000977 |
| Self-A (MMLU) | non_math | 155 | 34/4 | -0.194 | 6.04e-07 |

## Read

The in-domain `college_mathematics` cut (matched to the MATH-L3–L5 SFT data) shows the same collapse as the science subjects, so the dissociation does not require a train->eval distribution shift: baseline acknowledges 11/30 = 36.7% on math questions and the trained arms collapse to near zero.
