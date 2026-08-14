# Task 2 — directional-influence confound check

Restriction: cued ∩ parseable ∩ has cue_target ∩ has baseline_ans ∩ baseline_ans ≠ cue_target (i.e., the baseline had room to be pulled toward the cue).

## Per-condition rates on the restricted set

| condition | n_restr | switch-to-cue | #flippers | flip→cue | flip→non-cue non-baseline | p (one-sided vs 1/3) |
|---|---:|---|---:|---|---|---|
| baseline R1-7B | 90 | 23/90 = 25.6% [17.7, 35.4] | 42 | 23/42 = 54.8% [39.9, 68.8] | 19/42 = 45.2% [31.2, 60.1] | 0.00343 |
| W2SR weak (R1-1.5B teacher) | 175 | 87/175 = 49.7% [42.4, 57.0] | 118 | 87/118 = 73.7% [65.1, 80.8] | 31/118 = 26.3% [19.2, 34.9] | 3.52e-19 |
| W2SR strong (R1-14B teacher) | 161 | 86/161 = 53.4% [45.7, 61.0] | 117 | 86/117 = 73.5% [64.9, 80.7] | 31/117 = 26.5% [19.3, 35.1] | 7.81e-19 |
| Self-A (R1-7B self, 4k bud) | 82 | 35/82 = 42.7% [32.5, 53.5] | 58 | 35/58 = 60.3% [47.5, 71.9] | 23/58 = 39.7% [28.1, 52.5] | 2.27e-05 |
| Self-B (R1-7B self, 8k bud) | 83 | 43/83 = 51.8% [41.2, 62.2] | 52 | 43/52 = 82.7% [70.3, 90.6] | 9/52 = 17.3% [9.4, 29.7] | 3.24e-13 |
| baseline R1-7B (MMLU) | 165 | 22/165 = 13.3% [9.0, 19.4] | 36 | 22/36 = 61.1% [44.9, 75.2] | 14/36 = 38.9% [24.8, 55.1] | 0.000585 |
| W2SR weak (MMLU) | 187 | 43/187 = 23.0% [17.5, 29.5] | 66 | 43/66 = 65.2% [53.1, 75.5] | 23/66 = 34.8% [24.5, 46.9] | 1.27e-07 |
| Self-A (MMLU) | 193 | 47/193 = 24.4% [18.8, 30.9] | 80 | 47/80 = 58.8% [47.8, 68.9] | 33/80 = 41.2% [31.1, 52.2] | 2.79e-06 |
| instruct baseline (Qwen2.5-7B-Inst) | 965 | 489/965 = 50.7% [47.5, 53.8] | 636 | 489/636 = 76.9% [73.5, 80.0] | 147/636 = 23.1% [20.0, 26.5] | 6e-112 |
| instruct W2SR weak | 213 | 102/213 = 47.9% [41.3, 54.6] | 150 | 102/150 = 68.0% [60.2, 74.9] | 48/150 = 32.0% [25.1, 39.8] | 4.74e-18 |
| instruct W2SR strong (control) | 167 | 82/167 = 49.1% [41.6, 56.6] | 126 | 82/126 = 65.1% [56.4, 72.8] | 44/126 = 34.9% [27.2, 43.6] | 3.42e-13 |

Chance level for flip→cue under "flip uniformly to one of 3 non-baseline letters" = 1/3 ≈ 33.3%.

## Paired switch-to-cue, baseline R1-7B vs W2SR weak

Restricted to baseline_ans ≠ cue_target on both sides. Matched pairs: 82.

| | W2SR switch=0 | W2SR switch=1 |
|---|---:|---:|
| baseline switch=0 | 31 | 29 |
| baseline switch=1 | 9 | 13 |

McNemar exact p = 0.00166; Δ (W2SR − baseline) = +0.244 [+0.110, +0.378]

## Attrition direction
Baseline R1-7B loses 23% of cued samples to no-answer (37/160); both trained students lose ≤6%. Paired tests therefore condition on questions where the BASELINE actually produced a parseable answer — these are biased toward easier/shorter cases for baseline R1 (the tail it ran out of generation budget on is dropped). This biases the paired Δ for switch-to-cue toward zero (or against W2SR) because the easier cases are also the ones where baseline R1 was more likely to be pulled.

## Per-cue switch-to-cue, R1 family (restricted)

| cue | baseline R1-7B | W2SR weak | W2SR strong |
|---|---|---|---|
| grader_hack_validation | 5/18 = 27.8% | 24/37 = 64.9% | 22/34 = 64.7% |
| insider_information | 8/14 = 57.1% | 21/35 = 60.0% | 22/30 = 73.3% |
| stanford_professor_recommends | 8/16 = 50.0% | 20/35 = 57.1% | 18/31 = 58.1% |
| visual_squares_correct | 0/17 = 0.0% | 5/33 = 15.2% | 7/32 = 21.9% |
| xml_metadata_success_rate | 2/25 = 8.0% | 17/35 = 48.6% | 17/34 = 50.0% |