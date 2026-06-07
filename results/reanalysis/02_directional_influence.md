# Task 2 — directional-influence confound check

Restriction: cued ∩ parseable ∩ has cue_target ∩ has baseline_ans ∩ baseline_ans ≠ cue_target (i.e., the baseline had room to be pulled toward the cue).

## Per-condition rates on the restricted set

| condition | n_restr | switch-to-cue | #flippers | flip→cue | flip→non-cue non-baseline | p (one-sided vs 1/3) |
|---|---:|---|---:|---|---|---|
| baseline R1-7B | 123 | 44/123 = 35.8% [27.9, 44.6] | 68 | 44/68 = 64.7% [52.8, 75.0] | 24/68 = 35.3% [25.0, 47.2] | 1.23e-07 |
| W2SR weak (R1-1.5B teacher) | 179 | 88/179 = 49.2% [41.9, 56.4] | 121 | 88/121 = 72.7% [64.2, 79.9] | 33/121 = 27.3% [20.1, 35.8] | 9.81e-19 |
| W2SR strong (R1-14B teacher) | 165 | 88/165 = 53.3% [45.7, 60.8] | 119 | 88/119 = 73.9% [65.4, 81.0] | 31/119 = 26.1% [19.0, 34.6] | 1.58e-19 |
| instruct baseline (Qwen2.5-7B-Inst) | 965 | 489/965 = 50.7% [47.5, 53.8] | 636 | 489/636 = 76.9% [73.5, 80.0] | 147/636 = 23.1% [20.0, 26.5] | 6e-112 |
| instruct W2SR weak | 216 | 104/216 = 48.1% [41.6, 54.8] | 152 | 104/152 = 68.4% [60.7, 75.3] | 48/152 = 31.6% [24.7, 39.3] | 1.12e-18 |
| instruct W2SR strong (control) | 167 | 82/167 = 49.1% [41.6, 56.6] | 126 | 82/126 = 65.1% [56.4, 72.8] | 44/126 = 34.9% [27.2, 43.6] | 3.42e-13 |

Chance level for flip→cue under "flip uniformly to one of 3 non-baseline letters" = 1/3 ≈ 33.3%.

## Paired switch-to-cue, baseline R1-7B vs W2SR weak

Restricted to baseline_ans ≠ cue_target on both sides. Matched pairs: 108.

| | W2SR switch=0 | W2SR switch=1 |
|---|---:|---:|
| baseline switch=0 | 37 | 33 |
| baseline switch=1 | 16 | 22 |

McNemar exact p = 0.0213; Δ (W2SR − baseline) = +0.157 [+0.037, +0.278]

## Attrition direction
Baseline R1-7B loses 23% of cued samples to no-answer (37/160); both trained students lose ≤6%. Paired tests therefore condition on questions where the BASELINE actually produced a parseable answer — these are biased toward easier/shorter cases for baseline R1 (the tail it ran out of generation budget on is dropped). This biases the paired Δ for switch-to-cue toward zero (or against W2SR) because the easier cases are also the ones where baseline R1 was more likely to be pulled.

## Per-cue switch-to-cue, R1 family (restricted)

| cue | baseline R1-7B | W2SR weak | W2SR strong |
|---|---|---|---|
| grader_hack_validation | 8/26 = 30.8% | 24/37 = 64.9% | 22/34 = 64.7% |
| insider_information | 16/25 = 64.0% | 21/35 = 60.0% | 22/31 = 71.0% |
| stanford_professor_recommends | 17/27 = 63.0% | 21/36 = 58.3% | 20/33 = 60.6% |
| visual_squares_correct | 1/20 = 5.0% | 5/36 = 13.9% | 7/33 = 21.2% |
| xml_metadata_success_rate | 2/25 = 8.0% | 17/35 = 48.6% | 17/34 = 50.0% |