# Task A — cue-target-vs-correct confound

Restriction (Task 2 baseline): cued ∩ parseable ∩ has cue_target ∩ has baseline_ans ∩ has correct_letter ∩ baseline_ans ≠ cue_target (room to be pulled). Stratified by whether the cue points at the ground-truth correct answer.

Why this matters: in the cue-at-correct stratum, `answer == cue_target` is indistinguishable from being right. The directional claim only survives cleanly in the cue-at-wrong stratum.

## Per-condition rates by stratum

### R1-distill family

| condition | stratum | n_restr | switch-to-cue | #flippers | flip→cue (95% CI) | p vs 1/3 |
|---|---|---:|---|---:|---|---|
| baseline R1-7B | pooled | 90 | 23/90 = 25.6% [17.7, 35.4] | 42 | 23/42 = 54.8% [39.9, 68.8] | 0.00343 |
| baseline R1-7B | cue_at_wrong | 66 | 16/66 = 24.2% [15.5, 35.8] | 24 | 16/24 = 66.7% [46.7, 82.0] | 0.000859 |
| baseline R1-7B | cue_at_correct | 24 | 7/24 = 29.2% [14.9, 49.2] | 18 | 7/18 = 38.9% [20.3, 61.4] | 0.391 |
| W2SR weak (R1-1.5B teacher) | pooled | 175 | 87/175 = 49.7% [42.4, 57.0] | 118 | 87/118 = 73.7% [65.1, 80.8] | 3.52e-19 |
| W2SR weak (R1-1.5B teacher) | cue_at_wrong | 124 | 63/124 = 50.8% [42.1, 59.4] | 85 | 63/85 = 74.1% [63.9, 82.2] | 1.78e-14 |
| W2SR weak (R1-1.5B teacher) | cue_at_correct | 51 | 24/51 = 47.1% [34.1, 60.5] | 33 | 24/33 = 72.7% [55.8, 84.9] | 4.3e-06 |
| W2SR strong (R1-14B teacher) | pooled | 161 | 86/161 = 53.4% [45.7, 61.0] | 117 | 86/117 = 73.5% [64.9, 80.7] | 7.81e-19 |
| W2SR strong (R1-14B teacher) | cue_at_wrong | 123 | 71/123 = 57.7% [48.9, 66.1] | 96 | 71/96 = 74.0% [64.4, 81.7] | 4.79e-16 |
| W2SR strong (R1-14B teacher) | cue_at_correct | 38 | 15/38 = 39.5% [25.6, 55.3] | 21 | 15/21 = 71.4% [50.0, 86.2] | 0.000405 |

### Instruct family

| condition | stratum | n_restr | switch-to-cue | #flippers | flip→cue (95% CI) | p vs 1/3 |
|---|---|---:|---|---:|---|---|
| instruct baseline (Qwen2.5-7B-Inst) | pooled | 965 | 489/965 = 50.7% [47.5, 53.8] | 636 | 489/636 = 76.9% [73.5, 80.0] | 6e-112 |
| instruct baseline (Qwen2.5-7B-Inst) | cue_at_wrong | 750 | 413/750 = 55.1% [51.5, 58.6] | 520 | 413/520 = 79.4% [75.7, 82.7] | 3.92e-103 |
| instruct baseline (Qwen2.5-7B-Inst) | cue_at_correct | 215 | 76/215 = 35.3% [29.3, 41.9] | 116 | 76/116 = 65.5% [56.5, 73.5] | 1.47e-12 |
| instruct W2SR weak | pooled | 213 | 102/213 = 47.9% [41.3, 54.6] | 150 | 102/150 = 68.0% [60.2, 74.9] | 4.74e-18 |
| instruct W2SR weak | cue_at_wrong | 153 | 75/153 = 49.0% [41.2, 56.9] | 108 | 75/108 = 69.4% [60.2, 77.3] | 1.99e-14 |
| instruct W2SR weak | cue_at_correct | 60 | 27/60 = 45.0% [33.1, 57.5] | 42 | 27/42 = 64.3% [49.2, 77.0] | 3.99e-05 |
| instruct W2SR strong (control) | pooled | 167 | 82/167 = 49.1% [41.6, 56.6] | 126 | 82/126 = 65.1% [56.4, 72.8] | 3.42e-13 |
| instruct W2SR strong (control) | cue_at_wrong | 114 | 54/114 = 47.4% [38.4, 56.5] | 79 | 54/79 = 68.4% [57.5, 77.6] | 2.19e-10 |
| instruct W2SR strong (control) | cue_at_correct | 53 | 28/53 = 52.8% [39.7, 65.6] | 47 | 28/47 = 59.6% [45.3, 72.4] | 0.000201 |

## Paired switch-to-cue on the cue-at-wrong stratum (baseline R1-7B vs W2SR weak)

Matched pairs: 45; 2×2: (0,0)=16, baseline-only=2, W2SR-only=17, both=10.
McNemar exact p = 0.000729; Δ = +0.333 [+0.156, +0.489].

## Attrition direction
Baseline R1-7B loses 23% of cued samples to no-answer (parseable filter); trained students lose ≤6%. The paired test conditions on baseline producing an answer, which selects easier/shorter cases for baseline — biases the paired Δ toward zero (against W2SR). So Δ is a conservative lower bound.
