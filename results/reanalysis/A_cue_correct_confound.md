# Task A — cue-target-vs-correct confound

Restriction (Task 2 baseline): cued ∩ parseable ∩ has cue_target ∩ has baseline_ans ∩ has correct_letter ∩ baseline_ans ≠ cue_target (room to be pulled). Stratified by whether the cue points at the ground-truth correct answer.

Why this matters: in the cue-at-correct stratum, `answer == cue_target` is indistinguishable from being right. The directional claim only survives cleanly in the cue-at-wrong stratum.

## Per-condition rates by stratum

### R1-distill family

| condition | stratum | n_restr | switch-to-cue | #flippers | flip→cue (95% CI) | p vs 1/3 |
|---|---|---:|---|---:|---|---|
| baseline R1-7B | pooled | 123 | 44/123 = 35.8% [27.9, 44.6] | 68 | 44/68 = 64.7% [52.8, 75.0] | 1.23e-07 |
| baseline R1-7B | cue_at_wrong | 95 | 36/95 = 37.9% [28.8, 47.9] | 47 | 36/47 = 76.6% [62.8, 86.4] | 1.57e-09 |
| baseline R1-7B | cue_at_correct | 28 | 8/28 = 28.6% [15.3, 47.1] | 21 | 8/21 = 38.1% [20.8, 59.1] | 0.399 |
| W2SR weak (R1-1.5B teacher) | pooled | 179 | 88/179 = 49.2% [41.9, 56.4] | 121 | 88/121 = 72.7% [64.2, 79.9] | 9.81e-19 |
| W2SR weak (R1-1.5B teacher) | cue_at_wrong | 127 | 64/127 = 50.4% [41.8, 58.9] | 87 | 64/87 = 73.6% [63.4, 81.7] | 2.02e-14 |
| W2SR weak (R1-1.5B teacher) | cue_at_correct | 52 | 24/52 = 46.2% [33.3, 59.5] | 34 | 24/34 = 70.6% [53.8, 83.2] | 9.99e-06 |
| W2SR strong (R1-14B teacher) | pooled | 165 | 88/165 = 53.3% [45.7, 60.8] | 119 | 88/119 = 73.9% [65.4, 81.0] | 1.58e-19 |
| W2SR strong (R1-14B teacher) | cue_at_wrong | 126 | 72/126 = 57.1% [48.4, 65.4] | 97 | 72/97 = 74.2% [64.7, 81.9] | 2.14e-16 |
| W2SR strong (R1-14B teacher) | cue_at_correct | 39 | 16/39 = 41.0% [27.1, 56.6] | 22 | 16/22 = 72.7% [51.8, 86.8] | 0.000183 |

### Instruct family

| condition | stratum | n_restr | switch-to-cue | #flippers | flip→cue (95% CI) | p vs 1/3 |
|---|---|---:|---|---:|---|---|
| instruct baseline (Qwen2.5-7B-Inst) | pooled | 965 | 489/965 = 50.7% [47.5, 53.8] | 636 | 489/636 = 76.9% [73.5, 80.0] | 6e-112 |
| instruct baseline (Qwen2.5-7B-Inst) | cue_at_wrong | 750 | 413/750 = 55.1% [51.5, 58.6] | 520 | 413/520 = 79.4% [75.7, 82.7] | 3.92e-103 |
| instruct baseline (Qwen2.5-7B-Inst) | cue_at_correct | 215 | 76/215 = 35.3% [29.3, 41.9] | 116 | 76/116 = 65.5% [56.5, 73.5] | 1.47e-12 |
| instruct W2SR weak | pooled | 216 | 104/216 = 48.1% [41.6, 54.8] | 152 | 104/152 = 68.4% [60.7, 75.3] | 1.12e-18 |
| instruct W2SR weak | cue_at_wrong | 153 | 75/153 = 49.0% [41.2, 56.9] | 108 | 75/108 = 69.4% [60.2, 77.3] | 1.99e-14 |
| instruct W2SR weak | cue_at_correct | 63 | 29/63 = 46.0% [34.3, 58.2] | 44 | 29/44 = 65.9% [51.1, 78.1] | 1.01e-05 |
| instruct W2SR strong (control) | pooled | 167 | 82/167 = 49.1% [41.6, 56.6] | 126 | 82/126 = 65.1% [56.4, 72.8] | 3.42e-13 |
| instruct W2SR strong (control) | cue_at_wrong | 114 | 54/114 = 47.4% [38.4, 56.5] | 79 | 54/79 = 68.4% [57.5, 77.6] | 2.19e-10 |
| instruct W2SR strong (control) | cue_at_correct | 53 | 28/53 = 52.8% [39.7, 65.6] | 47 | 28/47 = 59.6% [45.3, 72.4] | 0.000201 |

## Paired switch-to-cue on the cue-at-wrong stratum (baseline R1-7B vs W2SR weak)

Matched pairs: 64; 2×2: (0,0)=19, baseline-only=9, W2SR-only=20, both=16.
McNemar exact p = 0.0614; Δ = +0.172 [+0.016, +0.328].

## Attrition direction
Baseline R1-7B loses 23% of cued samples to no-answer (parseable filter); trained students lose ≤6%. The paired test conditions on baseline producing an answer, which selects easier/shorter cases for baseline — biases the paired Δ toward zero (against W2SR). So Δ is a conservative lower bound.
