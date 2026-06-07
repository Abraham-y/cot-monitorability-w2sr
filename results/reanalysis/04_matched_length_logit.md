# Task 4 — matched-length acknowledgment residual

## Data construction

Baseline = `r1_7b_baseline` (DeepSeek-R1-Distill-Qwen-7B, 40 samples × 5 cues). W2SR weak = thickened on cues 01/03/04 from `r1_7b_w2sr_full` (198 samples each) and 40 samples each on cues 02/05 from `r1_7b_w2sr`, so all 5 cues are present while the long-CoT tail mass for the three text cues uses the bigger n.

Records with judge label: baseline n = 160, W2SR weak n = 634.

## Length-distribution honesty check

| condition | n | p05 | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline R1-7B | 160 | 3406 | 8534 | 18692 | 26642 | 32357 | 35710 |
| W2SR weak (mix) | 634 | 809 | 1118 | 1464 | 1953 | 21712 | 37815 |

- W2SR weak records ≥ 9,600 chars: **39 / 634 (6.2%)** — the matched-long residual rests on this tail.
- Baseline records ≥ 9,600 chars: 110 / 160 (68.8%).
- Overlap region [max(p05), min(p95)] = [3,406, 21,712] chars: baseline 89, W2SR 55.

## Logistic-regression coefficient on condition (W2SR weak vs baseline)

Model: `ack ~ condition + log(CoT_chars) + cue`, cluster-robust SE on qid. Coefficient is on the **logit scale**: negative means W2SR is less likely to acknowledge at matched length.

| fit | n | W2SR vs baseline coefficient | interpretation |
|---|---:|---|---|
| (i) full data, main effects | 794 | -1.294 [-2.152, -0.437] (p=0.0031) | residual after log-length control |
| (i') full data, with condition × log(length) interaction | 794 | main: -4.156 [-10.171, +1.860] (p=0.176); interaction: +0.305 [-0.345, +0.954] (p=0.358) | does the W2SR drop depend on length |
| (ii) trimmed overlap [3,406, 21,712] | 144 | -1.952 [-3.251, -0.653] (p=0.00323) | residual on common length support |
| (iii) long-only (≥9,600) | 149 | -1.376 [-2.604, -0.149] (p=0.028) | residual on the long tail only |

Raw ack rate on the long subset: baseline = 0.291 (n=110); W2SR weak = 0.154 (n=39).

## Read
After controlling for log(CoT length) and cue, the W2SR-vs-baseline ack drop shrinks but does not vanish (full-data residual coefficient -1.29 on the logit scale, p=0.0031). On the matched-long tail (≥9.6k chars) W2SR contributes only 39 samples vs baseline's 110; the residual claim is real-direction but underpowered, and any inference about length-matched behavior is dominated by that small W2SR tail.