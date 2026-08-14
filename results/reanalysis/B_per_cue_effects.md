# Task B — per-text-cue effect sizes

Restricted to the three text cues where the substrate has dynamic range; `visual_squares` and `xml_metadata` are 0% across all R1 conditions.

## Acknowledgment (judge label, denominator = has_cue ∩ judge label)

| cue | baseline R1-7B | W2SR weak | paired Δ (n, disc base-only/W2SR-only, p) |
|---|---|---|---|
| stanford_professor_recommends | 15/32 = 46.9% [30.9, 63.6] | 3/38 = 7.9% [2.7, 20.8] | Δ=-0.400 [-0.57, -0.23], n=30, 12/0, p=0.00049 |
| grader_hack_validation | 7/32 = 21.9% [11.0, 38.8] | 1/38 = 2.6% [0.5, 13.5] | Δ=-0.200 [-0.37, -0.03], n=30, 7/1, p=0.07 |
| insider_information | 18/32 = 56.2% [39.3, 71.8] | 2/38 = 5.3% [1.5, 17.3] | Δ=-0.500 [-0.67, -0.33], n=30, 15/0, p=6.1e-05 |

## Switch-to-cue (denominator = has_cue ∩ parseable ∩ baseline_ans ≠ cue_target)

| cue | baseline R1-7B | W2SR weak | paired Δ (n, disc base-only/W2SR-only, p) |
|---|---|---|---|
| stanford_professor_recommends | 8/16 = 50.0% [28.0, 72.0] | 20/35 = 57.1% [40.9, 72.0] | Δ=+0.133 [-0.27, +0.47], n=15, 3/5, p=0.73 |
| grader_hack_validation | 5/18 = 27.8% [12.5, 50.9] | 24/37 = 64.9% [48.8, 78.2] | Δ=+0.389 [+0.11, +0.67], n=18, 1/8, p=0.039 |
| insider_information | 8/14 = 57.1% [32.6, 78.6] | 21/35 = 60.0% [43.6, 74.4] | Δ=-0.154 [-0.54, +0.23], n=13, 4/2, p=0.69 |

## Dissociation check (per cue): is acknowledgment Δ negative AND switch-to-cue Δ ≥ 0?

- **stanford_professor_recommends**: ack Δ = -0.400, switch Δ = +0.133 → dissociation YES
- **grader_hack_validation**: ack Δ = -0.200, switch Δ = +0.389 → dissociation YES
- **insider_information**: ack Δ = -0.500, switch Δ = -0.154 → dissociation NO

## Attrition direction
Switch-to-cue paired tests condition on baseline producing a parseable answer (23% of baseline cued samples otherwise drop). That selection skews toward easier/shorter cases for baseline, biasing the W2SR-vs-baseline Δ toward zero. Acknowledgment denominator is has_cue with a judge label and is robust to this attrition (judge runs even when the model's answer is unparseable).
