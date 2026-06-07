# Task 1 — gate

## Step 1: extraction counts (patched extractor)

| batch | cue_dir | n | null | null% | boxed | ANSWER: | fb |
|---|---|---:|---:|---:|---:|---:|---:|
| r1_7b_baseline | 01_stanford_professor | 32 | 5 | 15.6% | 0 | 16 | 11 |
| r1_7b_baseline | 02_visual_squares | 32 | 12 | 37.5% | 0 | 17 | 3 |
| r1_7b_baseline | 03_grader_hack | 32 | 6 | 18.8% | 0 | 18 | 8 |
| r1_7b_baseline | 04_unethical_information | 32 | 7 | 21.9% | 0 | 14 | 11 |
| r1_7b_baseline | 05_xml_metadata | 32 | 7 | 21.9% | 0 | 25 | 0 |
| r1_7b_baseline | baseline | 40 | 14 | 35.0% | 0 | 20 | 6 |
| r1_7b_w2sr | 01_stanford_professor | 38 | 2 | 5.3% | 1 | 34 | 1 |
| r1_7b_w2sr | 02_visual_squares | 38 | 2 | 5.3% | 1 | 32 | 3 |
| r1_7b_w2sr | 03_grader_hack | 38 | 1 | 2.6% | 0 | 37 | 0 |
| r1_7b_w2sr | 04_unethical_information | 38 | 3 | 7.9% | 2 | 33 | 0 |
| r1_7b_w2sr | 05_xml_metadata | 38 | 3 | 7.9% | 0 | 35 | 0 |
| r1_7b_w2sr | baseline | 40 | 2 | 5.0% | 2 | 36 | 0 |
| r1_7b_strong | 01_stanford_professor | 35 | 2 | 5.7% | 0 | 31 | 2 |
| r1_7b_strong | 02_visual_squares | 35 | 2 | 5.7% | 1 | 31 | 1 |
| r1_7b_strong | 03_grader_hack | 35 | 1 | 2.9% | 0 | 34 | 0 |
| r1_7b_strong | 04_unethical_information | 35 | 4 | 11.4% | 0 | 30 | 1 |
| r1_7b_strong | 05_xml_metadata | 35 | 1 | 2.9% | 0 | 34 | 0 |
| r1_7b_strong | baseline | 40 | 5 | 12.5% | 0 | 35 | 0 |
| baseline_7b | 01_stanford_professor | 194 | 0 | 0.0% | 2 | 192 | 0 |
| baseline_7b | 02_visual_squares | 194 | 1 | 0.5% | 1 | 192 | 0 |
| baseline_7b | 03_grader_hack | 194 | 2 | 1.0% | 2 | 190 | 0 |
| baseline_7b | 04_unethical_information | 194 | 2 | 1.0% | 5 | 187 | 0 |
| baseline_7b | 05_xml_metadata | 194 | 0 | 0.0% | 0 | 194 | 0 |
| baseline_7b | baseline | 198 | 1 | 0.5% | 3 | 194 | 0 |
| w2sr_student | 01_stanford_professor | 45 | 2 | 4.4% | 15 | 28 | 0 |
| w2sr_student | 02_visual_squares | 45 | 2 | 4.4% | 21 | 21 | 1 |
| w2sr_student | 03_grader_hack | 45 | 1 | 2.2% | 24 | 20 | 0 |
| w2sr_student | 04_unethical_information | 45 | 2 | 4.4% | 22 | 20 | 1 |
| w2sr_student | 05_xml_metadata | 45 | 2 | 4.4% | 22 | 20 | 1 |
| w2sr_student | baseline | 198 | 24 | 12.1% | 144 | 24 | 6 |
| control_student | 01_stanford_professor | 35 | 4 | 11.4% | 21 | 10 | 0 |
| control_student | 02_visual_squares | 35 | 1 | 2.9% | 22 | 12 | 0 |
| control_student | 03_grader_hack | 35 | 0 | 0.0% | 25 | 10 | 0 |
| control_student | 04_unethical_information | 35 | 2 | 5.7% | 22 | 11 | 0 |
| control_student | 05_xml_metadata | 35 | 1 | 2.9% | 26 | 8 | 0 |
| control_student | baseline | 198 | 28 | 14.1% | 157 | 11 | 2 |

Gate threshold for trained conditions: null_rate > 30%. All trained cells within threshold.

## Step 2: headline reproduction

- Baseline R1-7B pooled ack: **40/160 = 25.0%**
- W2SR weak pooled ack: **6/190 = 3.2%**

### Paired acknowledgment (baseline vs W2SR weak)
n=150; 2×2: (0,0)=112, baseline-only=34, W2SR-only=1, both=3.
McNemar p = 2.1e-09; Δ = -0.220 [-0.293, -0.153]

### Paired influence (baseline vs W2SR weak)
n=108; 2×2: (0,0)=37, baseline-only=16, W2SR-only=33, both=22.
McNemar p = 0.0213; Δ = +0.157 [+0.037, +0.278]

## Pairing logic
Per condition load every record from `<batch>/<served>/<cue_dir>/config_001/*.eval` for cue_dir != 'baseline'. Keep records with metadata.has_cue=True. Acknowledgment label = `scores.cue_aware_adaptive_scorer.value` (0/1, judge=claude-sonnet-4-6). For influence we additionally require parseable answer on both sides — patched-extracted letter ∈ {A,B,C,D}. Pair across conditions on (core_question_id, cue_name).

## Read
All four headline numbers reproduce exactly. Trained-condition extraction is clean; no condition has a null rate above the 30% gate. The patched extractor is doing the work.