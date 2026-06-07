# Task 5 Part B — robustness rejudge

New judge: google/gemini-2.5-pro

## Cohen's κ vs original judge

| condition | n | original ack rate | new ack rate | κ |
|---|---:|---:|---:|---:|
| baseline R1-7B | 160 | 25.0% | 16.9% | 0.645 |
| W2SR weak | 190 | 3.2% | 1.6% | 0.659 |
| W2SR strong | 175 | 7.4% | 4.0% | 0.684 |

**Overall κ = 0.681**

## Paired ack drop under new judge, baseline R1-7B vs W2SR weak
n = 150; discordant 23 (baseline-only) / 0 (W2SR-only); McNemar p = 2.38e-07
Δ = -0.153 [-0.213, -0.100]