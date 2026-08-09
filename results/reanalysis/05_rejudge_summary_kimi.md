# Task 5 Part B — robustness rejudge

New judge: moonshotai/kimi-k2-0905

## Cohen's κ vs original judge

| condition | n | original ack rate | new ack rate | κ |
|---|---:|---:|---:|---:|
| baseline R1-7B | 154 | 24.7% | 12.3% | 0.475 |
| W2SR weak | 185 | 3.2% | 1.6% | 0.659 |
| W2SR strong | 166 | 7.8% | 5.4% | 0.611 |

**Overall κ = 0.556**

## Paired ack drop under new judge, baseline R1-7B vs W2SR weak
n = 140; discordant 17 (baseline-only) / 1 (W2SR-only); McNemar p = 0.000145
Δ = -0.114 [-0.171, -0.057]