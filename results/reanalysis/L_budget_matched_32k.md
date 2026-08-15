# Task L — budget-matched re-run at 32k

Both arms re-run at a 32,000-token generation budget; identical to the 8k
configs in every other respect. The 8k comparison advantaged the trained arm,
because only the baseline was routinely truncated before it could answer.

## Per-arm (cued cells)

| arm | budget | n | parseable | ack | hit cap | median CoT |
|---|---|---:|---:|---:|---:|---:|
| baseline | 8k | 160 | 56.2% | 40/160 = 25.0% | 43.8% | 18,692 |
| baseline | 32k | 165 | 73.3% | 39/165 = 23.6% | 24.8% | 17,998 |
| cotsft | 8k | 180 | 78.3% | 26/180 = 14.4% | 21.7% | 13,354 |
| cotsft | 32k | 175 | 82.9% | 29/175 = 16.6% | 16.0% | 13,405 |

## Paired, baseline vs CoT-preserving arm

| budget | metric | n | Δ | 95% CI | p | disc (base/cotsft) |
|---|---|---:|---:|---|---:|---:|
| 8k | acknowledgment | 155 | -0.110 | [-0.181, -0.039] | 0.004551 | 25/8 |
| 8k | influence | 83 | +0.048 | [-0.060, +0.157] | 0.5235 | 9/13 |
| 32k | acknowledgment | 150 | -0.080 | [-0.153, -0.013] | 0.04277 | 21/9 |
| 32k | influence | 95 | +0.032 | [-0.074, +0.147] | 0.7111 | 13/16 |

## Read

- **Missingness closes.** Baseline parseable 56.2% -> 73.3%; hit-cap 43.8% -> 24.8%. The across-arm parseability gap narrows from 22.1pp to 9.5pp.
- **The behavioural null is not a truncation artifact.** The paired influence CI is essentially unchanged in width (0.217 -> 0.221) despite n rising 83 -> 95. More budget does not resolve it; only more questions would.
- **The acknowledgment effect weakens at the fairer budget.** Δ -0.110 (p = 0.004551) -> -0.080 (p = 0.04277). Baseline ack 25.0% -> 23.6%; arm 14.4% -> 16.6%.
- **Baseline uncued accuracy was substantially a budget artifact**: 0.275 at 8k -> 0.425 at 32k (all-items convention).
- Truncation is not eliminated: 24.8% of baseline cued samples still exhaust even the 32k budget.
