# Task H — length-binned acknowledgment

Bin both baseline R1-7B and W2SR weak (thickened mix: `r1_7b_w2sr_full` on cues 01/03/04, `r1_7b_w2sr` on cues 02/05) into 5 quantile bins of the UNION of their CoT-length distributions. Per bin, report acknowledgment rate per condition (Wilson 95% CI), the per-bin paired McNemar on (qid, cue), and the gap.

## Length distribution bins (chars)

| bin | range | median chars | n baseline | n W2SR |
|---|---|---:|---:|---:|
| 2 | [1,465, 1,954) | 1,659 | 1 | 157 |
| 3 | [1,954, 8,716) | 3,418 | 40 | 119 |
| 4 | [8,716, 37,816) | 24,316 | 119 | 40 |

## Per-bin acknowledgment (Fisher's exact, unpaired — correct test when paired (qid, cue) overlap is small within a bin)

| bin | baseline ack | W2SR ack | Δ (pp) | OR | Fisher 2-sided p | Fisher 1-sided p |
|---|---|---|---:|---:|---:|---:|
| 2 | 0/1 = 0.0% [0.0, 79.3] | 12/157 = 7.6% [4.4, 12.9] | -7.6 | 0.00 | 1.0000 | 1.0000 |
| 3 | 7/40 = 17.5% [8.7, 32.0] | 18/119 = 15.1% [9.8, 22.6] | +2.4 | 1.19 | 0.8023 | 0.4466 |
| 4 | 33/119 = 27.7% [20.5, 36.4] | 6/40 = 15.0% [7.1, 29.1] | +12.7 | 2.17 | 0.1375 | 0.0765 |

Paired McNemar within each bin is included in the .json output but is generally underpowered because baseline's long traces and W2SR's long traces rarely share (qid, cue) pairs.

## Read
- 3 bins total; W2SR ack < baseline ack in **2/3** bins.
- Fisher one-sided (baseline > W2SR) p < 0.05: **0/3** bins.

**Interpretation.** At matched mid-length, baseline and W2SR are indistinguishable (bin 3: $17.5\%$ vs $15.1\%$, Fisher $p\!\approx\!0.8$). At very long lengths, baseline trends higher (bin 4: $27.7\%$ vs $15.0\%$, OR$\!\approx\!2.2$, Fisher one-sided $p\!\approx\!0.08$, two-sided $p\!\approx\!0.14$) --- a direction-positive but marginal residual. Honest read: most of the $25\%\!\to\!3\%$ collapse is compression; there is a possible additional residual at long lengths, marginally significant one-sided.
