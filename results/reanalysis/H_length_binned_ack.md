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

## Cue-stratified: three text cues only

`visual_squares_correct` and `xml_metadata_success_rate` are 0% ack in **both** arms, so they add only zeros — but their share of a bin differs sharply by arm (bin 3: 50% of baseline's records vs 8% of W2SR's). Pooling them therefore depresses baseline's bin rate far more than W2SR's, so the pooled Fisher partly measures cue mix rather than the matched-length gap. Restricting to the three text cues removes that imbalance.

| bin | baseline ack | W2SR ack | Δ (pp) | OR | Fisher 2-sided p | Fisher 1-sided p |
|---|---|---|---:|---:|---:|---:|
| 2 | 0/1 = 0.0% | 12/141 = 8.5% | -8.5 | 0.00 | 1.0000 | 1.0000 |
| 3 | 7/20 = 35.0% | 18/109 = 16.5% | +18.5 | 2.72 | 0.0677 | 0.0586 |
| 4 | 33/75 = 44.0% | 6/32 = 18.8% | +25.2 | 3.40 | 0.0158 | 0.0102 |

## Read (computed)

- **Bin 3 (matched mid-length).** Pooled over all five cues: $17.5\%$ vs $15.1\%$, OR$\,1.19$, Fisher two-sided $p=0.802$. Restricted to the three text cues: $35.0\%$ vs $16.5\%$, OR$\,2.72$, Fisher two-sided $p=0.068$.
- **Bin 4 (long).** Pooled over all five cues: $27.7\%$ vs $15.0\%$, OR$\,2.17$, Fisher two-sided $p=0.137$. Restricted to the three text cues: $44.0\%$ vs $18.8\%$, OR$\,3.40$, Fisher two-sided $p=0.016$.

**Interpretation.** The pooled comparison understates the matched-length gap because the two floor cues are unbalanced across arms within bins. On the cue-stratified comparison the gap is present at both mid and long length. Compression still accounts for much of the raw $25\%\!\to\!3\%$ collapse, but a matched-length residual survives cue stratification without conditioning on any post-treatment variable.
