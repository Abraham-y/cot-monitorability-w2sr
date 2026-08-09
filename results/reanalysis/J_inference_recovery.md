# Task J — Inference-time system-prompt recovery test

**n = 100 matched (qid, cue) samples per condition, 3 conditions = 300 generations total.**

**W2SR endpoint: `/vol/merged/w2sr_r1_7b` served via Modal vLLM (T=0).**
**Judge: `claude-sonnet-4.5` via OpenRouter, cue_aware rubric.**

## Ack rate by condition

| condition | n | ack | ack rate |
|---|---:|---:|---:|
| acknowledge | 99 | 5 | 5.1% |
| icl_example | 100 | 9 | 9.0% |
| long_cot | 99 | 6 | 6.1% |
| none | 99 | 1 | 1.0% |

## Paired McNemar (recovery vs no-system-prompt)

| comparison | n_pairs | recovery-only | none-only | both | neither | p |
|---|---:|---:|---:|---:|---:|---:|
| acknowledge vs none | 98 | 4 | 0 | 1 | 93 | 0.125 |
| icl_example vs none | 99 | 8 | 0 | 1 | 90 | 0.007812 |
| long_cot vs none | 98 | 5 | 0 | 1 | 92 | 0.0625 |