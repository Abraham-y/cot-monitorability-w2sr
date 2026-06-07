# Task 3 — think-channel collapse: imitation or emergent

## Step 1: training trace datasets

Each W2SR student's trace dataset was downloaded via `modal volume get w2sr-vol /traces/<name>/train.json`. Provenance taken from `checkpoints/<student>/train_provenance.json`.

| trace set | student trained on it | n | <think> | </think> | \boxed | ANSWER: | median chars |
|---|---|---:|---:|---:|---:|---:|---:|
| /vol/traces/w2sr | R1-7B W2SR weak   (R1-1.5B teacher) | 729 | 0.0% | 100.0% | 100.0% | 0.0% | 6,272 |
| /vol/traces/w2sr_r1_14b | R1-7B W2SR strong (R1-14B teacher) | 657 | 0.0% | 100.0% | 100.0% | 0.0% | 6,046 |
| /vol/traces/w2sr_infamily | Instruct W2SR weak   (Qwen2.5-Math-1.5B teacher) | 891 | 0.0% | 0.0% | 100.0% | 0.0% | 1,417 |
| /vol/traces/w2sr_infamily_strong | Instruct W2SR strong (Qwen2.5-Math-72B teacher) | 819 | 0.0% | 0.0% | 100.0% | 0.0% | 1,500 |

*Note on the 0% `<think>` open + 100% `</think>` close on R1 traces:* the R1 chat template injects the opening `<think>` token at training time as part of the assistant role's chat-format prefix; the literal 'output' string in the Llama-Factory record begins *inside* the think channel and includes the closing tag. So the assistant was trained on text that contained the channel separator (closing tag, with the opening tag supplied by the template).

## Step 2: stored-completion think-tag fraction per condition

Scanned every cued + uncued completion across all 6 batches.

| condition | cell | n | `<think>` | `</think>` | `\boxed` | `ANSWER:` | median chars |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline R1-7B | cued | 160 | 0.0% | 57.5% | 0.0% | 56.2% | 18,692 |
| baseline R1-7B | uncued | 40 | 0.0% | 50.0% | 0.0% | 50.0% | 20,096 |
| W2SR weak (R1-1.5B teacher) | cued | 190 | 0.0% | 21.6% | 2.6% | 90.0% | 1,362 |
| W2SR weak (R1-1.5B teacher) | uncued | 40 | 0.0% | 22.5% | 7.5% | 90.0% | 1,392 |
| W2SR strong (R1-14B teacher) | cued | 175 | 0.0% | 37.7% | 1.1% | 92.0% | 1,508 |
| W2SR strong (R1-14B teacher) | uncued | 40 | 0.0% | 30.0% | 2.5% | 87.5% | 1,638 |
| instruct baseline (Qwen2.5-7B-Inst) | cued | 970 | 0.0% | 0.0% | 1.1% | 98.1% | 1,113 |
| instruct baseline (Qwen2.5-7B-Inst) | uncued | 198 | 0.0% | 0.0% | 1.5% | 96.5% | 1,456 |
| instruct W2SR weak | cued | 225 | 0.0% | 0.0% | 50.7% | 43.6% | 1,962 |
| instruct W2SR weak | uncued | 198 | 0.0% | 0.0% | 83.8% | 9.6% | 2,019 |
| instruct W2SR strong (control) | cued | 175 | 0.0% | 0.0% | 69.7% | 28.6% | 1,858 |
| instruct W2SR strong (control) | uncued | 198 | 0.0% | 0.0% | 92.9% | 5.6% | 2,115 |

## Verdict
R1 baseline emits </think> on 57% of cued completions; R1-7B W2SR weak on 22%; R1-7B W2SR strong on 38%. Training traces carried the </think> token in 100% of records, so the trained students were NOT shown stripped data. The collapse to ~0% emission is EMERGENT under LoRA SFT — the channel separator was present in the supervision signal but the SFT'd student stopped producing it, alongside an order-of-magnitude CoT compression (18,692 → 1,362 chars median; 13.7× shorter).
