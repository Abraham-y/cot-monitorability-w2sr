# Task 3 — think-channel collapse: imitation or emergent

## Step 1: training trace datasets

Each W2SR student's trace dataset was downloaded via `modal volume get w2sr-vol /traces/<name>/train.json`. Provenance taken from `checkpoints/<student>/train_provenance.json`.

| trace set | student trained on it | n | <think> | </think> | \boxed | ANSWER: | median chars |
|---|---|---:|---:|---:|---:|---:|---:|
| /vol/traces/w2sr | R1-7B W2SR weak   (R1-1.5B teacher) | 729 | 0.0% | 100.0% | 100.0% | 0.0% | 6,272 |
| /vol/traces/w2sr_r1_14b | R1-7B W2SR strong (R1-14B teacher) | 657 | 0.0% | 100.0% | 100.0% | 0.0% | 6,046 |
| /vol/traces/w2sr_infamily | Instruct W2SR weak   (Qwen2.5-Math-1.5B teacher) | 891 | 0.0% | 0.0% | 100.0% | 0.0% | 1,417 |
| /vol/traces/w2sr_infamily_strong | Instruct W2SR strong (Qwen2.5-Math-72B teacher) | 819 | 0.0% | 0.0% | 100.0% | 0.0% | 1,500 |

*Note on the 0% `<think>` open + 100% `</think>` close on R1 traces:* the R1 generation prompt ends `<|Assistant|><think>`, so the stored 'output' string begins *inside* the think channel and ends with the closing tag. **This table describes the trace FILES, not the supervision.** The R1-Distill chat template splits assistant content on `</think>` and keeps only the last segment, so rendering these rows through it yields answer-only training text with the reasoning removed. Any inference of the form 'the traces contained the CoT, therefore the student was trained on the CoT' is invalid — that was the error in earlier versions of this task. See `src/train_student.build_sft_text`.

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
| Self-A (R1-7B self, 4k bud) | cued | 95 | 0.0% | 28.4% | 0.0% | 86.3% | 1,258 |
| Self-A (R1-7B self, 4k bud) | uncued | 40 | 0.0% | 12.5% | 2.5% | 47.5% | 356 |
| Self-B (R1-7B self, 8k bud) | cued | 100 | 0.0% | 26.0% | 4.0% | 82.0% | 1,291 |
| Self-B (R1-7B self, 8k bud) | uncued | 40 | 0.0% | 7.5% | 0.0% | 47.5% | 343 |
| baseline R1-7B (MMLU) | cued | 190 | 0.0% | 86.8% | 6.3% | 81.6% | 3,223 |
| baseline R1-7B (MMLU) | uncued | 40 | 0.0% | 97.5% | 17.5% | 80.0% | 3,332 |
| W2SR weak (MMLU) | cued | 195 | 0.0% | 43.6% | 11.8% | 86.7% | 1,141 |
| W2SR weak (MMLU) | uncued | 40 | 0.0% | 25.0% | 15.0% | 85.0% | 1,116 |
| Self-A (MMLU) | cued | 195 | 0.0% | 48.2% | 8.7% | 91.3% | 1,192 |
| Self-A (MMLU) | uncued | 40 | 0.0% | 37.5% | 15.0% | 87.5% | 1,141 |
| instruct baseline (Qwen2.5-7B-Inst) | cued | 970 | 0.0% | 0.0% | 1.1% | 98.1% | 1,113 |
| instruct baseline (Qwen2.5-7B-Inst) | uncued | 198 | 0.0% | 0.0% | 1.5% | 96.5% | 1,456 |
| instruct W2SR weak | cued | 225 | 0.0% | 0.0% | 50.7% | 43.6% | 1,962 |
| instruct W2SR weak | uncued | 198 | 0.0% | 0.0% | 83.8% | 9.6% | 2,019 |
| instruct W2SR strong (control) | cued | 175 | 0.0% | 0.0% | 69.7% | 28.6% | 1,858 |
| instruct W2SR strong (control) | uncued | 198 | 0.0% | 0.0% | 92.9% | 5.6% | 2,115 |

## Verdict
R1 baseline emits </think> on 57% of cued completions; R1-7B W2SR weak on 22%; R1-7B W2SR strong on 38%. The trace FILES carry </think> in 100% of records, but that is not what the student was trained on: the R1-Distill chat template splits assistant content on </think> and keeps only the final segment, so the tokenized supervision was the answer only, with the reasoning span removed. The drop in </think> emission and the CoT compression are therefore explained by the supervision format, NOT emergent under SFT. (Earlier versions of this file asserted the opposite; the check was run on the trace files rather than on the rendered training text. See src/train_student.build_sft_text.) Compression measured here: (18,692 → 1,362 chars median; 13.7× shorter).
