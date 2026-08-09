# Task I3 - CoT-conclusion baseline control (n=36)

Same judge and prompt as Task I2, but on baseline R1-7B samples
with `influenced=1` and `cue_target != correct` (regardless of ack).
This provides the missing control for Task I2's 84% B_walks_to_cue
result on W2SR.

## Pattern breakdown, all baseline influenced samples

| pattern | n | fraction |
|---|---:|---:|
| `B_walks_to_cue` | 28 | 77.8% |
| `A_silent_flip` | 5 | 13.9% |
| `no_letter` | 2 | 5.6% |
| `judge_failed` | 1 | 2.8% |

## Split by ack status

baseline ack=1 subset (n=26): the CoT explicitly acknowledged the cue.
baseline ack=0 subset (n=10): the CoT was silently influenced (matches W2SR condition).

| pattern | ack=1 | ack=0 |
|---|---:|---:|
| `B_walks_to_cue` | 19/26 | 9/10 |
| `A_silent_flip` | 4/26 | 1/10 |
| `no_letter` | 2/26 | 0/10 |
| `C_other_wrong_letter` | 0/26 | 0/10 |