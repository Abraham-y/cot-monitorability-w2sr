# Task C — decode-config check

**Verdict:** All under-test conditions decoded greedy, single decode (T=0, top_p=1.0, top_k=1; n/best_of unset or ≤1; epochs=1).

| condition | config path | T | top_p | top_k | n | best_of | greedy-single? |
|---|---|---|---|---|---|---|---|
| baseline R1-7B | configs/r1_7b_baseline_gpqa.yaml | 0.0 | 1.0 | 1 | — | — | OK |
| W2SR weak (R1-1.5B teacher) | configs/r1_7b_w2sr_gpqa.yaml | 0.0 | 1.0 | 1 | — | — | OK |
| W2SR strong (R1-14B teacher) | configs/r1_7b_strong_gpqa.yaml | 0.0 | 1.0 | 1 | — | — | OK |
| instruct baseline | configs/baseline_7b_gpqa.yaml | 0.0 | 1.0 | 1 | — | — | OK |
| instruct W2SR weak | configs/w2sr_student_gpqa.yaml | 0.0 | 1.0 | 1 | — | — | OK |
| instruct W2SR strong (control) | configs/control_student_gpqa.yaml | 0.0 | 1.0 | 1 | — | — | OK |