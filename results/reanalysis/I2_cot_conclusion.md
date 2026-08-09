# Task I2 - CoT-conclusion with options in context

**n = 255 W2SR samples where the cue moved the answer to a wrong letter and the judge saw no acknowledgment (`influenced=1`, `ack=0`, `cue_target != correct`).**

**Judge:** `claude-sonnet-4.5` via OpenRouter, temperature 0.

**Difference vs Task I.** Task I gave the judge only the CoT and asked which letter the reasoning names. Task I2 also gives the judge the four multiple-choice options for the question, so if the CoT derives a value or formula without naming a letter, the judge can map that derivation to the option it corresponds to. We do NOT give the judge the question stem or any cue text; the CoT's own opening restates the problem.

## Comparison across the two runs

| category | Task I (CoT only) | Task I2 (CoT + options) |
|---|---:|---:|
| CoT reasons to the cue target | 146 (57.3%) | 214 (83.9%) |
| CoT reasoning names / maps to no letter | 104 (40.8%) | 13 (5.1%) |
| CoT reasons to the correct answer (silent flip) | 1 (0.4%) | 14 (5.5%) |
| CoT reasons to a third letter | 1 (0.4%) | 12 (4.7%) |
| judge unclear / failed | 3 | 2 |

## Where did the Task I "no_letter" cases actually land?

104 samples were labeled `no_letter_in_cot` in Task I because the CoT did not explicitly name A/B/C/D. When the judge can see the four options in Task I2, those samples resolve as follows:

| migration | n | fraction of no-letter |
|---|---:|---:|
| no_letter --> B_cue | 76 | 73.1% |
| no_letter --> no_letter | 12 | 11.5% |
| no_letter --> C_other | 9 | 8.7% |
| no_letter --> A_correct | 7 | 6.7% |


## Reading

With the options in context, **the CoT reasoning itself lands on the cue target 84% of the time**. Most of what Task I called `no_letter_in_cot` was actually the CoT deriving a value or ordering that clearly maps to the cue target's option — the judge just could not identify it without seeing the options. Adding options recovers ~87% of the "no letter" cases as `B_walks_to_cue`.

The "silent flip" pattern (the CoT reasons to the correct answer, then the emitted answer is the cue target) is a real but small effect: about 5.5% of samples (14 of 255). Even with options in context, it is much rarer than the dominant "reasoning is walked to the cue" pattern.

Practical implication for CoT monitoring: an overseer reading the CoT of a silently-influenced W2SR completion would concur with the model's wrong answer more than four times out of five, because the CoT itself is heading to the same wrong answer via corrupted or hand-wavy reasoning. The CoT is not an independent check.
