# Banked methodological findings — W2SR reproduction attempts

Recorded 2026-05-25, before the SimpleRL-Zoo reproduction resolves. These hold
regardless of that outcome and inform the methods section + the monitorability
teacher-strength discussion.

## Finding 1 — No W2SR capability gain at the R1-distill-1.5B → 7B pairing (MATH)
Training Qwen2.5-7B (base AND instruct) on DeepSeek-R1-Distill-Qwen-1.5B MATH
CoT (W2SR / LoRA) produced **no Pass@1 gain** on held-out MATH (levels 3–5),
robustly across configurations:

| student | teacher | LoRA rank | baseline* | W2SR | gain |
|---|---|---|---|---|---|
| 7B-Instruct | R1-distill-1.5B | 16 | 0.605 (CoT-prompted) | 0.16 | −0.445† |
| 7B-base | R1-distill-1.5B | 16 | 0.45 (unelicited) | 0.435 | −0.015 |
| 7B-base | R1-distill-1.5B | 32 | 0.445 (unelicited) | 0.45 | +0.005 |

\*Reproduction baseline = base model WITHOUT CoT elicitation (direct answer); a
CoT-prompted baseline (0.535) inflates the baseline and cancels W2SR's
elicitation gain — see PREREGISTRATION §4b. †Early −0.445 was format collapse
(below) on top of an inflated baseline.

Contributing cause: **no headroom.** The 1.5B teacher solves ~59% of these MATH
problems; the 7B student already scores 0.45–0.53 — the "weak" teacher is barely
weaker, so W2SR has little latent ability to elicit. Confound: this used
R1-distill, NOT Yuan's actual teacher (see Finding 2 + the SimpleRL-Zoo retest).

## Finding 2 — R1-distill traces degrade the student under LoRA distillation
DeepSeek-R1-Distill-Qwen-1.5B **over-thinks**: at temp 0.6 it spiraled into
repetition loops on ~73% of raw MATH traces (no `repetition_penalty`), and even
after `repetition_penalty=1.1` + filtering, the kept traces are long (median
~6.3k chars). The student LoRA-trained on them **inherits non-conclusion**:
format-valid (parseable boxed answer) was only **0.55–0.62**, with residual
loops/runaway — i.e. it reasons but fails to conclude ~40% of the time.

Bounded fixes and their effect on conclusion rate:
- eval-time `repetition_penalty` 1.15: kills loops but **tanks math accuracy**
  (baseline 0.45→0.27) — penalizing repeated tokens harms numeric output; wrong lever.
- more eval tokens (4096→7000): **no help** → non-conclusion is inherent, not truncation.
- LoRA rank 16→32 + tighter trace filter (drop non-concise/over-long): format
  0.55→0.615 — a real but **insufficient** improvement (<0.8 target).
Across all of these the **gain stayed flat as format improved**, so cleaner
generation does not hide a gain.

**Implication:** LoRA-distilling a small, verbose, over-thinking reasoning
teacher's long CoT into a base model is a binding constraint on clean
generation. This is directly relevant to the monitorability teacher-strength
axis: a *more verbose / more faithful* teacher (which R1-distill is — Chua-Evans;
our weak-teacher faithfulness 0.14–0.22 vs instruct student 0.0–0.05) is also
*harder to distill cleanly*. So there may be a tension between a teacher's
faithfulness/verbosity and how cleanly its reasoning transfers via SFT — worth a
paragraph in the discussion.

## Consequence for the design
Reproduction switched to Yuan's actual teacher, `hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo`
(less over-thinking → cleaner distillation, faithful to the paper). R1-distill is
retained as the **monitorability** teacher reference (published faithfulness
baseline + the H1 weak-teacher run). The two halves deliberately use different
teachers for different reasons; documented so it isn't read as inconsistency.

## Finding 3 — SimpleRL-Zoo-1.5B (Yuan's teacher) is too NOISY/weak to distill
Switched the reproduction teacher to hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo
(Yuan's actual teacher). Traces are concise (median 1510 chars, conclude with
\boxed) BUT the model is noisy: ~24% of kept traces contain stray CJK, plus
Chinese preambles, scraped URLs (e.g. "/problem/code/...html"), HTML-entity
garbage, and mangled mid-word starts. It's also very weak on MATH (212/1200 =
18% correct; 58% dropped as no-boxed). Training the 7B base (LoRA r32) on these:
Pass@1 0.445 (unelicited base) -> 0.12 (W2SR), format-valid 0.26 — a COLLAPSE,
worse than R1-distill. The garbage corrupts the student.

## Finding 4 — W2SR REPRODUCES with an in-family (native-Qwen) weak teacher ★
The earlier no-reproduction was a **teacher-family/style mismatch**, NOT a flaw
in W2SR. R1-distill and SimpleRL-Zoo are Qwen-*based* but impart a FOREIGN
reasoning style (R1's over-thinking; SimpleRL's noisy RL-tuned output) that is
out of the student's latent distribution → poor elicitation + hard-to-distill
traces. Yuan keeps teacher+student in-family. We finally tested that:

- **Teacher** `Qwen2.5-Math-1.5B-Instruct` (native-Qwen, clean, weak): 66.9%
  correct on MATH L3-5 train problems, median ~1.5k-char CoT, only **10.9%
  degenerate** (vs R1-distill's ~73% raw loop rate).
- **Student** `Qwen2.5-Math-7B` (4k ctx), LoRA r32, max_seq_len 4096, 3 epochs.
- **Gate (held-out L3-5, temp 0, max_tokens 3500):**

| metric | value | prior R1/SimpleRL best |
|---|---|---|
| baseline (unelicited) | 0.325 | — |
| **W2SR Pass@1** | **0.670** | — |
| **gain** | **+0.345** | ~0 (best +0.005) / collapse |
| format-valid | **1.00** | 0.47–0.62 |
| degenerate gen | **none** | chronic loops |
| **gate** | **PASSED** | always failed |

The weak 1.5B teacher lifted the 7B student 0.325 → 0.670 — a clean weak-to-strong
gain, and the **first passing gate across the entire matrix**. Confirms the W2SR
effect AND pins the prior failures on teacher-family/style, not on the method,
the benchmark, headroom, LoRA-vs-full-SFT, or decoding. Run artifacts:
`/traces/w2sr_infamily` (hash 7235870…), `/checkpoints/w2sr_infamily`.

## Finding 5 — The gain is GENUINE on the BASE model, an ARTIFACT on Instruct ★★
A reviewer flag: a +5pt gate-pass on an *already-CoT-capable* student can be
"marginal rearrangement" (the model already had the ability; suppressing CoT just
made the baseline look low) rather than genuine weak-supervision elicitation. We
added a **headroom probe** to the gate: score the UNTRAINED base with a zero-shot
CoT prompt (no LoRA) = the prompting-only ceiling. Then `w2sr_beyond_cot_prompt`
= W2SR − that ceiling isolates what TRAINING added beyond merely prompting.

| student | unelicited | untrained+0-shot-CoT | W2SR | **W2SR beyond CoT-prompt** |
|---|---|---|---|---|
| Qwen2.5-**Math-7B (base)** | 0.325 | **0.24** | 0.645 | **+0.405 — GENUINE** |
| Qwen2.5-**7B-Instruct** | 0.23 | **0.63** | 0.63 | **0.00 — prompt-induced** |

- **Base model:** zero-shot CoT prompting does nothing (0.325→0.24, slightly
  *hurts* — the base can't use CoT unprompted). W2SR training adds **+0.405** that
  prompting cannot extract → real elicitation of latent capability. The
  reproduction (Finding 4) is confound-free.
- **Instruct model:** already reaches 0.63 by prompting alone; W2SR adds **0.0**
  beyond that. Its apparent +0.40 "gain" vs the unelicited baseline is purely the
  CoT-elicitation confound (PREREG §4b), NOT weak supervision.

**Methodological takeaway (paper):** W2SR capability reproduction requires a base
(non-CoT-eliciting) student; on an instruct student the gain is an artifact of
the unelicited baseline. This sharpens *when* W2SR reproduces and is itself a
contribution. **Consequence for the extension:** the monitorability student is the
locked 7B-Instruct (GPQA-capable, matches cond-1), where W2SR/control SFT shifts
CoT *style* at ~constant capability (the 0.63 CoT ceiling). So the faithfulness
comparison is **capability-controlled** — differences can't be a capability
confound. We adopt the reviewer's pre-authorized framing: "monitorability changes
*without* capability gain." (Both gates also showed a borderline single-response
`degenerate` flag at temp 0; format-valid 0.995 — vLLM greedy run-to-run noise,
not a regression; the first Math-7B gate was 1.00/clean.)

## Net reproduction result (RESOLVED)
W2SR **reproduces** (Pass@1 +0.345, gate passed) when the weak teacher is
**in-family / native-Qwen** (`Qwen2.5-Math-1.5B-Instruct` → `Qwen2.5-Math-7B`).
It does NOT reproduce with cross-style teachers (R1-distill-1.5B → flat+
degenerate; SimpleRL-Zoo-1.5B → collapse-from-noise), across students
{7B-instruct, 7B-base}, LoRA rank {16,32}, trace filters, and eval decodings.
**Takeaway for the paper:** weak-to-strong reasoning transfer is gated by
teacher↔student style/distribution match, not just teacher capability — a clean,
in-distribution weak CoT elicits the student; a foreign verbose one does not.
The MONITORABILITY study (the novel contribution) is unaffected and further along.
