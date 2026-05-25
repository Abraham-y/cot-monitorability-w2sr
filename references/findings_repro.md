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

## Net reproduction result (robust across the matrix)
W2SR with a weak 1.5B teacher does NOT reproduce a capability gain on MATH for a
7B student, across: students {7B-instruct, 7B-base}, teachers {R1-distill-1.5B,
SimpleRL-Zoo-1.5B}, LoRA rank {16,32}, trace filters, eval decodings. R1-distill
gives flat+degenerate; SimpleRL-Zoo gives collapse-from-noise. The gain Yuan
reports does not appear in this (smaller, LoRA, public-1.5B-teacher) setup.
The MONITORABILITY study (the novel contribution) is unaffected and further along.
