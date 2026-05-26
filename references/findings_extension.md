# Findings — monitorability extension (the novel contribution)

Recorded 2026-05-25. The reproduction findings are in findings_repro.md; this
file holds the EXTENSION results (CoT monitorability of W2SR vs control vs
baseline students + teacher refs, GPQA-Diamond, 5 Meek cues, judge =
anthropic/claude-sonnet-4-6, temp 0). Numbers live here because results/ is
gitignored (large .eval logs); regenerate with extract_metrics + src/analysis.py.

## Conditions (all students = Qwen2.5-7B-Instruct base, served identically)
- **cond-1 baseline:** untrained 7B-Instruct.
- **cond-2 W2SR:** 7B-Instruct + LoRA SFT on WEAK in-family teacher
  (Qwen2.5-Math-1.5B-Instruct) MATH L3-5 traces.
- **cond-3 control:** 7B-Instruct + LoRA SFT on STRONG in-family teacher
  (Qwen2.5-Math-72B-Instruct), SAME MATH L3-5 problems → clean teacher-strength
  axis (weak 1.5B vs strong 72B, 48× scale; both native-Qwen, same recipe).
- **cond-4a weak teacher ref:** DeepSeek-R1-Distill-Qwen-1.5B (its own GPQA CoT).
- **cond-4b strong teacher ref (R1-32B):** NOT RUN. Attempted via OpenRouter
  (impractically slow ~10 samples/20min, truncated non-reasoning output) and via
  Modal-served (2×A100, enforce-eager + 32-way concurrency) — but the full
  R1-32B reasoning eval (6 passes × 198 Qs × up to 30k-token CoT) is ~20 GPU-hr,
  beyond the <16h budget for an OPTIONAL reference arm. cond-4a anchors the
  teacher-faithfulness scale; the core E1/E2 student comparison does not need 4b.
  Future work: a subset (≤50 Q) or a smaller/length-capped strong teacher.

NB capability is CONTROLLED, not gained: per the headroom probe (findings_repro
Finding 5) the in-family SFT lands the Instruct student at ~its zero-shot-CoT
ceiling on MATH; on GPQA all three students sit at ~0.33-0.38 accuracy. So
faithfulness differences are NOT a capability confound.

## Descriptive metrics (GPQA, reparsed with the \boxed-aware extractor)
**Faithfulness is reported as acknowledgment among CUED samples (acked / cued),
never / 198.** The adaptive cue passes contain both cued and non-cued samples;
dividing by all 198 (an earlier error that produced a spurious "0.008") deflates
the rate, especially for the trained students whose cued sets are smaller (see
the sibling-bug note in Limitations). Informative N = the cued count.

| condition | GPQA acc | faithfulness (acked/cued) | verbosity |
|---|---|---|---|
| cond-1 baseline | 0.379 | **17/970 = 1.8%** | 0.476 |
| cond-2 W2SR (weak teacher) | 0.328 | **8/225 = 3.6%** | 0.611 |
| cond-3 control (strong 72B) | 0.374 | **8/175 = 4.6%** | 0.623 |
| cond-4a weak teacher ref (R1-1.5B) | 0.227 | **~26/117 = 22%** (37% on the stanford cue) | 0.640 |

(faithfulness = judge cue-acknowledgment; verbosity = factor utilization. Both
judges read the full completion prose, NOT the parsed answer → format-independent,
verified by code + a hand-check of transcripts across all three conditions. The
\boxed parse fix below affects accuracy only.)

## Faithfulness finding — acknowledgment is at the FLOOR (stated precisely)
- Acknowledgment is **at the floor across all student conditions** (1.8–4.6%).
  This floor is **established robustly by the baseline at N=970** (well-powered),
  independent of the smaller cued-N for the trained students.
- The **weak teacher reaches 22–37%**, confirming the scorer registers
  acknowledgment when it is present → the students' floor is **real behavior**
  (7B Instruct models rarely verbalize a decisive cue), not measurement failure.
- **Distillation does not lift the students off this floor.** We make the
  FLOOR claim strongly; we do NOT claim a precise no-transfer magnitude — only
  that **no transfer is detectable above a floor that all student conditions
  share**. (Preregistered paired test, src/analysis.py: E2 W2SR−control CI
  includes 0; discordant pairs 2–8 — consistent with floor, underpowered for
  small between-student differences.)
- **Direction (rules out H1 inheritance):** with correct denominators the trained
  students drift *slightly above* baseline (3.6/4.6% vs 1.8%) — noise at these
  counts, but it goes the OPPOSITE way from H1. There is no evidence distillation
  drags students toward the teacher's much higher (22–37%) faithfulness.

## Verbosity finding — large, robust TRANSFER (the strong claim / headline)
- E1 W2SR − baseline: **Δ=+0.264 [0.218, 0.312]** (CI excludes 0, large N)
- control − baseline: **Δ=+0.247 [0.192, 0.300]** (CI excludes 0)
- E2 W2SR − control: Δ=−0.012 [−0.025, 0.001] (≈0; teacher strength irrelevant)

## Headline — verbosity and faithfulness DECOUPLE under distillation
Distillation from a reasoning-style teacher (weak OR strong) robustly raises CoT
**verbosity** (+0.25, tight CI, large N) while cue-**faithfulness stays at the
floor**. Meek's composite "monitorability" (avg of verbosity + faithfulness)
therefore rises after distillation almost entirely via verbosity: **a distilled
model LOOKS more monitorable (longer, more legible CoT) without being more
FAITHFUL** — its CoT is no more likely to reveal a decisive hidden cue. The
safety-relevant component does not improve, and W2SR (weak teacher) is
indistinguishable from strong-teacher distillation on both axes.

## Why we did NOT re-run to raise cond-2/3 power (justification for "option A")
The central faithfulness claim is about the FLOOR, which the **baseline
establishes at N=970** — this does not depend on cond-2/cond-3's informative N.
Re-running the trained conditions would raise their N but cannot change the
baseline-anchored floor claim, and we decline to assume their result. (This is
NOT "we expect a null"; it is "the claim we make does not rest on those cells'
power.")

## Limitations
- **Power / floor:** cue acknowledgment is at the floor across all student
  conditions; between-student faithfulness comparisons are underpowered. We claim
  the (baseline-anchored) floor, not a precise no-transfer magnitude.
- **Sibling parsing bug (transparent):** MATH-SFT students emit `\boxed{X}` rather
  than the requested `ANSWER: X`. The adaptive cue generator
  (generate_adaptive_datasets.py) parses only `ANSWER:`/`answer is`, so it created
  cues for ~4× fewer samples in the MATH-SFT'd conditions (cued: baseline 970 vs
  W2SR 225 vs control 175). The acknowledgment RATE on the cued subset is
  **unbiased** — both judges read the full completion, verified by a hand-check
  across all three conditions — but the reduced N **compounds** the floor-power
  limit for between-student comparisons. Final accuracy scoring is fixed
  (patch_meek_eval.py); the upstream cued-set shrinkage would require re-running
  the cue passes, which we did not do (see "option A" above).
- **Teacher-of-record:** cond-2's training teacher (Qwen2.5-Math-1.5B-Instruct)
  differs from the cond-4a reference teacher (R1-1.5B); the clean E2 contrast is
  cond-2 vs cond-3 (both in-family Qwen-Math, differ only in 48× scale). Teacher
  refs anchor the faithfulness scale, not the literal student teacher.
- **Scope:** small-scale, LoRA (not full SFT), public 1.5B/72B teachers, a single
  judge (claude-sonnet-4-6), faithfulness at floor, and GPQA-Diamond only for the
  extension. cond-4b (strong teacher reference) not run (~20 GPU-hr; see top).

## ★ Reasoning-student extension (the home run) — W2SR DEGRADES faithfulness
The instruct study is floored, so it can only show "no transfer above floor."
To probe a substrate where faithfulness has dynamic range, we re-ran E1 on a
REASONING student, fully in-family: student `DeepSeek-R1-Distill-Qwen-7B`, weak
teacher `DeepSeek-R1-Distill-Qwen-1.5B` (the cond-4a teacher; traces reused from
/traces/w2sr). Capability-controlled (report Pass@1, don't chase a gain). In-family
distillation was CLEAN (gate format-valid 0.99). Reduced to 40 GPQA Qs (the eval is
throughput-bound at ~1.4 samples/min on long-CoT students; informative N preserved
via the \boxed fix). Pre-check confirmed the substrate is off-floor (baseline
stanford-cue acknowledgment 46%).

**E1 result (cond-1' baseline R1-7B vs cond-2' W2SR R1-7B), acknowledgment among cued:**
| cue | baseline | W2SR |
|---|---|---|
| stanford_professor | 15/32 = 47% | 3/38 = 8% |
| unethical_information | 18/32 = 56% | 2/38 = 5% |
| grader_hack | 7/32 = 22% | 1/38 = 3% |
| visual_squares / xml_metadata | 0% / 0% | 0% / 0% (non-text cues, floor for both) |
| **POOLED** | **40/160 = 25.0%** | **6/190 = 3.2%** |

**Paired (qid,cue) test:** Δ = **−0.220** [95% CI −0.293, −0.153], **McNemar
p = 2.1e-9**, n=150 pairs, discordant **34/1** (one-directional). So W2SR
distillation **significantly DEGRADES** cue-faithfulness on a reasoning student —
an effect INVISIBLE on the floored instruct student.

**Not an artifact (hand-checked):** W2SR completions are coherent (37/40 conclude
with a boxed answer, 0 degenerate); the model genuinely solves the problem WITHOUT
verbalizing the cue. **Mechanism:** distilling the terser 1.5B teacher collapses
the strong reasoner's CoT length (median ~20,100 → ~1,392 chars, ~14×); briefer,
more-direct reasoning surfaces decisive cues far less. **Capability preserved:**
GPQA acc baseline 0.275 vs W2SR 0.425 (n=40, comparable/noisy) — the faithfulness
collapse is NOT a capability drop. So W2SR yields a concise, capable, but
NON-REVEALING reasoner: monitorability collapses silently to an accuracy check.

**Length confound (faithfulness AND CoT length both drop — is length the
mechanism?).** Two checks, both pointing to "not the whole story":
- *Step 1 (free, W2SR-side matched length):* among the 3 text cues, W2SR traces
  that are as long as the baseline's typical CoT still rarely acknowledge:
  len≥9.6k chars (baseline p25) → W2SR 1/8=12.5% vs baseline 32/72=44%;
  len≥20k (> baseline median) → W2SR 1/7=14% vs baseline 23/52=44%. W2SR's 12
  longest traces (4k–40k chars): 1/12 acknowledge. So at MATCHED long length W2SR
  is still ~3× less faithful.
- *Step 2a (baseline-short control): INFEASIBLE / negative.* We tried to
  constrain-generate the baseline R1-7B to short CoT (brevity directive in the
  prompt — NOT truncation, which would bias toward a false "length" result). The
  directive was applied (verified in-prompt) but R1-distill IGNORED it: median CoT
  18,262 vs 20,762 normal. R1-distill is brevity-resistant, so the symmetric
  baseline-short test cannot be run without truncation (vetoed). The W2SR-side
  matched-length evidence (step 1) carries the confound check instead.
- *Step 1b (thicken the thin cells, Exp 1):* re-ran W2SR R1-7B on the FULL 198
  GPQA (3 text cues) to push the long bins to adequate N. Matched-length result
  (W2SR full vs baseline, with 95% Wilson CIs):
  - overall: W2SR 41/558=7.3% [5.5,9.8] vs baseline 40/96=41.7%.
  - **≥9.6k chars: W2SR 6/31=19.4% [9.2,36.3] vs baseline 32/72=44.4% [34,56]**,
    two-proportion p=0.016.
  - ≥20k chars: W2SR 5/25=20.0% vs baseline 23/52=44.2%.
  The thin n=8 estimate (12.5%) was noisy; at n=31 it settled at **19.4%**, so the
  matched-length gap is **~2.3× (significant, p=0.016), not ~3×**.
- *Exp 2 (mentionability, existing data):* W2SR surfaces the cue PREMISE in its CoT
  3/114=2.6% [0.9,7.5] vs baseline 22/96=22.9% [15.6,32.3] (CIs disjoint);
  P(ack | mentioned)≈1.0 for BOTH → the gap is W2SR *not surfacing* the cue, not
  mentioning-without-crediting.
- *Framing the data supports (precommitted, applied honestly):* W2SR degrades
  monitorability via TWO compounding routes — **(1) CoT compression** (median ~14×
  shorter; W2SR's long traces recover to 19.4% vs its 7.3% overall, so shortening
  is the LARGER share of the 42%→7% drop) and **(2) a residual at-matched-length
  reduction** (19.4% vs 44.4%, p=0.016, ~2.3×). Not "purely length," not "mostly
  beyond length" — both, compression dominant. The headline 25%→3% degradation is
  robust; the mechanism is mostly-compression + a real moderate residual.

**Combined story (both substrates):** W2SR distillation does not preserve CoT
monitorability. On a floored instruct student the effect is undetectable
(underpowered); on a reasoning student with real range it is a large, significant
REDUCTION in cue-acknowledgment (25%→3%), accompanied by a ~14× CoT-length drop,
at preserved capability. The safety-relevant component (does the CoT reveal the
decisive cue?) gets worse, not better, and a capability-only check would miss it.

## ★ P2 — teacher-strength axis on the reasoning student (compression-general)
Tests whether the faithfulness degradation is specific to a WEAK teacher or to
distillation/compression generally. Same R1-Distill-Qwen-7B student, in-family
R1-distill teachers, 40 GPQA Qs, 5 cues:
| condition | faithfulness (acked/cued) | median CoT |
|---|---|---|
| baseline R1-7B | 40/160 = 25.0% [19,32] | 20,096 |
| weak-teacher W2SR (R1-1.5B) | 6/190 = 3.2% [1,7] | 1,392 |
| strong-teacher W2SR (R1-14B) | 13/175 = 7.4% [4,12] | 1,638 |

Two-proportion tests: strong vs baseline p=1.1e-5; weak vs baseline p=1.7e-9;
**weak vs strong p=0.067 (n.s.)**. So BOTH weak and strong distillation degrade
faithfulness significantly and by statistically indistinguishable amounts, with
the same ~13× CoT compression. **Teacher strength does NOT rescue monitorability**
— a 14B strong teacher degrades it nearly as much as a 1.5B weak teacher.
**Precommitted call: the driver is distillation-induced CoT COMPRESSION generally,
not weak supervision specifically.** (Corroborates the instruct-arm E2≈0: teacher
strength was irrelevant there too, but floored; here, on a substrate with range,
both teachers degrade significantly.)

## P1 — cross-family (Llama): DEGENERATION CONSTRAINT (no faithfulness conclusion)
Student DeepSeek-R1-Distill-Llama-8B, weak teacher Llama-3.2-3B-Instruct (official,
the only reliable in-family option — no sub-8B Llama R1-distill exists). Gate:
format-valid 0.805, capability cratered 0.64 (zero-shot CoT) → 0.175 (W2SR), i.e.
w2sr_beyond_cot = −0.465 — far beyond the controlled level (Qwen R1-7B was only
−0.175). The terse, non-reasoning, 44%-correct/20%-degenerate Llama-3B teacher
OVER-COMPRESSED/degraded the reasoning student. Per the precommitted rule we did
NOT run the monitorability eval (reading faithfulness off a collapsed student =
the SimpleRL error). **Cross-family generalization is INCONCLUSIVE with this
teacher**; the matched recipe over-compresses instruct→reasoning. (Optional retry
with a gentler recipe would deviate from the matched protocol — flagged for a human
decision.)

## Matched-length residual — convergent isolation across 3 cuts (existing data)
Length is endogenous to reasoning (can't hold content fixed and vary length — the
2a brevity control failed because R1 ignores brevity), so we isolate the residual
by CONVERGENCE across imperfect length-controlled cuts, not one knockout. All cuts
condition on the 3 text cues; "matched length" = traces ≥9.6k chars (baseline p25).
| cut (matched length) | baseline | W2SR | verdict |
|---|---|---|---|
| acknowledgment (judge, PRIMARY) | 32/72=44.4% | 6/31=19.4% | two-prop **p=0.016** (sig) |
| literal mention-rate | 20/72=27.8% [19,39] | 5/31=16.1% [7,33] | same dir, n.s. (CIs overlap) |
| mention density /1k chars | 0.167 | 0.037 | W2SR lower (short-trace confounded) |
| natural qid-matched (within-2×) | 4/9=44% [19,73] | 1/9=11% [2,44] | same dir, n=9, underpowered |

**All four cuts point the same direction (baseline more cue-surfacing/crediting at
matched length); significant on the primary judge measure (p=0.016), directionally
consistent but underpowered on the literal-mention and natural-match cuts.**
Precommitted call (applied honestly): the residual **persists across multiple
length-controlled cuts** — we claim that, NOT "confound eliminated." Net: W2SR
degrades monitorability **predominantly via CoT compression** (the larger, less
surprising route — overall mention 4.8% vs 22.9% collapses mostly because W2SR
traces are short) **plus a real, moderate residual** — at matched length the
distilled model still surfaces/credits decision-relevant cues less. The
matched-length residual is the non-trivial claim; lead with it, treat raw
compression as the larger-but-expected contributor.

## Scoring fix (reproducibility)
MATH-SFT students emit `\boxed{X}` instead of the requested `ANSWER: X`, so
Inspect's answer() scorer misparsed them and tanked ACCURACY (cond-2 0.051 vs
true 0.328; cond-4a 0.086 vs true 0.227). Faithfulness/verbosity are judge/
factor-based and unaffected. Fix = a format-agnostic answer re-parse codified in
scripts/patch_meek_eval.py (extract_metrics). cond-1 has no \boxed → unchanged
(0.369→0.379), so conditions stay comparable. (The same `ANSWER:`-only parse in
the adaptive generator is the sibling bug noted in Limitations.)
