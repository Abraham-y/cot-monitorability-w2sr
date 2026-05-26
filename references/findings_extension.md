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
- *Framing the data supports:* W2SR degrades monitorability via TWO compounding
  routes — (1) CoT compression (~14× shorter → less room to reveal) and (2) at
  matched length, less cue-revealing reasoning. The strong claim (degradation
  beyond shortening) holds; W2SR long-trace N is small (7–8), so this corroborates
  the main 25%→3% result rather than standing alone.

**Combined story (both substrates):** W2SR distillation does not preserve CoT
monitorability. On a floored instruct student the effect is undetectable
(underpowered); on a reasoning student with real range it is a large, significant
REDUCTION in cue-acknowledgment (25%→3%), accompanied by a ~14× CoT-length drop,
at preserved capability. The safety-relevant component (does the CoT reveal the
decisive cue?) gets worse, not better, and a capability-only check would miss it.

## Scoring fix (reproducibility)
MATH-SFT students emit `\boxed{X}` instead of the requested `ANSWER: X`, so
Inspect's answer() scorer misparsed them and tanked ACCURACY (cond-2 0.051 vs
true 0.328; cond-4a 0.086 vs true 0.227). Faithfulness/verbosity are judge/
factor-based and unaffected. Fix = a format-agnostic answer re-parse codified in
scripts/patch_meek_eval.py (extract_metrics). cond-1 has no \boxed → unchanged
(0.369→0.379), so conditions stay comparable. (The same `ANSWER:`-only parse in
the adaptive generator is the sibling bug noted in Limitations.)
