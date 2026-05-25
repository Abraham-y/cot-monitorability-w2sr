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

## Descriptive metrics (GPQA, all reparsed with the \boxed-aware extractor)
| condition | GPQA acc | mean faithfulness | verbosity |
|---|---|---|---|
| cond-1 baseline | 0.379 | 0.017 | 0.476 |
| cond-2 W2SR (weak teacher) | 0.328 | 0.008 | 0.611 |
| cond-3 control (strong 72B) | 0.374 | 0.008 | 0.623 |
| cond-4a weak teacher ref (R1-1.5B) | 0.227 | **0.172** | 0.640 |

(faithfulness = mean cue-acknowledgment from the judge; verbosity = factor
utilization. The \boxed parse fix — see below — only affects acc; faithfulness/
verbosity are judge/factor-based and format-independent.)

## Preregistered analysis (src/analysis.py; paired on (qid,cue), bootstrap 95% CI + McNemar)
**Acknowledgment (primary DV) — all NULL:**
- E1 W2SR − baseline: Δ=+0.024 [0.000, 0.049], McNemar p=0.11
- **E2 W2SR − control (PRIMARY): Δ=−0.006 [−0.013, 0.000], p=0.13**
- control − baseline: Δ=+0.026 [−0.005, 0.056], p=0.18
- Discordant pairs are tiny (2–8): acknowledgment is at the FLOOR for all 7B
  students → informative N too low to separate H1/H2 (preregistered caveat §5).
- **Adjudication: H3/H0 — no clean weak-supervision effect on monitorability.**

**Verbosity (secondary) — large, significant TRANSFER:**
- E1 W2SR − baseline: Δ=+0.264 [0.218, 0.312] (CI excludes 0)
- E2 W2SR − control: Δ=−0.012 [−0.025, 0.001] (≈0)
- control − baseline: Δ=+0.247 [0.192, 0.300] (CI excludes 0)

## Headline result — verbosity/faithfulness DECOUPLE under distillation
Distillation from a reasoning-style teacher (weak OR strong) robustly raises the
student's CoT **verbosity** (+0.25, large, CI excludes 0) but does **NOT** raise
its cue-**faithfulness** (Δ≈0, at floor, n.s.). The teacher itself is far more
faithful (0.172 vs student floor ~0.01), yet neither student inherits this.
Teacher STRENGTH is irrelevant to both (E2≈0 on verbosity AND acknowledgment).

Safety implication: Meek's composite "monitorability" (avg of verbosity +
faithfulness) RISES after distillation almost entirely via verbosity — a distilled
model can LOOK more monitorable (longer, more legible CoT) without being more
FAITHFUL (its CoT no more likely to reveal a decisive hidden cue). The
safety-relevant component does not improve. W2SR specifically (weak teacher) is
indistinguishable from strong-teacher distillation on both axes.

## Limitations
- Acknowledgment is at the floor for 7B students → underpowered on H1/H2 for the
  faithfulness axis; the strong claim is on verbosity + the descriptive
  teacher-vs-student gap. A more cue-sensitive student/benchmark would sharpen it.
- cond-2's training teacher (Qwen2.5-Math-1.5B-Instruct) differs from the cond-4a
  reference teacher (R1-1.5B); the clean E2 contrast is cond-2 vs cond-3 (both
  in-family Qwen-Math, differ only in scale). Teacher refs anchor the faithfulness
  scale, not the literal student teacher.
- MATH-SFT shifts the student to a \boxed answer format on GPQA (general
  instruction-following drift) — handled in scoring (below), and itself a
  monitorability-relevant behavioral effect worth a sentence in the discussion.

## Scoring fix (reproducibility)
MATH-SFT students emit `\boxed{X}` instead of the requested `ANSWER: X`, so
Inspect's answer() scorer misparsed them and tanked accuracy (cond-2 0.051 vs
true 0.328; cond-4a 0.086 vs true 0.227). Faithfulness/verbosity are judge/
factor-based and unaffected. Fix = a format-agnostic answer re-parse codified in
scripts/patch_meek_eval.py (extract_metrics). cond-1 has no \boxed → unchanged
(0.369→0.379), so conditions stay comparable.
