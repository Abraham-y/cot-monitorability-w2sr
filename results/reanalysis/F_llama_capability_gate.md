# Task 1 — Llama cross-family capability gate (v3, clean infra)

## Why v3

v1 had 50% empty completions on both conditions (Modal vLLM cold-start saturation under 24 parallel requests). v3 adds: real warmup probe (send a completion, verify non-empty content), lowered eval concurrency to 4, raised max_tokens to 16000 (no truncation), raised VLLMServer max_model_len to 32768, and Modal scaledown_window to 30 min. v3 reads 0% empty on both conditions.

## Results (GPQA Diamond, baseline-pass only)

| condition | n | empty | parseable | acc (total) | acc (non-empty) | acc (parseable) | median CoT chars (non-empty) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline R1-Distill-Llama-8B | 30 | 0/30 = 0.0% | 29/30 = 96.7% | 15/30 = 50.0% | 15/30 = 50.0% | 15/29 = 51.7% | 15420 |
| Llama-Self-B (R1-Llama-8B self, 8k bud) | 40 | 0/40 = 0.0% | 40/40 = 100.0% | 9/40 = 22.5% | 9/40 = 22.5% | 9/40 = 22.5% | 1424 |

Note: baseline has n=30 (10 samples were retry-exhausted at the concurrency=4 setting and dropped from the final log without empties in the kept set); Self-B has n=40 (no retries needed). Gate comparison is on accuracy rates, not paired (qid, cue), so the different sample sizes are not a problem.

## Infra / sanity checks

- baseline empty rate: 0.0% (≤10.0% required to trust eval) — **PASS**
- Self-B empty rate: 0.0% (≤10.0%) — **PASS**
- baseline non-empty accuracy: 50.0% (≥30.0% required for plausibility) — **PASS**

Eval is trustworthy. Computing the capability gate.

## Capability gate

- accuracy drop (total denom):    **+27.5 pp**
- accuracy drop (non-empty denom):**+27.5 pp**
- accuracy drop (parseable denom):**+29.2 pp**
- CoT compression (non-empty median): **10.8×** (15420 → 1424 chars)
- gate threshold: >5.0pp drop = FAIL

**GATE: FAIL**

FAIL (clean eval). At the Qwen-matched recipe, R1-Distill-Llama-8B self-distillation loses ~28 pp of GPQA accuracy and compresses CoT 10.8× (15,420 → 1,424 chars median, non-empty). This is genuine substrate sensitivity: the same SFT recipe that holds Qwen-7B capability (Qwen Self-B: GPQA acc preserved, ~14× compression) craters Llama-8B. Per the spec, NOT proceeding to monitorability eval; report as substrate-dependence finding.

## Implication for the paper
Cross-family generalization at the **matched Qwen recipe** is NOT supported. R1-Distill-Llama-8B is more sensitive to math-CoT LoRA SFT than R1-Distill-Qwen-7B: the same recipe that compresses Qwen's CoT ~14× while *preserving* GPQA accuracy compresses Llama's CoT ~11× while *losing* ~28pp of GPQA accuracy. This places the Llama-Self-B arm squarely in the over-compression confound regime the spec named, and per protocol we DO NOT run the cued monitorability eval (its faithfulness number would be uninterpretable). Reported as substrate-dependence: the dissociation we report on Qwen-R1-distill may or may not generalize cross-family; at matched recipe Llama-8B fails the capability gate before that question becomes measurable.
