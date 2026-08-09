# Reference notes — one line per paper: what we use it for

## PDFs downloaded (references/pdfs/)
- **2505.20072** Yuan et al., *Incentivizing Strong Reasoning from Weak Supervision* (W2SR). THE paper we build on. Source of the SFT recipe + the capability claim we replicate. Recipe details are in `modal_app.py` and `src/config.py` (the standalone recipe note was folded into those before release).
- **2510.27378** Meek et al., *Measuring CoT Monitorability through Faithfulness and Verbosity*. THE measurement. Source of the monitorability score, cue set, judge prompt, verbosity operationalization. Code: `external/monitorability-eval`.
- **2505.05410** Chen et al. (Anthropic), *Reasoning Models Don't Always Say What They Think*. Source of the six cue categories — pull exact hint templates.
- **2305.04388** Turpin et al., *LMs Don't Always Say What They Think*. Origin of the cued-input perturbation paradigm.
- **2602.17053** RFEval. Counterfactual reasoning-faithfulness benchmark; sets up H0 (SFT-preserves).
- **2603.22582** *Lie to Me*. Multi-judge panel + validation-judge protocol; informs our judge validation (we use single strong judge + hand-labeled set).

## Code cloned (external/)
- **w2sr** (github.com/W2SR-ARR/Code): training recipe (LLaMA-Factory), trace gen (`infer/`), math eval (`eval/`). No teacher checkpoints or trace data shipped.
- **monitorability-eval** (github.com/ajmeek/measuring_cot_monitorability): Inspect eval, scorers, cue prompts, GPQA/MMLU/BBH configs. MIT licensed. HF datasets: `ameek/causal_factors`, `ameek/measuring_cot_monitorability_transcripts`.
- **gair-w2s-reasoning** (github.com/GAIR-NLP/weak-to-strong-reasoning): alternative W2S-reasoning recipe; fallback if Yuan repo is hard to run.

## Strong-teacher trace sources (spec 8.1, control condition)
Prefer public R1 trace datasets over generating from a 32B model. Candidates to
wire into `TeacherSpec.trace_dataset`, filtered to our problem set + matched on
volume: **OpenR1-Math** (`open-r1/OpenR1-Math-220k`), **OpenThoughts**
(`open-thoughts/OpenThoughts-114k`). Both are R1-distilled long CoT on math, so
trace style matches the weak-teacher arm. Verify license + exact problem overlap
before use.

## Not obtained
- **Chua & Evans** faithfulness code (2501.08156): no clean standalone repo found; code is spread across the Owain Evans group's repos. Lower-priority cross-check (spec 13.1 item 4) and the Meek eval already implements the cued-input paradigm, so not blocking. Their published numbers still give us a free literature faithfulness baseline for the DeepSeek-R1-Distill teachers (a reason we locked that teacher series).
