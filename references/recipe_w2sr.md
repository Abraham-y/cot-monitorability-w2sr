# W2SR training recipe — extracted from external/w2sr (Yuan et al.)

Source: `external/w2sr/llama-factory/examples/weak2strong/` and `examples/train_full/`.
The repo trains via **LLaMA-Factory** (`llamafactory-cli train <config>.yaml`), not TRL.

## Exact recipe (representative configs)

| Setting | 1.5B-teacher → Math-7B (`weak2strong/qwen2.5-math-7b/1.5brl-sft_base.yaml`) | qwen2.5-7b base (`train_full/qwen2.5-7b_sft_base.yaml`) |
|---|---|---|
| finetuning_type | **full** (DeepSpeed ZeRO-3) | full (ZeRO-3) |
| template | `qwen` | `qwen` |
| cutoff_len | **4096** | 4096 |
| per_device_train_batch_size | 2 | 2 |
| gradient_accumulation_steps | 16 | 16 |
| GPUs | 4 → **global batch = 128** | 4 → global batch 128 |
| learning_rate | **2e-5** | 1e-5 |
| num_train_epochs | **10** | 5 |
| lr_scheduler_type | cosine | cosine |
| warmup_ratio | **0.1** | 0.1 |
| bf16 | true | true |
| save_steps | 66 (= 1 epoch at global batch 128) | 66 |

## Divergences from project_spec.md §8.2 (RECONCILE before final runs)

The spec says: *"Before finalizing, read Yuan et al.'s training script and match
their recipe where it differs."* Here is the diff:

| Knob | Spec §8.2 | Yuan repo | Note |
|---|---|---|---|
| Method | LoRA (pilot) / full | **full only** (ZeRO-3, 4×GPU) | Repo never uses LoRA. LoRA is our compute concession; expect to deviate from their result. |
| Epochs | 3 | **5–10** | Big difference — 3 may under-train. |
| Effective batch | 32 | **128** | |
| Warmup ratio | 0.03 | **0.1** | |
| Max seq len | 8192 | **4096** | Repo used Math-7B (4k ctx). We use Qwen2.5-7B (32k) so 8192 is safe. |
| LR | 1e-4 LoRA / 1e-5 full | 2e-5 (1.5b teacher) / 1e-5 (qwq) | |

`src/config.py:SFTConfig` currently holds the **spec** numbers, flagged inline.
Decision needed: how closely to match the repo given LoRA + single-GPU Modal.
NB: the same SFTConfig is shared by the W2SR (weak-teacher) and control
(strong-teacher) arms — only teacher strength may differ between them (spec 6.1).

## Teacher checkpoints (resolved)

Yuan's "Reasoner" teachers are the public `hkust-nlp/*-SimpleRL-Zoo` series
(e.g. `hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo`, Apache 2.0, on HF — confirmed
trivially loadable; referenced in `infer/generate.py` + `infer/infer.sh`). We do
NOT depend on them: the locked teacher axis uses the public
DeepSeek-R1-Distill-Qwen series (spec 5.2). The SimpleRL-Zoo teacher is
registered as `config.YUAN_SIMPLERL_1_5B` for an optional later robustness check.

## Data format

- Dataset is registered in LLaMA-Factory's `data/dataset_info.json` (NOT shipped
  in this clone — the actual trace `.json` files are absent). Names look like
  `math-hard-qwen2.5-1.5brl-qwen-base-template`.
- Trace generation: `external/w2sr/infer/generate.py` (vLLM), then
  `infer/split_true_false.py` splits correct/incorrect, `infer/judge_correct.py`
  grades. Prompt templates: `infer/utils/` + `eval/prompts/qwen-base-template/`.
- We must reproduce the LLaMA-Factory record format (instruction/output, or
  sharegpt) when writing our generated traces. Confirm against `dataset_info.json`
  schema before `train_student.py` is finalized.

## Eval (their math-accuracy eval, for the capability gate only)

`external/w2sr/eval/eval_passk.py`: vLLM, `max_tokens 32768`, `top_p 0.95`,
Pass@k. We reuse the answer-extraction logic in `eval/utils/grader.py` /
`parser.py` for the Pass@1 capability gate (spec §9.4), NOT for monitorability.
