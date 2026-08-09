"""Stage 2: SFT a base student on a trace dataset (spec 7, 8.2).

Handles BOTH the W2SR (weak-teacher) and CONTROL (strong-teacher) datasets,
selected by which trace dir is passed; the SAME SFTConfig is used for both so
the only difference between conditions is teacher strength (spec 6.1).

Recipe fidelity (spec 13.4, references/recipe_w2sr.md): the Yuan repo trains via
LLaMA-Factory + DeepSpeed (full SFT). For our LoRA pilot on single-GPU Modal we
use TRL/peft following their DOCUMENTED hyperparameters (config.SFTConfig). This
is a deliberate compute concession — LLaMA-Factory is the higher-fidelity path
for a final full-SFT run. Heavy imports are lazy so the chat-formatting logic is
unit-testable locally without torch.
"""

from __future__ import annotations

import json
from pathlib import Path

from src import config

def to_chat_messages(row: dict) -> list[dict]:
    """One SFT example -> chat messages. `row` = {"content", "output"}.
    NO system prompt — must match generate_traces.build_prompt_messages (used at
    gate/eval time) so the student is trained and prompted identically. The
    trace `content` already carries the reason suffix; train on the CoT as the
    assistant turn.

    NOTE: do NOT render this with `apply_chat_template(...)` for SFT — see
    `build_sft_text` for why that silently deletes the CoT on R1-Distill.
    """
    return [
        {"role": "user", "content": row["content"]},
        {"role": "assistant", "content": row["output"]},
    ]


def build_sft_text(tok, row: dict) -> str:
    """Render one SFT example to training text, PRESERVING the chain of thought.

    Why this is not `apply_chat_template(to_chat_messages(row))`: the official
    DeepSeek-R1-Distill chat template contains, for assistant turns,

        {% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}

    i.e. it deletes everything up to and including `</think>`. Every R1-substrate
    trace carries `</think>` (the generation prompt ends `<|Assistant|><think>\\n`,
    so the completion is `...reasoning...</think>\\n\\nanswer`). Rendering the
    assistant turn through the template therefore trains the student on the final
    answer ONLY, with the reasoning stripped — producing exactly the terse,
    `</think>`-free student that answer-only supervision predicts.

    HISTORY NOTE: every R1-substrate checkpoint reported in the paper was
    trained BEFORE this fix, i.e. on the template-stripped, answer-only
    supervision; the paper discloses this and defines the measured intervention
    accordingly (writeup_workshop.tex, "What the supervision actually
    contained"). Re-running this (fixed) code trains on CoT-preserving
    supervision and is therefore a DIFFERENT intervention from the one the
    paper's numbers describe — it will not reproduce the paper's arms.

    Instead we render only the USER turn with `add_generation_prompt=True` — which
    emits the model's own assistant prefix verbatim (`<|Assistant|><think>\\n` on
    R1-Distill, `<|im_start|>assistant\\n` on ChatML) — and append the raw trace
    text plus EOS. This is byte-identical to what the model sees at eval time,
    so train and inference formats match, and the CoT survives.

    `assert_cot_preserved` below hard-fails if a future template change ever
    strips the reasoning again.
    """
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": row["content"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    eos = tok.eos_token or ""
    return f"{prefix}{row['output']}{eos}"


def assert_cot_preserved(texts: list[str], rows: list[dict], n_check: int = 32) -> None:
    """Hard-fail if the rendered SFT text lost the reasoning span.

    Guards against the `apply_chat_template` CoT-stripping failure mode: a
    silent data bug that still trains and still lowers loss, so nothing else
    catches it. Checks a prefix sample of rows.
    """
    for text, row in list(zip(texts, rows))[:n_check]:
        out = row["output"]
        if "</think>" not in out:
            continue                      # nothing to strip on this row
        reasoning = out.split("</think>")[0]
        probe = reasoning.strip()[:200]   # head of the CoT, before any strip point
        if probe and probe not in text:
            raise AssertionError(
                "SFT text lost the chain-of-thought: the reasoning span before "
                "`</think>` is absent from the rendered training example. This is "
                "the apply_chat_template stripping bug — see build_sft_text(). "
                f"\n  rendered head: {text[:300]!r}"
                f"\n  expected CoT head: {probe[:120]!r}")


def load_train_rows(train_json: Path) -> list[dict]:
    return json.loads(Path(train_json).read_text())


def train_student_lora(
    base_student: str,
    train_json: Path,
    out_dir: Path,
    cfg: config.SFTConfig | None = None,
) -> str:
    """LoRA SFT the student on the trace dataset; return the adapter checkpoint
    path. Runs on a GPU (Modal). Logs loss curve + records the data hash next to
    the checkpoint (spec 16). After training, call validate_training.validate()
    BEFORE trusting any monitorability number (spec 9).
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    cfg = cfg or config.SFTConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_student)
    rows = load_train_rows(train_json)
    texts = [build_sft_text(tok, r) for r in rows]
    # Fail loudly if the CoT did not survive rendering (see build_sft_text).
    assert_cot_preserved(texts, rows)
    # The chat template already emits BOS; tokenizing the rendered string with
    # add_bos_token=True would prepend a second one.
    if getattr(tok, "add_bos_token", False) and tok.bos_token and texts \
            and texts[0].startswith(tok.bos_token):
        tok.add_bos_token = False
    # With CoT-preserving rendering, an 8k-budget trace plus prompt can exceed
    # max_seq_len; TRL then truncates the TAIL, silently deleting the final
    # answer and EOS. Fail loudly instead of training on decapitated examples.
    n_over = sum(
        1 for t in texts
        if len(tok(t, add_special_tokens=False).input_ids) > cfg.max_seq_len
    )
    if n_over:
        raise AssertionError(
            f"{n_over}/{len(texts)} rendered SFT examples exceed max_seq_len="
            f"{cfg.max_seq_len} tokens and would be tail-truncated (losing the "
            "final answer + EOS). Raise cfg.max_seq_len or filter these rows.")
    ds = Dataset.from_list([{"text": t} for t in texts])

    model = AutoModelForCausalLM.from_pretrained(
        base_student, torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    )
    lora = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    # effective batch = per_device * grad_accum (single device on Modal)
    per_device = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device)
    targs = TRLSFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        max_length=cfg.max_seq_len,
        bf16=cfg.bf16,
        logging_steps=1,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=targs, train_dataset=ds, peft_config=lora)
    trainer.train()
    trainer.save_model(str(out_dir))

    # provenance next to the checkpoint (spec 16): which data produced this model
    manifest = Path(train_json).parent / "manifest.json"
    (out_dir / "train_provenance.json").write_text(json.dumps({
        "base_student": base_student,
        "train_json": str(train_json),
        "data_manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
        "sft_config": cfg.__dict__,
        "loss_log": trainer.state.log_history,
    }, indent=2, default=str))
    return str(out_dir)
