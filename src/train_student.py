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

# qwen-base template turns (must match generate_traces: the trace `content`
# already carries the reason suffix; we wrap it as the user turn and train on
# the teacher CoT as the assistant turn).
SYSTEM = "You are a helpful assistant."


def to_chat_messages(row: dict) -> list[dict]:
    """One SFT example -> chat messages. `row` = {"content", "output"} from
    generate_traces.to_llama_factory_records."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["content"]},
        {"role": "assistant", "content": row["output"]},
    ]


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
    ds = Dataset.from_list([
        {"text": tok.apply_chat_template(to_chat_messages(r), tokenize=False)}
        for r in rows
    ])

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
