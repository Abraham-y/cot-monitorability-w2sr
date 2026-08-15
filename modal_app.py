"""Modal app: image, volume, secrets, decorated GPU entrypoints (spec 12).

Workload = three discrete GPU jobs (generate / train / eval), not a service,
so per-second serverless GPU billing fits.

Dev loop (spec 12): write + debug all logic locally with config.TINY_MODEL on
CPU/MPS so a full pass takes seconds, then flip `gpu=` for real runs.

    modal run modal_app.py::smoke          # spec 15.1 plumbing test
    modal run modal_app.py::generate ...
    modal run modal_app.py::train ...
    modal run modal_app.py::evaluate ...

PREREQ (blocked on user): a Modal account, and the HF token stored as a Modal
Secret named "huggingface" (NEVER inline tokens -- spec 12).
"""

import modal

from src import serving

VOL_MOUNT = "/vol"
HF_HOME = f"{VOL_MOUNT}/hf_cache"

# Heavy image for the real GPU jobs (generate / train / eval).
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "transformers", "trl", "peft", "accelerate", "vllm",
        "inspect-ai", "datasets", "numpy", "scipy", "statsmodels",
        "matplotlib", "pandas",
        "sympy",   # src/grading.py math grader (self-contained, no latex2sympy2)
    )
    .env({  # vLLM is used here too (gen_traces, gate) — same flashinfer/HF env
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": HF_HOME,
        "VLLM_USE_V1": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",   # debian_slim has no nvcc
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    })
    .add_local_python_source("src")
)

# Dedicated image for vLLM serving (lighter than the full training image).
vllm_image = (
    modal.Image.debian_slim()
    .pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # fast weight downloads
        "HF_HOME": HF_HOME,                # cache weights on the Volume
        "VLLM_USE_V1": "1",
        # debian_slim has no CUDA toolkit (nvcc); flashinfer's sampler/attention
        # JIT would fail at startup. Disable flashinfer JIT paths — eval is
        # greedy (temp 0) so we don't need the flashinfer fast sampler.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    })
    .add_local_python_source("src")
)

# Minimal image for the smoke test — it only needs to print the GPU name, so
# don't pay the multi-minute build of the heavy image just to plumb-test Modal.
smoke_image = modal.Image.debian_slim()

app = modal.App("w2sr-monitorability", image=image)

# Persistent volume caches HF weights + stores traces/checkpoints/eval outputs,
# avoiding multi-GB cold-start re-downloads (spec 12).
volume = modal.Volume.from_name("w2sr-vol", create_if_missing=True)

# Lazy reference — resolved only when a function that USES it runs (gated model
# downloads in the future train/generate jobs). Serving Qwen Instruct + our own
# checkpoints needs no HF token, so VLLMServer below omits it; create the secret
# (modal secret create huggingface HF_TOKEN=hf_...) before the training jobs.
hf_secret = modal.Secret.from_name("huggingface")


@app.function(gpu="T4", image=smoke_image)
def smoke() -> str:
    """Spec 15.1: confirm Modal works end to end by printing the GPU name."""
    import subprocess
    return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


@app.local_entrypoint()
def main():
    print("GPU:", smoke.remote())


# --------------------------------------------------------------------------
# vLLM serving — the STUDENT (untrained baseline + trained checkpoints) is
# always served this way, so the headline baseline-vs-W2SR comparison goes
# through one identical serving path (spec 8.3, user decision). Inspect hits
# the OpenAI-compatible endpoint; see src/serving.inspect_env / inspect_model_string.
#
# Parametrized by `model` (HF id like "Qwen/Qwen2.5-7B-Instruct", or a
# checkpoint path on the Volume like "/vol/checkpoints/w2sr-weak-1.5b").
# 7B fits on one A100-40GB; override for other sizes at lookup time, e.g.
#   VLLMServer.with_options(gpu="L4")(model="Qwen/Qwen2.5-1.5B-Instruct")
#
# Usage (once the `huggingface` secret exists):
#   modal deploy modal_app.py
#   # then get the endpoint URL for a given model:
#   modal run modal_app.py::serve_url --model Qwen/Qwen2.5-7B-Instruct
MINUTES = 60


@app.cls(
    image=vllm_image,
    gpu="A100-80GB",  # A 7B reasoning student (R1-distill-7B) at 32k ctx with 32-way
    # concurrency needs the KV headroom, and the paper's R1-7B evals ran on 80GB.
    # For a small instruct 7B at 8k, "A100" (40GB) suffices; for the 32B teacher
    # use gpu="A100-80GB:2" + tensor_parallel=2.
    volumes={VOL_MOUNT: volume},
    secrets=[hf_secret],
    timeout=60 * MINUTES,
    scaledown_window=30 * MINUTES,  # bumped from 10→30 so the server doesn't recycle mid-eval
)
@modal.concurrent(max_inputs=32)
class VLLMServer:
    # Deployed web endpoint serves whatever this default is (flip + redeploy per
    # model). Currently: baseline reasoning student R1-distill-7B (cond-1', 32k ctx
    # for long <think> CoT). For the W2SR student use /vol/merged/w2sr_r1_7b.
    model: str = modal.parameter(default="/vol/merged/w2sr_r1_7b")
    max_model_len: int = modal.parameter(default=32768)
    tensor_parallel: int = modal.parameter(default=1)

    @modal.web_server(port=serving.VLLM_PORT, startup_timeout=20 * MINUTES)
    def serve(self):
        import subprocess
        extra = []
        if self.tensor_parallel > 1:
            extra += ["--tensor-parallel-size", str(self.tensor_parallel)]
            # skip cudagraph compile on multi-GPU TP — it adds ~5min to cold
            # start and hangs the shm broadcast, so the eval warmup never latches
            # before the 10-min idle scaledown recycles the container.
            extra += ["--enforce-eager"]
        cmd = serving.vllm_serve_command(
            self.model, max_model_len=self.max_model_len, extra=extra or None,
        )
        subprocess.Popen(cmd)


def _gen_traces_impl(
    teacher_model: str, out_dir: str,
    n_problems: int, n_per_problem: int, keep_incorrect: bool,
    temperature: float, top_p: float, max_tokens: int,
    teacher_system: str, levels: str,
    teacher_max_model_len: int, tensor_parallel: int,
    enforce_eager: bool = False,
) -> dict:
    """Shared Stage-1 body (offline batched vLLM, like Yuan's generate.py): load
    MATH (levels 3-5), sample CoT from `teacher_model`, grade with the W2SR
    grader, write train.json + manifest + held_out.json to `out_dir` on the
    volume. Same problems/seed across W2SR and control to match sets (spec 6.1).
    `tensor_parallel`>1 shards a big teacher (e.g. 72B) across multiple GPUs.
    Called by gen_traces (1×A100) and gen_traces_big (2×A100-80GB)."""
    import json as _json
    from pathlib import Path
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from src import problems, generate_traces as gt

    lv = tuple(int(x) for x in levels.split(","))
    train, held_out = problems.load_math_problems(n_train=n_problems, n_held_out=200, levels=lv)
    tok = AutoTokenizer.from_pretrained(teacher_model)
    # teacher_system: SimpleRL-Zoo (Qwen) wants Yuan's qwen-base system prompt;
    # R1-distill wants none (its template forces <think>). The student's ChatML
    # injects the same default system, so this stays consistent.
    def _msgs(problem):
        sys_msg = [{"role": "system", "content": teacher_system}] if teacher_system else []
        return sys_msg + gt.build_prompt_messages(problem)
    prompts = [
        tok.apply_chat_template(_msgs(p["problem"]), tokenize=False, add_generation_prompt=True)
        for p in train
    ]
    # teacher_max_model_len: lower to 4096 for 4k-context teachers
    # (e.g. Qwen2.5-Math-* series; 8192 exceeds their max_position_embeddings).
    # enforce_eager: skip cudagraph capture — slower but robust on multi-GPU TP
    # (the 72B 4-GPU run died mid-generation in the cudagraph shm broadcast).
    llm = LLM(model=teacher_model, max_model_len=teacher_max_model_len,
              gpu_memory_utilization=0.9, trust_remote_code=True,
              tensor_parallel_size=tensor_parallel, enforce_eager=enforce_eager)
    sp = SamplingParams(temperature=temperature, top_p=top_p,
                        max_tokens=max_tokens, n=n_per_problem,
                        repetition_penalty=1.1)   # suppress 1.5B repetition spirals
    outputs = llm.generate(prompts, sp)

    grade = gt.default_grader()
    raw = [
        gt.TraceRecord(p["problem"], str(p["gt_answer"]), o.text, grade(o.text, str(p["gt_answer"])))
        for p, out in zip(train, outputs) for o in out.outputs
    ]
    # keep correct+incorrect (Yuan) but DROP degenerate/looping traces — they
    # teach the student to loop (caused the first run's format collapse).
    n_degenerate = sum(gt.is_degenerate(t.response) for t in raw)
    kept = [t for t in raw if (t.is_correct or keep_incorrect) and not gt.is_degenerate(t.response)]
    rows = gt.to_llama_factory_records(kept)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "train.json").write_text(_json.dumps(rows, ensure_ascii=False, indent=2))
    (out / "held_out.json").write_text(_json.dumps(held_out, indent=2))
    manifest = {
        "teacher_model": teacher_model, "n_problems": len(train),
        "n_traces_total": len(raw), "n_correct": sum(t.is_correct for t in raw),
        "n_kept": len(kept), "keep_incorrect": keep_incorrect,
        "n_degenerate_dropped": n_degenerate,
        "data_hash": gt.dataset_hash(rows),
        "total_output_chars": sum(len(t.response) for t in kept),
    }
    (out / "manifest.json").write_text(_json.dumps(manifest, indent=2))
    volume.commit()
    return manifest


@app.function(
    image=image, gpu="A100", volumes={VOL_MOUNT: volume},
    secrets=[hf_secret], timeout=6 * 60 * 60,
)
def gen_traces(
    teacher_model: str, out_dir: str,
    n_problems: int = 1000, n_per_problem: int = 1, keep_incorrect: bool = True,
    temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 4096,
    teacher_system: str = "", levels: str = "3,4,5",
    teacher_max_model_len: int = 8192,
) -> dict:
    """Stage 1 on a single A100 (teachers up to ~14B). See _gen_traces_impl."""
    return _gen_traces_impl(
        teacher_model, out_dir, n_problems, n_per_problem, keep_incorrect,
        temperature, top_p, max_tokens, teacher_system, levels,
        teacher_max_model_len, tensor_parallel=1)


@app.function(
    image=image, gpu="A100-80GB:4", volumes={VOL_MOUNT: volume},
    secrets=[hf_secret], timeout=6 * 60 * 60,
)
def gen_traces_big(
    teacher_model: str, out_dir: str,
    n_problems: int = 1000, n_per_problem: int = 1, keep_incorrect: bool = True,
    temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 4096,
    teacher_system: str = "", levels: str = "3,4,5",
    teacher_max_model_len: int = 8192,
) -> dict:
    """Stage 1 for a big teacher (e.g. Qwen2.5-Math-72B-Instruct) sharded across
    4×A100-80GB (tensor_parallel=4). 72B bf16 (~144GB) nearly fills 2×80GB,
    leaving no KV-cache headroom (engine init failed); 4×80GB gives ~150GB free
    for KV. Used for the strong end of the teacher axis."""
    return _gen_traces_impl(
        teacher_model, out_dir, n_problems, n_per_problem, keep_incorrect,
        temperature, top_p, max_tokens, teacher_system, levels,
        teacher_max_model_len, tensor_parallel=4, enforce_eager=True)


@app.function(
    image=image,            # heavy image (torch/trl/peft)
    gpu="A100",
    volumes={VOL_MOUNT: volume},
    timeout=16 * 60 * 60,   # keep runs < 16h (spec 15)
)
def train(base_student: str, train_json: str, out_dir: str, max_seq_len: int = 0) -> str:
    """Stage 2 on GPU: LoRA SFT the student on a trace dataset (W2SR or control).
    `train_json` and `out_dir` are paths ON the volume (e.g. /vol/traces/w2sr/
    train.json, /vol/checkpoints/w2sr). Returns the checkpoint path.

    Get traces onto the volume first (run stage 1 locally against the weak-teacher
    endpoint, then `modal volume put w2sr-vol <local> /vol/traces/<cond>`)."""
    import dataclasses
    from pathlib import Path
    from src import config
    from src.train_student import train_student_lora
    cfg = config.SFTConfig()
    if max_seq_len:  # e.g. 4096 for Qwen2.5-Math-7B (4k context)
        cfg = dataclasses.replace(cfg, max_seq_len=max_seq_len)
    ckpt = train_student_lora(base_student, Path(train_json), Path(out_dir), cfg)
    volume.commit()  # persist checkpoint to the volume
    return ckpt


@app.function(
    image=image, gpu="A100", volumes={VOL_MOUNT: volume},
    secrets=[hf_secret], timeout=6 * 60 * 60,
)
def gate(base_student: str, adapter_dir: str, held_out_json: str, is_control: bool = False,
         rep_penalty: float = 1.0, max_tokens: int = 4096, max_model_len: int = 8192,
         headroom_probe: bool = False) -> dict:
    """Run the spec 9 gate = the W2SR reproduction check. One vLLM load serves
    BOTH the untrained baseline (no adapter) and the trained student (LoRA
    adapter) on the SAME held-out MATH, so we get the capability GAIN directly.
    Greedy decoding (temp 0). Writes gate_report.json next to the adapter.

    headroom_probe: also score the UNTRAINED base with the CoT prompt (zero-shot
    CoT, no LoRA). Lets us separate genuine W2SR elicitation from "just prompting
    it to think": if cot_prompted_baseline ~= W2SR, the gain is prompt-induced,
    not weak-supervision-induced; if unelicited << cot_prompted < W2SR, the model
    was genuinely under-elicited and training added real capability (PREREG §5)."""
    import json as _json
    from pathlib import Path
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    from src import generate_traces as gt, validate_training as vt, config

    held = _json.loads(Path(held_out_json).read_text())
    gts = [str(p["gt_answer"]) for p in held]
    tok = AutoTokenizer.from_pretrained(base_student)
    # REPRODUCTION baseline = UNELICITED (no-CoT) prompt on the untrained base,
    # so W2SR's reasoning-elicitation gain isn't cancelled (Yuan's mechanism).
    # The trained W2SR student is prompted in its CoT training format.
    base_prompts = [tok.apply_chat_template(gt.build_direct_prompt(p["problem"]),
                                            tokenize=False, add_generation_prompt=True) for p in held]
    trained_prompts = [tok.apply_chat_template(gt.build_prompt_messages(p["problem"]),
                                               tokenize=False, add_generation_prompt=True) for p in held]
    # greedy + no penalty is the worst case for loops; rep_penalty curbs the
    # trained model's eval-time repetition/runaway (same decoding for both arms).
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, repetition_penalty=rep_penalty)

    cfg = config.SFTConfig()
    llm = LLM(model=base_student, enable_lora=True, max_lora_rank=cfg.lora_rank,
              max_model_len=max_model_len, gpu_memory_utilization=0.9)
    base_out = llm.generate(base_prompts, sp)                              # untrained, unelicited
    lora = LoRARequest("w2sr", 1, adapter_dir)
    trained_out = llm.generate(trained_prompts, sp, lora_request=lora)     # trained student (CoT)

    grade = gt.default_grader()
    base_resp = [o.outputs[0].text for o in base_out]
    trained_resp = [o.outputs[0].text for o in trained_out]
    baseline_pass1 = vt.pass1(base_resp, gts, grade)

    prov = Path(adapter_dir) / "train_provenance.json"
    loss_log = _json.loads(prov.read_text()).get("loss_log", []) if prov.exists() else []
    report = vt.validate(loss_log, trained_resp, gts, baseline_pass1, grade,
                         is_control=is_control)
    out = {**report.__dict__, "pass1_gain": report.pass1_gain}
    if headroom_probe:
        # untrained base, CoT prompt, no LoRA = the zero-shot-CoT ceiling.
        cot_out = llm.generate(trained_prompts, sp)
        cot_pass1 = vt.pass1([o.outputs[0].text for o in cot_out], gts, grade)
        out["cot_prompted_baseline_pass1"] = cot_pass1
        out["headroom_unelicited_to_cot"] = cot_pass1 - baseline_pass1
        out["w2sr_beyond_cot_prompt"] = report.pass1 - cot_pass1
    (Path(adapter_dir) / "gate_report.json").write_text(_json.dumps(out, indent=2, default=str))
    volume.commit()
    return out


@app.function(
    image=image, gpu="A100", volumes={VOL_MOUNT: volume},
    secrets=[hf_secret], timeout=60 * 60,
)
def merge_adapter(base_student: str, adapter_dir: str, out_dir: str) -> str:
    """Merge a LoRA adapter into its base and save a full model to the volume,
    so the monitorability eval can serve it via the standard vllm_serve_command
    path (conds 2 & 3 = W2SR / control STUDENTS). The baseline (cond 1) and
    teacher refs (cond 4) are full HF models served directly; only the trained
    students need merging. Identical serving path keeps the comparison clean."""
    from pathlib import Path
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        base_student, torch_dtype=torch.bfloat16, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(base_student).save_pretrained(out)
    volume.commit()
    return str(out)


@app.local_entrypoint()
def serve_url(model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Print the OpenAI-compatible endpoint URL + the Inspect env/model string
    for a given served model (does not start a container)."""
    server = VLLMServer(model=model)
    url = server.serve.get_web_url()
    served = model.rstrip("/").split("/")[-1]
    print("endpoint:", url)
    print("inspect model:", serving.inspect_model_string(served))
    print("inspect env:", serving.inspect_env(url))


# train/evaluate get bigger GPUs (e.g. gpu="A100", or "A100-80GB" for 14B).
# Wire these to src.* once those stages are implemented.
