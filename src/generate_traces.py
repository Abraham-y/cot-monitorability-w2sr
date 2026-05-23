"""Stage 1: generate CoT traces from a teacher model (spec 7, 8.1).

Ground truth = the W2SR repo (spec 13.4): we reproduce its prompt template,
record shape, grading, and the LLaMA-Factory SFT format rather than improvising.
  - prompt: qwen-base-template (external/w2sr/infer/generate.py)
  - grading: extract_answer + math_equal (external/w2sr/infer/utils/{parser,grader})
  - LF SFT format: [{"content": <prompt>, "output": <response>}]
    (external/gair-w2s-reasoning/data/llama_factory/dataset_info.json: math_2)

W2SR keeps BOTH correct and incorrect traces for the main run (incorrect still
help); correct-only is the W2SR-P ablation. The W2SR (weak teacher) and CONTROL
(strong teacher) datasets must match on problem set, trace count, and token
volume (spec 6.1) — `generate_traces` is called once per TeacherSpec with the
same `problems`.

The actual vLLM generation runs on Modal (modal_app.py); the heavy step is
injected as `sample_fn` so this module is unit-testable locally with a mock.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

# Matches external/w2sr/infer/generate.py::make_conversation_base (qwen-base).
QWEN_BASE_SYSTEM = "You are a helpful assistant."
REASON_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt_messages(problem: str) -> list[dict]:
    """qwen-base-template chat messages (W2SR recipe)."""
    return [
        {"role": "system", "content": QWEN_BASE_SYSTEM},
        {"role": "user", "content": f"{problem}\n{REASON_SUFFIX}"},
    ]


@dataclass
class TraceRecord:
    question: str
    gt_answer: str
    response: str          # the teacher's full CoT trace
    is_correct: bool


def default_grader() -> Callable[[str, str], bool]:
    """The self-contained sympy-based MATH grader (src/grading.py). Replaces the
    W2SR grader, whose latex2sympy2 dep is broken on our stack — see grading.py
    for the rationale (only used for the Pass@1 gate + trace is_correct)."""
    from src.grading import grade
    return grade


def endpoint_sample_fn(
    base_url: str,
    served_model: str,
    *,
    temperature: float = 0.6,   # DeepSeek R1-distill recommended (spec 8.1)
    top_p: float = 0.95,
    max_tokens: int = 4096,
    n: int = 1,
    api_key: str = "EMPTY",
) -> Callable[[str, list[dict]], list[str]]:
    """A `sample_fn` that calls a Modal-served vLLM OpenAI-compatible endpoint
    (the SAME endpoint used to serve the weak teacher for eval — reuse, no new
    GPU function). Returns n CoT completions per prompt."""
    import json as _json
    import ssl
    import urllib.request

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    url = base_url.rstrip("/") + "/chat/completions"

    def sample(_model: str, messages: list[dict]) -> list[str]:
        body = _json.dumps({
            "model": served_model, "messages": messages, "n": n,
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600, context=ctx) as resp:
            data = _json.load(resp)
        return [c["message"]["content"] for c in data["choices"]]

    return sample


def to_llama_factory_records(traces: Iterable[TraceRecord]) -> list[dict]:
    """SFT training file rows: {"content": <prompt>, "output": <teacher CoT>}.
    Content is the flattened qwen-base user turn (the template is applied by
    LLaMA-Factory's `qwen` template at train time)."""
    rows = []
    for t in traces:
        rows.append({
            "content": f"{t.question}\n{REASON_SUFFIX}",
            "output": t.response,
        })
    return rows


def dataset_hash(rows: list[dict]) -> str:
    """Stable sha256 of the training rows (spec 16: hash every trace dataset)."""
    blob = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def generate_traces(
    teacher_model: str,
    problems: list[dict],                 # [{"problem": str, "gt_answer": str}, ...]
    sample_fn: Callable[[str, list[dict]], list[str]],  # (teacher, messages) -> n responses
    out_dir: Path,
    *,
    keep_incorrect: bool = True,
    grade_fn: Callable[[str, str], bool] | None = None,
    n_per_problem: int = 1,
) -> dict:
    """Sample CoT trajectories, grade them, and write:
      - traces_raw.jsonl   : full records (TraceRecord-shaped) for inspection
      - train.json         : LLaMA-Factory SFT rows for the KEPT traces
      - manifest.json      : teacher, counts, data hash, seed provenance
    Returns the manifest dict.

    `sample_fn(teacher_model, messages) -> [response, ...]` is the only heavy
    step (vLLM on Modal); inject a mock for local tests. `keep_incorrect=False`
    yields the W2SR-P (correct-only) ablation dataset.
    """
    grade_fn = grade_fn or default_grader()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw: list[TraceRecord] = []
    for ex in problems:
        problem, gt = ex["problem"], str(ex["gt_answer"])
        for resp in sample_fn(teacher_model, build_prompt_messages(problem))[:n_per_problem]:
            raw.append(TraceRecord(problem, gt, resp, grade_fn(resp, gt)))

    kept = [t for t in raw if (t.is_correct or keep_incorrect)]
    rows = to_llama_factory_records(kept)
    h = dataset_hash(rows)

    with (out_dir / "traces_raw.jsonl").open("w", encoding="utf-8") as f:
        for t in raw:
            f.write(json.dumps(t.__dict__, ensure_ascii=False) + "\n")
    (out_dir / "train.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    manifest = {
        "teacher_model": teacher_model,
        "n_problems": len(problems),
        "n_traces_total": len(raw),
        "n_correct": sum(t.is_correct for t in raw),
        "n_kept": len(kept),
        "keep_incorrect": keep_incorrect,
        "n_per_problem": n_per_problem,
        "data_hash": h,
        "total_output_chars": sum(len(t.response) for t in kept),  # volume proxy (spec 6.1)
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
