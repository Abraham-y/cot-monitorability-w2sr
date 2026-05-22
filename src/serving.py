"""Pure (Modal-free) helpers for serving a model via vLLM's OpenAI-compatible
server, and for telling Inspect how to reach it.

Kept Modal-free so it is unit-testable locally (the GPU wiring lives in
modal_app.py, which imports these). Spec 12: vLLM for eval inference; the
STUDENT (baseline + trained checkpoints) is always served this way so the
headline baseline-vs-W2SR comparison uses one identical serving path.
"""

from __future__ import annotations

VLLM_PORT = 8000


def vllm_serve_command(
    model: str,
    *,
    port: int = VLLM_PORT,
    served_model_name: str | None = None,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
    dtype: str = "bfloat16",
    extra: list[str] | None = None,
) -> list[str]:
    """Build the `vllm serve` argv for an OpenAI-compatible endpoint.

    `model` is an HF id (e.g. "Qwen/Qwen2.5-7B-Instruct") or a local checkpoint
    path on the Modal Volume (e.g. "/vol/checkpoints/w2sr-weak-1.5b"). The
    served name is what clients/Inspect reference; default to the basename so a
    checkpoint path doesn't leak into the API model id.
    """
    served = served_model_name or model.rstrip("/").split("/")[-1]
    cmd = [
        "vllm", "serve", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--served-model-name", served,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
    ]
    if extra:
        cmd += extra
    return cmd


def inspect_model_string(served_model_name: str) -> str:
    """Inspect references an OpenAI-compatible server as `openai/<model>`; the
    base URL is supplied separately (env var) — see `inspect_env`."""
    return f"openai/{served_model_name}"


def inspect_env(base_url: str, api_key: str = "EMPTY") -> dict[str, str]:
    """Env vars that point Inspect's OpenAI provider at a custom vLLM endpoint.
    vLLM ignores the key but the OpenAI client requires one to be set."""
    return {
        "OPENAI_BASE_URL": base_url.rstrip("/") + "/v1",
        "OPENAI_API_KEY": api_key,
    }
