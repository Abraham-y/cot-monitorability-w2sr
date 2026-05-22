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
    )
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
    gpu="A100",  # 40GB: comfortable for the 7B student. (L4 also fits 7B but
    # is slower; pilot 1.5B used L4. Drop back to L4 for cheap small-model runs.)
    volumes={VOL_MOUNT: volume},
    # No secret needed: Qwen Instruct + our checkpoints are ungated. Add
    # secrets=[hf_secret] here only if serving a gated model later.
    timeout=60 * MINUTES,
    scaledown_window=10 * MINUTES,  # spin down after 10 min idle (cost control)
)
@modal.concurrent(max_inputs=32)
class VLLMServer:
    # Default = the real 7B student. (Pilot validation used 1.5B on L4.)
    # The deployed web endpoint serves whatever this default is.
    model: str = modal.parameter(default="Qwen/Qwen2.5-7B-Instruct")
    max_model_len: int = modal.parameter(default=8192)

    @modal.web_server(port=serving.VLLM_PORT, startup_timeout=20 * MINUTES)
    def serve(self):
        import subprocess
        cmd = serving.vllm_serve_command(
            self.model, max_model_len=self.max_model_len,
        )
        subprocess.Popen(cmd)


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
