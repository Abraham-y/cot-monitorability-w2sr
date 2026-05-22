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

from __future__ import annotations

import modal

# Heavy image for the real GPU jobs (generate / train / eval).
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "transformers", "trl", "peft", "accelerate", "vllm",
        "inspect-ai", "datasets", "numpy", "scipy", "statsmodels",
        "matplotlib", "pandas",
    )
)

# Minimal image for the smoke test — it only needs to print the GPU name, so
# don't pay the multi-minute build of the heavy image just to plumb-test Modal.
smoke_image = modal.Image.debian_slim()

app = modal.App("w2sr-monitorability", image=image)

# Persistent volume caches HF weights + stores traces/checkpoints/eval outputs,
# avoiding multi-GB cold-start re-downloads (spec 12).
volume = modal.Volume.from_name("w2sr-vol", create_if_missing=True)
VOL_MOUNT = "/vol"

hf_secret = modal.Secret.from_name("huggingface")  # create before first run


@app.function(gpu="T4", image=smoke_image)
def smoke() -> str:
    """Spec 15.1: confirm Modal works end to end by printing the GPU name."""
    import subprocess
    return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


@app.local_entrypoint()
def main():
    print("GPU:", smoke.remote())


# train/evaluate get bigger GPUs (e.g. gpu="A100", or "A100-80GB" for 14B).
# Wire these to src.* once those stages are implemented.
