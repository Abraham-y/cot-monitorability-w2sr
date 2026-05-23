#!/usr/bin/env python3
"""Idempotent compatibility patch for the cloned Meek eval (external/, gitignored).

Why: the judge scorers always send BOTH temperature and top_p. Anthropic's
native API rejects that combo ("temperature and top_p cannot both be specified"),
and their validate_model_config crashes on top_p=None — so we can't simply null
it via config. This patch makes the cue_aware + factor_utilization scorers OMIT
top_p for `anthropic/*` judge models (keeping temperature=0 for determinism),
which lets us bill the judge to the Anthropic API key instead of OpenRouter.

Run after cloning/updating external/monitorability-eval:
    python scripts/patch_meek_eval.py
Re-running is safe (idempotent).
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCORERS = REPO / "external/monitorability-eval/src/measuring_cot_monitorability/scorers"
MARKER = "# w2sr-patch: anthropic temp/top_p"

OLD_BUILD = '''        # Build GenerateConfig from judge config
        generate_config_params = {
            "temperature": judge_config["temperature"],
            "top_p": judge_config["top_p"]
        }'''
NEW_BUILD = '''        # Build GenerateConfig from judge config
        # w2sr-patch: anthropic temp/top_p — Anthropic native API rejects
        # temperature+top_p together, so omit top_p for anthropic/* judges.
        generate_config_params = {"temperature": judge_config["temperature"]}
        if not model.startswith("anthropic/") and judge_config.get("top_p") is not None:
            generate_config_params["top_p"] = judge_config["top_p"]'''

OLD_ELSE = '''        # Use default deterministic settings
        judge_model = get_model(model, config=GenerateConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=1
        ))'''
NEW_ELSE = '''        # Use default deterministic settings (w2sr-patch: anthropic temp/top_p)
        if model.startswith("anthropic/"):
            judge_model = get_model(model, config=GenerateConfig(temperature=0.0))
        else:
            judge_model = get_model(model, config=GenerateConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=1
            ))'''


def patch(path: Path) -> str:
    t = path.read_text()
    if MARKER in t:
        return "already patched"
    if OLD_BUILD not in t or OLD_ELSE not in t:
        raise SystemExit(f"expected blocks not found in {path} — upstream changed?")
    path.write_text(t.replace(OLD_BUILD, NEW_BUILD).replace(OLD_ELSE, NEW_ELSE))
    return "patched"


# --- run_eval.py: expose max_connections via env (W2SR_MAX_CONNECTIONS) ------
# Inspect's eval() doesn't take max_connections and the eval hardcodes no
# concurrency, so OpenRouter/vLLM calls run at a low default — painfully slow
# for long-CoT reasoning models (the 32B teacher). Inject it into the
# generator GenerateConfig so we can crank parallelism per run.
RUN_EVAL = REPO / "external/monitorability-eval/scripts/evals/run_eval.py"
MARKER2 = "# w2sr-patch: max_connections"
OLD_GC = "        generation_config = GenerateConfig(**generate_config_kwargs)"
NEW_GC = '''        # w2sr-patch: max_connections — crank parallelism via env
        import os as _os
        _mc = _os.environ.get("W2SR_MAX_CONNECTIONS")
        if _mc:
            generate_config_kwargs["max_connections"] = int(_mc)
        generation_config = GenerateConfig(**generate_config_kwargs)'''


def patch_run_eval(path: Path) -> str:
    t = path.read_text()
    if MARKER2 in t:
        return "already patched"
    if OLD_GC not in t:
        raise SystemExit(f"expected GenerateConfig line not found in {path}")
    path.write_text(t.replace(OLD_GC, NEW_GC, 1))
    return "patched"


if __name__ == "__main__":
    for name in ("cue_aware_adaptive.py", "factor_utilization.py"):
        print(f"{name}: {patch(SCORERS / name)}")
    print(f"run_eval.py: {patch_run_eval(RUN_EVAL)}")
