#!/usr/bin/env python3
"""Verify OpenRouter returns FULL reasoning traces (thinking tokens) for
DeepSeek-R1-Distill, not just final answers.

Why this gates the design (user decision): R1-distill teacher *faithfulness* is
measured on the reasoning trace. If OpenRouter strips <think>...</think> content
and returns only the answer, we cannot score the teachers through OpenRouter and
must serve them on Modal vLLM instead (set config.TEACHER_VIA_OPENROUTER=False).

Run (needs an OpenRouter key):
    export OPENROUTER_API_KEY=sk-or-...
    python scripts/check_openrouter_thinking.py

Exit 0 + "THINKING TOKENS PRESENT" => teachers can go through OpenRouter.
Exit 1 + "NO THINKING TOKENS"      => fall back to Modal vLLM for teachers.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

MODEL = "deepseek/deepseek-r1-distill-qwen-1.5b"
PROMPT = "What is 17 * 24? Think step by step, then give the final answer."


def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("BLOCKED: set OPENROUTER_API_KEY to run this check.")
        return 2

    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.6,
        "max_tokens": 1024,
        # OpenRouter exposes reasoning via this flag for supported models:
        "reasoning": {"effort": "high"},
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)

    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    has_think_tags = "<think>" in content
    has_reasoning_field = bool(reasoning.strip())

    print(f"reasoning field chars: {len(reasoning)} | <think> in content: {has_think_tags}")
    print("--- content head ---")
    print(content[:400])
    if reasoning:
        print("--- reasoning head ---")
        print(reasoning[:400])

    if has_reasoning_field or has_think_tags:
        print("\nTHINKING TOKENS PRESENT -> teachers can run via OpenRouter.")
        return 0
    print("\nNO THINKING TOKENS -> set config.TEACHER_VIA_OPENROUTER=False; serve teachers on Modal.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
