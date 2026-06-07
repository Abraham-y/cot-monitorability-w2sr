#!/usr/bin/env bash
# Cross-family CAPABILITY GATE — runs ONLY the baseline (uncued) GPQA pass for a
# Llama condition. Args: <SERVED_MODEL> <CONFIG> <BATCH_NAME> <RESULTS_DIR>
# v2 fixes (re-run after 50% empty rate on the first attempt):
#   - LOWER concurrency: 8 parallel (was 24) — Modal VLLMServer max_inputs=32 but
#     cold-start saturation appears to have killed the first ~20 requests.
#   - REAL warmup probe: send a small completion, retry until a non-empty
#     response comes back. Only after that does the batch start, so the server
#     is provably warm.
#   - Higher max_tokens via the YAML (16000 not 8000) so long R1-Llama traces
#     aren't truncated to empty.
set -e
cd "/Users/abrahamyeung/CS 338 Project"
set -a && . ./.env && set +a
export OPENAI_BASE_URL="https://ayeung16--w2sr-monitorability-vllmserver-serve.modal.run/v1"
export OPENAI_API_KEY="EMPTY"
export W2SR_LIMIT=40
export W2SR_MAX_SAMPLES=4
export W2SR_MAX_CONNECTIONS=4
export W2SR_JUDGE_MAX_CONNECTIONS=24

SERVED="${1:?served model}"
CFG="${2:?config yaml}"
BN="${3:?batch name}"
RESULTS="${4:?results dir}"

echo "===== DEPLOY (expects VLLMServer default == $SERVED) ====="
/opt/miniconda3/bin/modal deploy modal_app.py 2>&1 | tail -3

echo "===== WARM UP — model-list AND generation probe ====="
python3 - "$SERVED" <<'PY'
import json, ssl, sys, urllib.request, urllib.error, certifi, time
served = sys.argv[1]
ctx = ssl.create_default_context(cafile=certifi.where())
endpoint = "https://ayeung16--w2sr-monitorability-vllmserver-serve.modal.run/v1"

# Phase 1: wait for the served model to show up in /v1/models
for i in range(80):
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(endpoint + "/models"),
                                              timeout=30, context=ctx))
        ids = [m["id"] for m in d["data"]]
        print(f"[{i}] models endpoint: {ids}", flush=True)
        if served in ids:
            print(f"  MODEL LISTED: {served}", flush=True); break
        time.sleep(12)
    except Exception as e:
        print(f"[{i}] /models error: {str(e)[:80]}", flush=True); time.sleep(12)
else:
    raise SystemExit(f"endpoint never listed {served}")

# Phase 2: send a real completion probe, require a NON-EMPTY response, otherwise
# loop. This proves vLLM has finished cold-start and can actually generate.
probe = {
    "model": served,
    "messages": [{"role": "user", "content": "Reply with exactly the word OK and nothing else."}],
    "temperature": 0.0, "max_tokens": 32,
}
req = urllib.request.Request(
    endpoint + "/chat/completions",
    data=json.dumps(probe).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
)
for i in range(30):
    try:
        r = urllib.request.urlopen(req, timeout=600, context=ctx)
        d = json.loads(r.read())
        content = d.get("choices", [{}])[0].get("message", {}).get("content") or ""
        print(f"[probe {i}] non-empty? {bool(content.strip())}  content={content[:60]!r}", flush=True)
        if content.strip():
            print(f"  WARMED: server returned non-empty content. Starting batch.", flush=True)
            break
        time.sleep(8)
    except urllib.error.HTTPError as e:
        body = e.read()[:200] if hasattr(e, "read") else b""
        print(f"[probe {i}] HTTP {e.code}: {body!r}", flush=True); time.sleep(8)
    except Exception as e:
        print(f"[probe {i}] {type(e).__name__}: {str(e)[:80]}", flush=True); time.sleep(8)
else:
    raise SystemExit("warmup probe never returned non-empty content; aborting")
PY

cd external/monitorability-eval
source ../../.venv-eval/bin/activate
CFG="../../$CFG"
rm -rf "logs/$BN"

echo "===== BASELINE PASS ONLY (concurrency=8) ====="
python scripts/evals/batch_eval.py --config-file "$CFG" --baseline --batch-name "$BN" --no-batch 2>&1 | tail -8

echo "===== EXTRACT METRICS ====="
python scripts/analysis/extract_metrics.py --output-dir "../../$RESULTS" --batch-name "$BN" --dataset gpqa 2>&1 | tail -5
echo "===== GATE FLOW DONE: $BN ====="
