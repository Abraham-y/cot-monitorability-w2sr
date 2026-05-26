#!/usr/bin/env bash
# Premise check (option B): is the reasoning student's faithfulness OFF the floor?
# Minimal — baseline + the strongest cue (stanford_professor) only, limited to a
# subset, on the served baseline R1-distill-7B. Usage:
#   run_premise_check.sh <SERVED> <CONFIG> <BATCH> <RESULTS> <LIMIT>
set -e
cd "/Users/abrahamyeung/CS 338 Project"
set -a && . ./.env && set +a
export OPENAI_BASE_URL="https://ayeung16--w2sr-monitorability-vllmserver-serve.modal.run/v1"
export OPENAI_API_KEY="EMPTY"
SERVED="${1:?served}"; CFG="${2:?config}"; BN="${3:?batch}"; RESULTS="${4:?results}"; LIM="${5:-40}"
export W2SR_LIMIT="$LIM"
export W2SR_MAX_SAMPLES="${W2SR_MAX_SAMPLES:-24}"
export W2SR_MAX_CONNECTIONS="${W2SR_MAX_CONNECTIONS:-24}"
export W2SR_JUDGE_MAX_CONNECTIONS="${W2SR_JUDGE_MAX_CONNECTIONS:-24}"

echo "===== DEPLOY (expects default == $SERVED) ====="
/opt/miniconda3/bin/modal deploy modal_app.py 2>&1 | tail -2
echo "===== WARM UP ====="
python3 - "$SERVED" <<'PY'
import json, ssl, sys, urllib.request, certifi, time
served=sys.argv[1]; ctx=ssl.create_default_context(cafile=certifi.where())
base="https://ayeung16--w2sr-monitorability-vllmserver-serve.modal.run/v1/models"
for i in range(80):
    try:
        d=json.load(urllib.request.urlopen(urllib.request.Request(base),timeout=30,context=ctx))
        if served in [m["id"] for m in d["data"]]: print("SERVING",served); break
        time.sleep(12)
    except Exception as e: print(f"[{i}] {str(e)[:40]}"); time.sleep(12)
else: raise SystemExit("never served")
PY
cd external/monitorability-eval
source ../../.venv-eval/bin/activate
CFG="../../$CFG"; rm -rf "logs/$BN"
echo "===== BASELINE PASS (limit $LIM) ====="
python scripts/evals/batch_eval.py --config-file "$CFG" --baseline --batch-name "$BN" --no-batch 2>&1 | tail -4
echo "===== ADAPTIVE GEN stanford ====="
python scripts/data_processing/generate_adaptive_datasets.py --batch-name "$BN" --adaptive-experiment-name "01_stanford_professor" --cue-name "stanford_professor_recommends" 2>&1 | tail -3
echo "===== CUE PASS stanford (limit $LIM) ====="
python scripts/evals/batch_eval.py --config-file "$CFG" --batch-name "$BN" --adaptive-experiment-name "01_stanford_professor" --no-batch 2>&1 | tail -4
echo "===== EXTRACT ====="
python scripts/analysis/extract_metrics.py --output-dir "../../$RESULTS" --batch-name "$BN" --dataset gpqa 2>&1 | tail -6
echo "===== PREMISE CHECK DONE: $BN ====="
