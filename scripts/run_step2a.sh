#!/usr/bin/env bash
# Step 2a length-control: serve baseline R1-distill-7B but CONSTRAIN-GENERATE short
# CoT (W2SR_BREVITY -> brevity directive in the prompt, NOT post-hoc truncation),
# then measure faithfulness on the 3 text cues. If baseline stays faithful at
# matched short length, CoT length is not the mechanism behind W2SR's drop.
# Usage: run_step2a.sh <SERVED> <CONFIG> <BATCH> <RESULTS> <LIMIT>
set -e
cd "/Users/abrahamyeung/CS 338 Project"
set -a && . ./.env && set +a
export OPENAI_BASE_URL="https://ayeung16--w2sr-monitorability-vllmserver-serve.modal.run/v1"
export OPENAI_API_KEY="EMPTY"
SERVED="${1:?served}"; CFG="${2:?config}"; BN="${3:?batch}"; RESULTS="${4:?results}"; LIM="${5:-40}"
export W2SR_BREVITY=1            # constrain-generate short, complete traces
export W2SR_LIMIT="$LIM"
export W2SR_MAX_SAMPLES=24 W2SR_MAX_CONNECTIONS=24 W2SR_JUDGE_MAX_CONNECTIONS=24

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
echo "===== BASELINE PASS (brevity, limit $LIM) ====="
python scripts/evals/batch_eval.py --config-file "$CFG" --baseline --batch-name "$BN" --no-batch 2>&1 | tail -3
for pair in "01_stanford_professor:stanford_professor_recommends" "03_grader_hack:grader_hack_validation" "04_unethical_information:insider_information"; do
  EXP="${pair%%:*}"; CUE="${pair##*:}"
  echo "===== ADAPTIVE GEN $EXP ====="
  python scripts/data_processing/generate_adaptive_datasets.py --batch-name "$BN" --adaptive-experiment-name "$EXP" --cue-name "$CUE" 2>&1 | tail -2
  echo "===== CUE PASS $EXP (brevity, limit $LIM) ====="
  python scripts/evals/batch_eval.py --config-file "$CFG" --batch-name "$BN" --adaptive-experiment-name "$EXP" --no-batch 2>&1 | tail -3
done
echo "===== EXTRACT ====="
python scripts/analysis/extract_metrics.py --output-dir "../../$RESULTS" --batch-name "$BN" --dataset gpqa 2>&1 | tail -4
echo "===== STEP2A DONE: $BN ====="
