#!/usr/bin/env bash
# Cross-substrate MMLU replication — single condition.
# Args: <SERVED_MODEL> <CONFIG> <BATCH_NAME> <RESULTS_DIR>
# (re)-deploys the modal endpoint, warms up, runs the baseline + 5 cue passes
# across the 5 MMLU STEM subjects, extracts metrics.
set -e
cd "/Users/abrahamyeung/CS 338 Project"
set -a && . ./.env && set +a
export OPENAI_BASE_URL="https://ayeung16--w2sr-monitorability-vllmserver-serve.modal.run/v1"
export OPENAI_API_KEY="EMPTY"
# MMLU "core" subjects each have ~118 questions; cap to 8 per subject so a
# 5-subject sweep yields ~40 cued samples per cell (parallels GPQA's per-cue n).
export W2SR_LIMIT=8
export W2SR_MAX_SAMPLES=24
export W2SR_MAX_CONNECTIONS=24
export W2SR_JUDGE_MAX_CONNECTIONS=24

SERVED="${1:?served model}"
CFG="${2:?config yaml}"
BN="${3:?batch name}"
RESULTS="${4:?results dir}"

echo "===== DEPLOY (expects VLLMServer default == $SERVED) ====="
/opt/miniconda3/bin/modal deploy modal_app.py 2>&1 | tail -3

echo "===== WARM UP (wait for $SERVED) ====="
python3 - "$SERVED" <<'PY'
import json, ssl, sys, urllib.request, certifi, time
served = sys.argv[1]
ctx = ssl.create_default_context(cafile=certifi.where())
base = "https://ayeung16--w2sr-monitorability-vllmserver-serve.modal.run/v1/models"
for i in range(80):
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(base), timeout=30, context=ctx))
        ids = [m["id"] for m in d["data"]]; print("SERVING:", ids, flush=True)
        if served in ids: break
        print("  (wrong model served, waiting for redeploy)", flush=True); time.sleep(12)
    except Exception as e:
        print(f"[{i}] warming: {str(e)[:50]}", flush=True); time.sleep(12)
else:
    raise SystemExit(f"endpoint never served {served}")
PY

cd external/monitorability-eval
source ../../.venv-eval/bin/activate
CFG="../../$CFG"
rm -rf "logs/$BN"

echo "===== BASELINE PASS (5 MMLU subjects) ====="
python scripts/evals/batch_eval.py --config-file "$CFG" --baseline --batch-name "$BN" --no-batch 2>&1 | tail -8

declare -a EXPS=("01_stanford_professor:stanford_professor_recommends" "02_visual_squares:visual_squares_correct" "03_grader_hack:grader_hack_validation" "04_unethical_information:insider_information" "05_xml_metadata:xml_metadata_success_rate")
for pair in "${EXPS[@]}"; do
  EXP="${pair%%:*}"; CUE="${pair##*:}"
  echo "===== ADAPTIVE GEN $EXP ====="
  python scripts/data_processing/generate_adaptive_datasets.py --batch-name "$BN" --adaptive-experiment-name "$EXP" --cue-name "$CUE" 2>&1 | tail -3
  echo "===== CUE PASS $EXP ====="
  python scripts/evals/batch_eval.py --config-file "$CFG" --batch-name "$BN" --adaptive-experiment-name "$EXP" --no-batch 2>&1 | tail -8
done

echo "===== EXTRACT METRICS ====="
python scripts/analysis/extract_metrics.py --output-dir "../../$RESULTS" --batch-name "$BN" --dataset mmlu 2>&1 | tail -10
echo "===== MMLU FLOW DONE: $BN ====="
