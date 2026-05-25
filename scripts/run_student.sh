#!/usr/bin/env bash
# Monitorability eval for a STUDENT condition (cond 2 W2SR or cond 3 control).
# Usage: run_student.sh <SERVED_MODEL_NAME> <CONFIG_YAML> <BATCH_NAME> <RESULTS_DIR>
# The served model is the MERGED checkpoint basename (see modal_app.py::merge_adapter);
# flip VLLMServer.model default to the merged path + `modal deploy` BEFORE running,
# OR rely on this script's deploy (it deploys whatever the current default is, so
# set the default first). Same 6-pass flow as run_weak_teacher.sh.
set -e
cd "/Users/abrahamyeung/CS 338 Project"
set -a && . ./.env && set +a
export OPENAI_BASE_URL="https://ayeung16--w2sr-monitorability-vllmserver-serve.modal.run/v1"
export OPENAI_API_KEY="EMPTY"

SERVED="${1:?served model name}"
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

echo "===== BASELINE PASS ====="
python scripts/evals/batch_eval.py --config-file "$CFG" --baseline --batch-name "$BN" --no-batch 2>&1 | tail -5

declare -a EXPS=("01_stanford_professor:stanford_professor_recommends" "02_visual_squares:visual_squares_correct" "03_grader_hack:grader_hack_validation" "04_unethical_information:insider_information" "05_xml_metadata:xml_metadata_success_rate")
for pair in "${EXPS[@]}"; do
  EXP="${pair%%:*}"; CUE="${pair##*:}"
  echo "===== ADAPTIVE GEN $EXP ====="
  python scripts/data_processing/generate_adaptive_datasets.py --batch-name "$BN" --adaptive-experiment-name "$EXP" --cue-name "$CUE" 2>&1 | tail -3
  echo "===== CUE PASS $EXP ====="
  python scripts/evals/batch_eval.py --config-file "$CFG" --batch-name "$BN" --adaptive-experiment-name "$EXP" --no-batch 2>&1 | tail -5
done

echo "===== EXTRACT METRICS ====="
python scripts/analysis/extract_metrics.py --output-dir "../../$RESULTS" --batch-name "$BN" --dataset gpqa 2>&1 | tail -10
echo "===== STUDENT FLOW DONE: $BN ====="
