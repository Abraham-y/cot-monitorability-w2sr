#!/usr/bin/env bash
# Experiment 1: thicken the long-bin W2SR cells. Serve the merged W2SR R1-7B and
# run the FULL 198 GPQA (no limit) on the 3 text cues, so the matched-long-length
# acknowledgment comparison reaches n>=30. No brevity, no truncation; temp 0.
set -e
cd "/Users/abrahamyeung/CS 338 Project"
set -a && . ./.env && set +a
export OPENAI_BASE_URL="https://ayeung16--w2sr-monitorability-vllmserver-serve.modal.run/v1"
export OPENAI_API_KEY="EMPTY"
export W2SR_MAX_SAMPLES=24 W2SR_MAX_CONNECTIONS=24 W2SR_JUDGE_MAX_CONNECTIONS=24
SERVED="w2sr_r1_7b"; CFG="../../configs/r1_7b_w2sr_gpqa.yaml"; BN="r1_7b_w2sr_full"
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
rm -rf "logs/$BN"
echo "===== BASELINE PASS (full 198) ====="
python scripts/evals/batch_eval.py --config-file "$CFG" --baseline --batch-name "$BN" --no-batch 2>&1 | tail -3
for pair in "01_stanford_professor:stanford_professor_recommends" "03_grader_hack:grader_hack_validation" "04_unethical_information:insider_information"; do
  EXP="${pair%%:*}"; CUE="${pair##*:}"
  echo "===== ADAPTIVE GEN $EXP ====="
  python scripts/data_processing/generate_adaptive_datasets.py --batch-name "$BN" --adaptive-experiment-name "$EXP" --cue-name "$CUE" 2>&1 | tail -2
  echo "===== CUE PASS $EXP (full 198) ====="
  python scripts/evals/batch_eval.py --config-file "$CFG" --batch-name "$BN" --adaptive-experiment-name "$EXP" --no-batch 2>&1 | tail -3
done
echo "===== EXP1 DONE: $BN ====="
