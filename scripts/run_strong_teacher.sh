set -e
cd "/Users/abrahamyeung/CS 338 Project"
set -a && . ./.env && set +a
# strong teacher is openrouter/... — no Modal endpoint / OPENAI_BASE_URL needed
cd external/monitorability-eval
source ../../.venv-eval/bin/activate
CFG="../../configs/strong_teacher_gpqa.yaml"; BN="strong_teacher"
rm -rf logs/$BN
echo "===== BASELINE PASS (32B via OpenRouter) ====="
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
python scripts/analysis/extract_metrics.py --output-dir "../../results/strong_teacher_metrics" --batch-name "$BN" --dataset gpqa 2>&1 | tail -10
echo "===== STRONG TEACHER FLOW DONE ====="
