#!/usr/bin/env bash
# scripts/reanalysis/run_all.sh
#
# Execute the entire W2SR / CoT-monitorability reanalysis suite from the stored
# .eval records and (for Task 3) the Modal volume w2sr-vol. Idempotent;
# overwrites results/reanalysis/* on each run.
#
# Order rationale:
#   01  gate            hard-fail headlines + extraction audit; nothing else
#                       runs if this fails
#   02  directional     Task 2 (full set, pooled)
#   A   cue-correct     confound check (Task A) — refines Task 2
#   B   per-cue         per-text-cue effect sizes (Task B)
#   03  think-channel   training-trace + completion think-tag scan;
#                       requires `modal volume get` for the 4 trace files
#   04  matched-length  logit residual; depends on r1_7b_w2sr_full
#   C   decode-check    static config audit
#   05  rejudge         per-cue ack matrix (free); rejudge under
#                       google/gemini-2.5-pro is gated by REJUDGE=1 (paid)
#
# Usage:
#   bash scripts/reanalysis/run_all.sh           # everything except the paid rejudge
#   REJUDGE=1 bash scripts/reanalysis/run_all.sh # also run Task 5 Part B (~$2)
#
# Requirements:
#   - .venv-eval/ activated (statsmodels, scipy, openai, modal pinned there)
#   - .env with OPENROUTER_API_KEY exported (only matters if REJUDGE=1)
#   - modal CLI authenticated (only matters once, for Task 3 trace fetch)

set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root

# shellcheck disable=SC1091
source .venv-eval/bin/activate
set -a; . ./.env; set +a             # OPENROUTER_API_KEY etc.

PYTHON=python3
SCRIPTS_DIR=scripts/reanalysis

run() { echo; echo "======== $* ========"; "$@"; }

# --- Task 3 prereq: fetch the four training trace files if missing ---
TRACE_DIR=/tmp/w2sr_traces
mkdir -p "$TRACE_DIR"
for stem in w2sr w2sr_r1_14b w2sr_infamily w2sr_infamily_strong; do
    if [[ ! -s "$TRACE_DIR/${stem}_train.json" ]]; then
        echo "fetching $stem/train.json from Modal ..."
        modal volume get w2sr-vol "/traces/$stem/train.json" \
            "$TRACE_DIR/${stem}_train.json"
    fi
done

run "$PYTHON" "$SCRIPTS_DIR/01_gate.py"
run "$PYTHON" "$SCRIPTS_DIR/02_directional_influence.py"
run "$PYTHON" "$SCRIPTS_DIR/A_cue_correct_confound.py"
run "$PYTHON" "$SCRIPTS_DIR/B_per_cue_effects.py"
run "$PYTHON" "$SCRIPTS_DIR/03_think_channel_collapse.py"
run "$PYTHON" "$SCRIPTS_DIR/04_matched_length_logit.py"
run "$PYTHON" "$SCRIPTS_DIR/C_decode_config_check.py"

# Task 5: Part A (per-cue ack matrix) is free; Part B (rejudge) is paid.
if [[ "${REJUDGE:-0}" == "1" ]]; then
    run "$PYTHON" "$SCRIPTS_DIR/05_robustness_rejudge.py" \
        --run-rejudge --concurrency 16
else
    run "$PYTHON" "$SCRIPTS_DIR/05_robustness_rejudge.py"
    echo
    echo "(Skipped Task 5 Part B rejudge; rerun with REJUDGE=1 to pay ~\$2 for"
    echo " 525 records under google/gemini-2.5-pro via OpenRouter.)"
fi

echo
echo "======== run_all.sh complete ========"
ls -1 results/reanalysis/
