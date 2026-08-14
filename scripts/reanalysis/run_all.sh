#!/usr/bin/env bash
# scripts/reanalysis/run_all.sh
#
# Execute the entire free (no-API) W2SR / CoT-monitorability reanalysis suite
# from the stored .eval records. Idempotent; overwrites results/reanalysis/* on
# each run.
#
# Order rationale:
#   01  gate            hard-fail headlines + extraction audit; nothing else
#                       runs if this fails
#   02  directional     Task 2 (full set, pooled) + flip-rate equivalence test
#   A   cue-correct     confound check (Task A) — refines Task 2
#   B   per-cue         per-text-cue effect sizes (Task B)
#   03  think-channel   training-trace + completion think-tag scan;
#                       needs `modal volume get` for the 4 trace files —
#                       SKIPPED automatically if they are not present
#   04  matched-length  logit residual; depends on r1_7b_w2sr_full
#   C   decode-check    static config audit
#   D   self-distill    teacher-strength invariance (Table 1 self-distill rows)
#   E   MMLU            cross-substrate replication (pooled)
#   E2  MMLU by subject per-subject split incl. the in-domain college_mathematics cut
#   F   Llama gate      cross-family capability gate (failed null)
#   G   rubric swap     within-judge strict-rubric robustness — PAID, gated by
#                       RUBRIC=1, and not bit-reproducible (judge noise)
#   H   length-binned   matched-length Fisher, all cued samples
#   H2  H by influenced same, split by influenced status (the safety-relevant cut)
#   05  rejudge         per-cue ack matrix (free); the out-of-family rejudge
#                       itself is gated by REJUDGE=1 (paid)
#
# Tasks I2/I3 (CoT-conclusion judge) and J (inference-time recovery) are NOT in
# this suite: they call a paid judge API and, for J, a served model on Modal.
# See README.md for their invocations.
#
# Usage:
#   bash scripts/reanalysis/run_all.sh           # everything free
#   REJUDGE=1 bash scripts/reanalysis/run_all.sh # also run Task 5 Part B (~$2)
#   RUBRIC=1  bash scripts/reanalysis/run_all.sh # also re-run Task G  (~$1.9)
#
# Requirements:
#   - .venv-eval/ activated (statsmodels, scipy, openai, modal pinned there)
#   - .env with OPENROUTER_API_KEY exported (only matters if REJUDGE=1)
#   - modal CLI authenticated (only for the optional Task 3 trace fetch)

set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root

# shellcheck disable=SC1091
source .venv-eval/bin/activate
if [[ -f ./.env ]]; then
    set -a; . ./.env; set +a         # OPENROUTER_API_KEY etc.
fi

PYTHON=python3
SCRIPTS_DIR=scripts/reanalysis

run() { echo; echo "======== $* ========"; "$@"; }

# --- Task 3 prereq: fetch the four training trace files if missing ---
# Task 3 is the only task needing anything beyond the stored .eval files. If the
# traces are absent and Modal is unavailable, skip it rather than killing the
# suite (set -e would otherwise abort every task after it).
TRACE_DIR=/tmp/w2sr_traces
mkdir -p "$TRACE_DIR"
HAVE_TRACES=1
for stem in w2sr w2sr_r1_14b w2sr_infamily w2sr_infamily_strong; do
    if [[ ! -s "$TRACE_DIR/${stem}_train.json" ]]; then
        echo "fetching $stem/train.json from Modal ..."
        if ! modal volume get w2sr-vol "/traces/$stem/train.json" \
                "$TRACE_DIR/${stem}_train.json" 2>/dev/null; then
            echo "  !! could not fetch $stem/train.json (modal unavailable?)"
            HAVE_TRACES=0
        fi
    fi
done

run "$PYTHON" "$SCRIPTS_DIR/01_gate.py"
run "$PYTHON" "$SCRIPTS_DIR/02_directional_influence.py"
run "$PYTHON" "$SCRIPTS_DIR/A_cue_correct_confound.py"
run "$PYTHON" "$SCRIPTS_DIR/B_per_cue_effects.py"

if [[ "$HAVE_TRACES" == "1" ]]; then
    run "$PYTHON" "$SCRIPTS_DIR/03_think_channel_collapse.py"
else
    echo
    echo "======== SKIPPING 03_think_channel_collapse.py ========"
    echo "  Training traces not in $TRACE_DIR. Fetch them with:"
    echo "    modal volume get w2sr-vol /traces/<stem>/train.json $TRACE_DIR/<stem>_train.json"
    echo "  (stems: w2sr w2sr_r1_14b w2sr_infamily w2sr_infamily_strong)"
fi

run "$PYTHON" "$SCRIPTS_DIR/04_matched_length_logit.py"
run "$PYTHON" "$SCRIPTS_DIR/C_decode_config_check.py"
run "$PYTHON" "$SCRIPTS_DIR/D_self_distillation_negcontrol.py"
run "$PYTHON" "$SCRIPTS_DIR/E_cross_substrate_mmlu.py"
run "$PYTHON" "$SCRIPTS_DIR/E2_mmlu_by_subject.py"
run "$PYTHON" "$SCRIPTS_DIR/F_llama_capability_gate.py"

# Task G calls the Gemini judge (~$1.9) and is NOT bit-reproducible run to run:
# the judge returns slightly different labels each time (an identical rerun moved
# baseline ack 9.4% -> 8.8%, discordant 15/0 -> 14/0, p 6.1e-5 -> 1.2e-4; same
# conclusion, different digits). It is therefore opt-in, and the committed
# labels/summary are the ones the manuscript cites.
if [[ "${RUBRIC:-0}" == "1" ]]; then
    run "$PYTHON" "$SCRIPTS_DIR/G_robustness_rubric.py"
else
    echo
    echo "======== SKIPPING G_robustness_rubric.py (paid, non-deterministic) ========"
    echo "  Rerun with RUBRIC=1 to pay ~\$1.9 and overwrite the committed labels."
fi

run "$PYTHON" "$SCRIPTS_DIR/H_length_binned_ack.py"
run "$PYTHON" "$SCRIPTS_DIR/H_length_binned_ack_by_influenced.py"

# Task K (CoT-preserving rerun) reads the r1_7b_w2sr_cotsft eval logs; skip
# gracefully if that batch is not on this machine.
if [[ -d external/monitorability-eval/logs/r1_7b_w2sr_cotsft ]]; then
    run "$PYTHON" "$SCRIPTS_DIR/K_cotsft_rerun.py"
else
    echo
    echo "======== SKIPPING K_cotsft_rerun.py (r1_7b_w2sr_cotsft logs absent) ========"
fi

# --- Figures ---
# These regenerate the manuscript figures FROM the JSON written above, so a
# result change can never silently leave a stale figure in the paper. That is
# not hypothetical: dissociation_bars.pdf had no generator at all and sat for
# nine days plotting a superseded baseline influence rate (35.8% vs 25.6%)
# after the answer-extraction fix. Keep these in the suite.
run "$PYTHON" "$SCRIPTS_DIR/make_dissociation_fig.py"
run "$PYTHON" "$SCRIPTS_DIR/make_mechanism_fig.py"
run "$PYTHON" "$SCRIPTS_DIR/make_length_binned_fig.py"

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
