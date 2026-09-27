#!/usr/bin/env bash
# Runs every evaluation step and then checks the results against expected_metrics/.
#
# Prerequisites:
#   1. python -m venv .venv && source .venv/bin/activate && pip install -r env/requirements.lock
#   2. bash scripts/fetch_data.sh
#
# Each step runs a module's main() with seed 42 and writes a JSON under models/;
# scripts/check_metrics.py then compares each one with its reference file.
#
# Written for stock macOS bash 3.2, so no associative arrays.
#
# Options:
#   FAST=1 bash scripts/reproduce.sh                            # skip the four slowest steps
#   STEPS="multidomain weblog_edgar" bash scripts/reproduce.sh  # run only named steps
#   PYTHON=/path/to/python bash scripts/reproduce.sh            # choose the interpreter
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONHASHSEED=0        # belt-and-suspenders; models also fix seed 42 internally
PY="${PYTHON:-python}"

# The order matters, because several steps load what an earlier step wrote:
#   lab_external loads models/lab_testbed/content_detector.joblib  -> after lab_testbed
#   adversarial and weblog_edgar load models/multidomain/text_only_zanbil.joblib -> after multidomain
#   multisite loads models/weblog_edgar/text_only_weblog.joblib    -> after weblog_edgar
#   family reads the SR-BH sample that sample rebuilds             -> after sample
# ablation_positives and coverage load no model; they sit next to the steps whose splits they reuse.
ALL_STEPS="lab_testbed lab_external single_source loso in_domain ablation_positives ml_vs_rules multidomain adversarial weblog_edgar multisite coverage sample family figure"
HEAVY_STEPS=" lab_testbed lab_external single_source loso ablation_positives coverage "     # skipped when FAST=1 (padded with spaces for word-match)

step_module() {   # echo the module name for a step, or empty if unknown
  case "$1" in
    lab_testbed)   echo lab_testbed_eval ;;
    lab_external)  echo lab_external_eval ;;
    single_source) echo single_source_eval ;;
    loso)          echo loso_eval ;;
    in_domain)     echo in_domain_train ;;
    ablation_positives) echo ablation_positives ;;
    ml_vs_rules)   echo ml_vs_rules_eval ;;
    multidomain)   echo multidomain ;;
    adversarial)   echo adversarial ;;
    weblog_edgar)  echo weblog_edgar_eval ;;
    multisite)     echo multisite_coverage_eval ;;
    coverage)      echo representation_coverage ;;
    sample)        echo rebuild_family_sample ;;
    family)        echo family_eval ;;
    figure)        echo make_figure ;;
    *)             echo "" ;;
  esac
}

STEPS="${STEPS:-$ALL_STEPS}"
fail=0
for step in $STEPS; do
  mod="$(step_module "$step")"
  if [ -z "$mod" ]; then echo "!! unknown step: $step"; continue; fi
  if [ "${FAST:-0}" = "1" ] && [ "${HEAVY_STEPS#* $step }" != "$HEAVY_STEPS" ]; then
    # family_head.joblib is a generated cache and is not committed. The `family` step needs it,
    # so when it is missing lab_testbed runs even under FAST=1 to rebuild it.
    if [ "$step" = "lab_testbed" ] && [ ! -f models/lab_testbed/family_head.joblib ]; then
      echo "== [FAST override] lab_testbed: family_head.joblib missing -> regenerating"
    else
      echo "== [skip:FAST] $step ($mod)"; continue
    fi
  fi
  echo "== reproduce: $step  ($mod)  =="
  t0=$(date +%s)
  if ! $PY -m "src.content_detector.$mod"; then
    echo "!! step '$step' FAILED (module src.content_detector.$mod)"; fail=1
  fi
  echo "   ($(( $(date +%s) - t0 ))s)"
done

echo
echo "================ VERIFY vs expected_metrics/ ================"
$PY scripts/check_metrics.py
rc=$?
[ "$fail" = "1" ] && echo "NOTE: at least one reproduce step errored (see above)."
exit $(( rc | fail ))
