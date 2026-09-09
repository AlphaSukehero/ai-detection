#!/usr/bin/env bash
# Train every model in one go. Usage:
#   scripts/train_all.sh            # full run, default epochs per script
#   EPOCHS=1 scripts/train_all.sh   # smoke test: is the whole pipeline wired up?
#   scripts/train_all.sh ecg mri    # only the named models
set -u
set -o pipefail
cd "$(dirname "$0")/.."
# Prefer the local venv; fall back to whatever python is on PATH (Colab, CI).
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=$(command -v python3 || command -v python)

declare -A JOBS=(
  [ecg]=scripts/train_ecg_cnn.py
  [mri]=scripts/train_mri_cnn.py
  [vgg16]=scripts/train_mri_vgg16.py
  [eurosat]=scripts/train_eurosat.py
)
ORDER=(ecg mri vgg16 eurosat)
[ $# -gt 0 ] && ORDER=("$@")

mkdir -p logs
fail=0
for name in "${ORDER[@]}"; do
  script=${JOBS[$name]:-}
  if [ -z "$script" ]; then echo "unknown model: $name" >&2; fail=1; continue; fi
  log=logs/train_$name.log
  echo "=== $name -> $log"
  start=$SECONDS
  if $PY "$script" 2>&1 | tee "$log"; then
    echo "--- $name ok in $((SECONDS-start))s"
  else
    echo "--- $name FAILED (see $log)"; fail=1
  fi
done
exit $fail
