#!/usr/bin/env bash
# Reproduce the paper's numbers and figures. Run from the repository root.
#   bash scripts/reproduce.sh          # fast: recompute all tests and figures from results/prereg/episodes.jsonl
#   bash scripts/reproduce.sh --full   # also rerun all 3240 formal episodes first (hours on a desktop CPU)
set -euo pipefail
PY=${PY:-.venv/bin/python}
if [[ "${1:-}" == "--full" ]]; then
  mv results/prereg/episodes.jsonl results/prereg/episodes.published.jsonl
  "$PY" experiments/prereg_run.py
fi
"$PY" experiments/prereg_analyze.py
"$PY" paper/workshop/scripts/recompute_medians_slopes.py
"$PY" paper/workshop/scripts/recompute_h4_sens.py
"$PY" paper/workshop/scripts/make_figs.py
git --no-pager diff --stat -- results/prereg paper/workshop/figs || true
