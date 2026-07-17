#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/cist-work}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/entrokv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/models/Llama-3.1-8B-Instruct}"
DATA_DIR="${DATA_DIR:-/root/autodl-fs/longbench}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
BUDGET_RATIO="${BUDGET_RATIO:-0.30}"
TASKS=(narrativeqa hotpotqa qasper)
EXP_DIR="$PROJECT/external/AdaKV/experiments/LongBench"
LOG_DIR="$PROJECT/outputs/logs"
BUDGET_DIR="$PROJECT/outputs/budgets"

mkdir -p "$LOG_DIR" "$BUDGET_DIR"
cd "$EXP_DIR"

run_mode() {
  local mode="$1"
  local name="$2"
  local extra=()
  if [[ "$mode" != "test" ]]; then
    extra+=(--gqa_support --gqa_func mean --budget_ratio "$BUDGET_RATIO")
  fi
  if [[ "$mode" == "entro" ]]; then
    extra+=(--entropy_alpha 0.5 --entropy_baseline 0.3)
    extra+=(--budget_log_path "$BUDGET_DIR/${name}_{dataset}.jsonl")
  fi

  "$PYTHON_BIN" pred.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --max_length "$MAX_LENGTH" \
    --mode "$mode" \
    --out_name "$name" \
    --tasks "${TASKS[@]}" \
    --max_samples "$MAX_SAMPLES" \
    "${extra[@]}" 2>&1 | tee "$LOG_DIR/${name}.log"
}

run_mode test full_cache
run_mode fix snapkv_fixed_30pct
run_mode entro entrokv_30pct

"$PYTHON_BIN" eval.py --allow_partial --models \
  full_cache snapkv_fixed_30pct entrokv_30pct 2>&1 | tee "$LOG_DIR/eval.log"

"$PYTHON_BIN" "$PROJECT/scripts/summarize_results.py" \
  pred/full_cache/*.jsonl pred/snapkv_fixed_30pct/*.jsonl pred/entrokv_30pct/*.jsonl \
  --output "$PROJECT/outputs/efficiency_summary.json"

"$PYTHON_BIN" "$PROJECT/scripts/plot_budgets.py" \
  "narrativeqa=$BUDGET_DIR/entrokv_30pct_narrativeqa.jsonl" \
  "hotpotqa=$BUDGET_DIR/entrokv_30pct_hotpotqa.jsonl" \
  --output "$PROJECT/outputs/budget_heatmaps.png"
