#!/usr/bin/env bash
set -euo pipefail

# 複数の設定とシードで一括実行するラッパー
# 例: ./scripts/run_experiments.sh
# デフォルトは csiro_biomass.yaml のみ。増やす場合は CONFIGS/SEED_MAP を編集。

CONFIGS=(
  "configs/csiro_biomass.yaml"
)

# 設定ごとのシード指定
declare -A SEED_MAP=(
  ["configs/csiro_biomass.yaml"]="42"
)

PYTHON_BIN=${PYTHON_BIN:-python}
SUMMARY_FILE="logs/experiment_summary.csv"
mkdir -p "$(dirname "$SUMMARY_FILE")"

# 進捗バー用
TOTAL_RUNS=0
for cfg in "${CONFIGS[@]}"; do
  seeds=${SEED_MAP[$cfg]}
  for _ in $seeds; do TOTAL_RUNS=$((TOTAL_RUNS+1)); done
done
DONE_RUNS=0

print_progress() {
  local done=$1 total=$2
  local width=30
  local pct=$(( 100 * done / total ))
  local filled=$(( width * done / total ))
  local bar=$(printf "%${filled}s" "" | tr ' ' '#')
  local space=$(printf "%$((width-filled))s" "" | tr ' ' '.')
  printf "\r[%s%s] %3d%% (%d/%d)" "$bar" "$space" "$pct" "$done" "$total"
}

write_summary() {
  local cfg="$1"
  local seed="$2"
  $PYTHON_BIN - "$cfg" "$seed" "$SUMMARY_FILE" <<'PY'
import json, yaml, sys, csv, pathlib
cfg, seed, summary_path = sys.argv[1], sys.argv[2], sys.argv[3]
cfg_path = pathlib.Path(cfg)
data = yaml.safe_load(cfg_path.read_text())
run_name = data.get("run_name", cfg_path.stem)
paths = data.get("paths", {})
models_dir = pathlib.Path(paths.get("models_dir", "models"))
metadata_path = models_dir / f"{run_name}_metadata.json"
if not metadata_path.exists():
    sys.exit(0)
meta = json.loads(metadata_path.read_text())
row = {
    "config": cfg,
    "seed": seed,
    "run_name": run_name,
    "oof_rmse": meta.get("oof_rmse"),
    "optuna_best_value": meta.get("optuna_best_value"),
    "fold_scores": ";".join(str(s) for s in meta.get("scores", [])),
    "optuna_best_params": meta.get("optuna_best_params"),
    "metadata_path": str(metadata_path),
}
summary = pathlib.Path(summary_path)
write_header = not summary.exists()
with summary.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

for cfg in "${CONFIGS[@]}"; do
  seeds=${SEED_MAP[$cfg]}
  for seed in $seeds; do
    echo ""
    echo "=== Running $cfg (seed=$seed) ==="
    ${PYTHON_BIN} scripts/train.py --config "$cfg" --seed "$seed" --skip-optuna
    write_summary "$cfg" "$seed"
    DONE_RUNS=$((DONE_RUNS+1))
    print_progress "$DONE_RUNS" "$TOTAL_RUNS"
  done
done
echo ""
