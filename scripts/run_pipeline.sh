#!/usr/bin/env bash
set -euo pipefail

# One-stop script to train and generate a submission using the specified config.
# Usage: bash scripts/run_pipeline.sh [config_path]
# Optional env vars:
#   SEED: random seed passed to train/predict (default: 42)
#   SKIP_OPTUNA: set to 0 to enable Optuna search (default: 1 to skip)
#   DRY_RUN: set to 1 to only validate config/data paths and skip training/predict

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

CONFIG_PATH=${1:-configs/csiro_biomass.yaml}
SEED=${SEED:-42}
SKIP_OPTUNA=${SKIP_OPTUNA:-1}
DRY_RUN=${DRY_RUN:-0}

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

# Extract minimal path info from YAML (input_dir, train/test filenames, sample submission)
CONFIG_FIELDS=$(python - <<'PY'
import sys, yaml
from pathlib import Path

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text())
paths = data.get("paths") or {}
files = data.get("files") or {}
input_dir = paths.get("input_dir", "data/csiro_biomass")
train = files.get("train", "train.csv")
test = files.get("test", "test.csv")
sample = files.get("sample_submission", "sample_submission.csv")
print("\t".join([str(input_dir), str(train), str(test), str(sample)]))
PY
"$CONFIG_PATH")

IFS=$'\t' read -r INPUT_DIR TRAIN_FILE TEST_FILE SAMPLE_FILE <<<"$CONFIG_FIELDS"

ensure_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[run_pipeline] Missing ${label}: $path" >&2
    exit 1
  fi
}

ensure_file "$CONFIG_PATH" "config"
ensure_file "$INPUT_DIR/$TRAIN_FILE" "train CSV"
ensure_file "$INPUT_DIR/$TEST_FILE" "test CSV"
ensure_file "$INPUT_DIR/$SAMPLE_FILE" "sample_submission CSV"

if [[ "$SKIP_OPTUNA" == "0" ]]; then
  OPTUNA_FLAG=""
else
  OPTUNA_FLAG="--skip-optuna"
fi

echo "[run_pipeline] Using config: $CONFIG_PATH"
echo "[run_pipeline] Seed: $SEED"
echo "[run_pipeline] Data dir: $INPUT_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[run_pipeline] Dry-run only; config and data files are present."
  exit 0
fi

python scripts/train.py --config "$CONFIG_PATH" --seed "$SEED" $OPTUNA_FLAG
python scripts/predict.py --config "$CONFIG_PATH" --seed "$SEED"

SUBMISSION_PATH=$(python - <<'PY'
import sys, yaml
from pathlib import Path

cfg_path = Path(sys.argv[1])
data = yaml.safe_load(cfg_path.read_text())
paths = data.get("paths") or {}
inference = data.get("inference") or {}
run_name = data.get("run_name", "csiro_biomass")
sub_dir = paths.get("submissions_dir", "submissions")
filename = inference.get("output_filename", f"{run_name}_submission.csv")
print(Path(sub_dir) / filename)
PY
"$CONFIG_PATH")

if [[ -f "$SUBMISSION_PATH" ]]; then
  echo "[run_pipeline] Submission created: $SUBMISSION_PATH"
else
  echo "[run_pipeline] Submission not found: expected $SUBMISSION_PATH" >&2
  exit 1
fi
