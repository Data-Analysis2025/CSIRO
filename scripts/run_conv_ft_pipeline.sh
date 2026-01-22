#!/usr/bin/env bash
set -euo pipefail

# Environment variables (override as needed):
#   DATA_DIR     : path to data directory (default: data)
#   MODEL_NAME   : timm model to fine-tune (default: convnextv2_large)
#   IMAGE_SIZE   : resize size for training (default: 576)
#   BATCH_SIZE   : training batch size (default: 8)
#   EPOCHS       : training epochs (default: 10)
#   FOLDS        : number of folds (default: 5)
#   OUT_DIR      : directory to store checkpoints (default: models)
#   RUN_TAG      : suffix for artifacts (default: current timestamp)
#   EMB_PREFIX   : embedding file prefix (default: convft_${MODEL_NAME}_${RUN_TAG})
#   ENSEMBLE_DIR : output dir for LGBM/XGB/Cat models (default: models/large_ensemble_${MODEL_NAME}_${RUN_TAG})
#   USE_ALB      : set to 1 to enable Albumentations aug (default: 1)
#   MIXUP_ALPHA  : mixup alpha (default: 0.4)
#   CUTMIX_ALPHA : cutmix alpha (default: 0.0)
#   SCHEDULER    : lr scheduler (default: cosine)

DATA_DIR=${DATA_DIR:-data}
MODEL_NAME=${MODEL_NAME:-convnextv2_large}
IMAGE_SIZE=${IMAGE_SIZE:-576}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-20}
FOLDS=${FOLDS:-5}
OUT_DIR=${OUT_DIR:-models}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
EMB_PREFIX=${EMB_PREFIX:-convft_${MODEL_NAME}_${RUN_TAG}}
ENSEMBLE_DIR=${ENSEMBLE_DIR:-models/large_ensemble_${MODEL_NAME}_${RUN_TAG}}
USE_ALB=${USE_ALB:-1}
MIXUP_ALPHA=${MIXUP_ALPHA:-0.4}
CUTMIX_ALPHA=${CUTMIX_ALPHA:-0.0}
SCHEDULER=${SCHEDULER:-cosine}
MIN_LR=${MIN_LR:-1e-5}

echo "=== Step 1: Fine-tuning ${MODEL_NAME} (${FOLDS} folds) ==="
python scripts/train_image.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --image-size "${IMAGE_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --folds "${FOLDS}" \
  --lr 5e-5 \
  --scheduler "${SCHEDULER}" \
  --min-lr "${MIN_LR}" \
  --mixup-alpha "${MIXUP_ALPHA}" \
  --cutmix-alpha "${CUTMIX_ALPHA}" \
  $( [[ "${USE_ALB}" == "1" ]] && echo "--use-albumentations" ) \
  --pretrained

CKPTS=()
for ((fold=0; fold<FOLDS; fold++)); do
  CKPT_PATH="${OUT_DIR}/${MODEL_NAME}/fold_${fold}/best.pt"
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Missing checkpoint: ${CKPT_PATH}" >&2
    exit 1
  fi
  CKPTS+=("${CKPT_PATH}")
done

echo "=== Step 2: Extracting Conv embeddings (${EMB_PREFIX}) ==="
python scripts/extract_conv_features.py \
  --data-dir "${DATA_DIR}" \
  --checkpoint-paths "${CKPTS[@]}" \
  --output-prefix "${EMB_PREFIX}"

echo "=== Step 3: Training LGBM/XGB/Cat on Conv embeddings ==="
python scripts/train_large_models.py \
  --train-embeddings "${DATA_DIR}/train_embeddings_${EMB_PREFIX}.csv" \
  --models-dir "${ENSEMBLE_DIR}"

echo "Pipeline finished. Embeddings stored with prefix ${EMB_PREFIX}. Models saved under ${ENSEMBLE_DIR}."
