#!/bin/bash

# VTA single-stock demo (conditional reasoning guidance enabled)

# Ensure we run from the backbone directory so `python run.py` resolves
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# --- Parameters ---
GPUS="1"
ROOT_PATH="data/stocknet"
DATA_PATH="AAPL.csv"             # choose a single ticker file
SEQ_LEN=10
PRED_LEN=10
LABEL_LEN=0
PATIENCE=10
TRAIN_EPOCHS=100
BATCH_SIZE=16
EVAL_BATCH_SIZE=4
VISUALIZATION=1
SEED=2021
SCALER="price_standard"
DATA_NAME="StockData_raw"
TARGET_FEATURE="Adj Close"
FEATURES="MS"
USE_ALIGNMENT=1
ALIGNMENT_TYPE="stats"
LOSS_STATS="min,max,mean"
# Reference GRPO CSV directory (relative to repo). Change if you used a different model folder.
# Example: data/grpo_input/stocknet/Qwen2.5-7B-Instruct
REF_CSV_DIR="$PWD/data/grpo_input/stocknet/Qwen2.5-7B-Instruct"
LEARNING_RATE=0.0001
GUIDANCE_SCALE=0.1
P_UNCOND=0.30

# --- Environment Setup ---
export CUDA_VISIBLE_DEVICES=$GPUS

echo "VTA single-stock demo (single configuration, conditional reasoning guidance enabled)"

# Format values for IDs/prints
LR_FORMATTED=$(printf "%.4f" $LEARNING_RATE)
GS_FORMATTED=$(printf "%.1f" $GUIDANCE_SCALE)
P_UNCOND_FORMATTED=$(printf "%.2f" $P_UNCOND)
LOSS_STATS_SANITIZED=$(echo "$LOSS_STATS" | tr ',' '_')

# Construct Model ID
MODEL_ID="vta_DEMO_SINGLE_cfg_lr${LR_FORMATTED}_align${ALIGNMENT_TYPE}_stats${LOSS_STATS_SANITIZED}_gs${GS_FORMATTED}_puncond${P_UNCOND_FORMATTED}"

echo "MODEL_ID: ${MODEL_ID}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Alignment Type: ${ALIGNMENT_TYPE}"
echo "Loss Stats: ${LOSS_STATS}"
echo "Guidance Scale: ${GUIDANCE_SCALE}"
echo "P Uncond: ${P_UNCOND}"
echo "Ticker file: ${DATA_PATH}"
echo "---------------------------------------------"

# --- Run on single file ---
python run.py \
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --single_stock 1 \
    --model_id ${MODEL_ID} \
    --data ${DATA_NAME} \
    --features ${FEATURES} \
    --seq_len ${SEQ_LEN} \
    --label_len ${LABEL_LEN} \
    --target "${TARGET_FEATURE}" \
    --pred_len ${PRED_LEN} \
    --patience ${PATIENCE} \
    --train_epochs ${TRAIN_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --seed ${SEED} \
    --scaler ${SCALER} \
    --ref_sequence_csv_dir ${REF_CSV_DIR} \
    --visualization ${VISUALIZATION} \
    --learning_rate ${LEARNING_RATE} \
    --use_alignment ${USE_ALIGNMENT} \
    --alignment_type ${ALIGNMENT_TYPE} \
    --loss_stats ${LOSS_STATS} \
    --use_cfg \
    --guidance_scale ${GUIDANCE_SCALE} \
    --p_uncond ${P_UNCOND}
