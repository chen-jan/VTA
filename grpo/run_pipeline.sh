#!/bin/bash

# Model Configuration
DATASET="stocknet" # change to your dataset folder name

# MODEL="meta-llama/Llama-3.2-3B-Instruct"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH=2048
LORA_RANK=32
CUDA_DEVICES="0,1,2,3"
GPU_MEMORY_UTILIZATION=0.50
STOCK="all_data"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Set up paths
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="results/${STOCK}_${MODEL}_"
mkdir -p $BASE_DIR

# Data paths
DATA_DIR="data/${DATASET}/${STOCK}"
TRAIN_X_PATH="${DATA_DIR}/${STOCK}_train_x.csv"
TRAIN_Y_PATH="${DATA_DIR}/${STOCK}_train_y.csv"
TEST_X_PATH="${DATA_DIR}/${STOCK}_test_x.csv"
TEST_Y_PATH="${DATA_DIR}/${STOCK}_test_y.csv"
FULL_X_PATH="${DATA_DIR}/${STOCK}_full_x.csv"
FULL_Y_PATH="${DATA_DIR}/${STOCK}_full_y.csv"

# Model save paths
MODELS_DIR="models/${MODEL}/${STOCK}"
mkdir -p $MODELS_DIR
GRPO_SAVED_LORA_1ST="${MODELS_DIR}/grpo_lora_1st"
SFT_SAVED_LORA="${MODELS_DIR}/sft_lora"
GRPO_SAVED_LORA_2ND="${MODELS_DIR}/grpo_lora_2nd"

# Log file
LOG_FILE="${BASE_DIR}/pipeline.log"

# SFT dataset path
SFT_DATASET_PATH="${BASE_DIR}/sft_data_generation/${STOCK}_sft_training_data.csv"

echo "Starting pipeline at $(date)" | tee $LOG_FILE

# Stage 1
echo "Stage 1: Training first GRPO model" | tee -a $LOG_FILE
python stage1_train_grpo.py \
    --stock ${STOCK} \
    --model ${MODEL} \
    --max-seq-length ${MAX_SEQ_LENGTH} \
    --lora-rank ${LORA_RANK} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --train-x-path ${TRAIN_X_PATH} \
    --train-y-path ${TRAIN_Y_PATH} \
    --test-x-path ${TEST_X_PATH} \
    --test-y-path ${TEST_Y_PATH} \
    --full-x-path ${FULL_X_PATH} \
    --full-y-path ${FULL_Y_PATH} \
    --grpo-saved-lora-1st ${GRPO_SAVED_LORA_1ST} \
    --grpo-saved-lora-2nd ${GRPO_SAVED_LORA_2ND} \
    --sft-saved-lora ${SFT_SAVED_LORA} \
    --results-dir ${BASE_DIR} 2>&1 | tee -a $LOG_FILE || exit 1

# Stage 1.5
echo "Stage 1.5: Generating SFT data" | tee -a $LOG_FILE
python sft_data_generation.py \
    --stock ${STOCK} \
    --model ${MODEL} \
    --max-seq-length ${MAX_SEQ_LENGTH} \
    --lora-rank ${LORA_RANK} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --train-x-path ${TRAIN_X_PATH} \
    --train-y-path ${TRAIN_Y_PATH} \
    --test-x-path ${TEST_X_PATH} \
    --test-y-path ${TEST_Y_PATH} \
    --full-x-path ${FULL_X_PATH} \
    --full-y-path ${FULL_Y_PATH} \
    --grpo-saved-lora-1st ${GRPO_SAVED_LORA_1ST} \
    --grpo-saved-lora-2nd ${GRPO_SAVED_LORA_2ND} \
    --sft-saved-lora ${SFT_SAVED_LORA} \
    --sft-data-path ${SFT_DATASET_PATH} \
    --results-dir ${BASE_DIR} 2>&1 | tee -a $LOG_FILE || exit 1

# Stage 2
echo "Stage 2: Training SFT model" | tee -a $LOG_FILE
python stage2_train_sft.py \
    --stock ${STOCK} \
    --model ${MODEL} \
    --max-seq-length ${MAX_SEQ_LENGTH} \
    --lora-rank ${LORA_RANK} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --train-x-path ${TRAIN_X_PATH} \
    --train-y-path ${TRAIN_Y_PATH} \
    --test-x-path ${TEST_X_PATH} \
    --test-y-path ${TEST_Y_PATH} \
    --full-x-path ${FULL_X_PATH} \
    --full-y-path ${FULL_Y_PATH} \
    --grpo-saved-lora-1st ${GRPO_SAVED_LORA_1ST} \
    --grpo-saved-lora-2nd ${GRPO_SAVED_LORA_2ND} \
    --sft-saved-lora ${SFT_SAVED_LORA} \
    --sft-data-path ${SFT_DATASET_PATH} \
    --results-dir ${BASE_DIR} 2>&1 | tee -a $LOG_FILE || exit 1

# Stage 3
echo "Stage 3: Training second GRPO model" | tee -a $LOG_FILE
python stage3_train_grpo_from_sft.py \
    --stock ${STOCK} \
    --model ${MODEL} \
    --max-seq-length ${MAX_SEQ_LENGTH} \
    --lora-rank ${LORA_RANK} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --train-x-path ${TRAIN_X_PATH} \
    --train-y-path ${TRAIN_Y_PATH} \
    --test-x-path ${TEST_X_PATH} \
    --test-y-path ${TEST_Y_PATH} \
    --full-x-path ${FULL_X_PATH} \
    --full-y-path ${FULL_Y_PATH} \
    --grpo-saved-lora-1st ${GRPO_SAVED_LORA_1ST} \
    --grpo-saved-lora-2nd ${GRPO_SAVED_LORA_2ND} \
    --sft-saved-lora ${SFT_SAVED_LORA} \
    --results-dir ${BASE_DIR} 2>&1 | tee -a $LOG_FILE || exit 1

# Split GRPO outputs into per-stock CSVs for backbone
echo "Splitting GRPO outputs into per-stock CSVs for backbone" | tee -a $LOG_FILE
python split_grpo_outputs.py \
    --results-dir ${BASE_DIR} \
    --full-x-path ${FULL_X_PATH} \
    --model ${MODEL} \
    --dataset-label stocknet \
    --output-root ${PWD}/../backbone/data/grpo_input 2>&1 | tee -a $LOG_FILE || exit 1

# # Compare models (optional)
# echo "Comparing models" | tee -a $LOG_FILE
# python compare_models.py \
#     --stock ${STOCK} \
#     --model ${MODEL} \
#     --max-seq-length ${MAX_SEQ_LENGTH} \
#     --lora-rank ${LORA_RANK} \
#     --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
#     --train-x-path ${TRAIN_X_PATH} \
#     --train-y-path ${TRAIN_Y_PATH} \
#     --test-x-path ${TEST_X_PATH} \
#     --test-y-path ${TEST_Y_PATH} \
#     --full-x-path ${FULL_X_PATH} \
#     --full-y-path ${FULL_Y_PATH} \
#     --grpo-saved-lora-1st ${GRPO_SAVED_LORA_1ST} \
#     --grpo-saved-lora-2nd ${GRPO_SAVED_LORA_2ND} \
#     --sft-saved-lora ${SFT_SAVED_LORA} \
#     --results-dir ${BASE_DIR} 2>&1 | tee -a $LOG_FILE || exit 1

echo "Pipeline completed at $(date)" | tee -a $LOG_FILE
echo "Results saved to: $BASE_DIR" | tee -a $LOG_FILE