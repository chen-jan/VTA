# GRPO: Generative Reinforcement Policy Optimization for Financial Time Series Forecasting

This README provides detailed information about the GRPO (Generative Reinforcement Policy Optimization) training pipeline for financial time series forecasting.

## Overview

GRPO is a multi-stage training approach that combines reinforcement learning and supervised fine-tuning to create more effective time series forecasting models for financial data. The pipeline processes raw OHLCV (Open, High, Low, Close, Volume) stock data and trains language models to predict future price movements.

## Data Processing

Before model training begins, raw OHLCV stock data is processed using `combined_preprocessor.py`, which:

1. Loads raw stock price data from CSV files located in `backbone/data/stocknet` (by default; configurable via CLI)
2. Processes this data through the following steps:
   - Normalizes price data using `StandardNorm` layer
   - Calculates various technical indicators, including:
     - Simple Moving Average (SMA)
     - Exponential Moving Average (EMA)
     - Momentum (MOM)
     - Stochastic oscillator (STOCH_K, STOCH_D)
     - Relative Strength Index (RSI)
     - Moving Average Convergence Divergence (MACD)
     - Williams %R (WILLR)
     - Commodity Channel Index (CCI)
     - Average Directional Index (ADX)
     - Bollinger Bands (BBANDS)
   - Computes statistical features:
     - Min/max values and their timesteps
     - Mean, median values and timesteps
     - Trend indicators
     - Lag correlations
3. Structures data into sequences for input (X) and prediction targets (Y)
4. Splits data into train, validation, and test sets
5. Outputs processed data to CSV files for each stock and split type in the format:
   - `{STOCK}_train_x.csv`, `{STOCK}_train_y.csv`
   - `{STOCK}_test_x.csv`, `{STOCK}_test_y.csv`
   - `{STOCK}_full_x.csv`, `{STOCK}_full_y.csv`

This processed data serves as the foundation for the GRPO training pipeline. Run:
```
python combined_preprocessor.py \
  --root_path backbone/data/stocknet \
  --output_dir grpo/data/stocknet
```

## Training Pipeline

The GRPO pipeline consists of three primary stages, plus an intermediate data-generation step, combining reinforcement learning and supervised fine-tuning:

### Stage 1: Initial GRPO Training (Reinforcement Learning)
- **Script**: `stage1_train_grpo.py`
- **Input**: Processed stock data (train_x, train_y, test_x, test_y)
- **Process**: 
  - Initializes a base LLM (e.g., `meta-llama/Llama-3.2-1B-Instruct`)
  - Applies LoRA (Low-Rank Adaptation) for efficient fine-tuning
  - Trains using GRPO, a reinforcement learning approach that optimizes the model to predict future price movements
  - Uses reward functions based on prediction accuracy and financial metrics
- **Output**: First GRPO LoRA weights saved to `grpo-saved-lora-1st`

### Stage 1.5: SFT Data Generation (Rejection Sampling)
- **Script**: `sft_data_generation.py`
- **Input**: 
  - Processed stock data
  - First GRPO model from Stage 1
- **Process**:
  - Performs rejection sampling by generating multiple predictions for each time window
  - Selects the best predictions based on accuracy metrics
  - Creates a high-quality dataset for supervised fine-tuning
- **Output**: SFT training dataset (`{STOCK}_sft_training_data.csv`)

### Stage 2: Supervised Fine-Tuning (SFT)
- **Script**: `stage2_train_sft.py`
- **Input**: 
  - SFT dataset generated in Stage 2
  - Base LLM model
- **Process**:
  - Fine-tunes the base model using supervised learning
  - Trains the model to directly predict the target values from the high-quality examples
  - Uses cross-entropy loss rather than RL objectives
- **Output**: SFT LoRA weights saved to `sft-saved-lora`

### Stage 3: Second GRPO Training (RL from SFT)
- **Script**: `stage3_train_grpo_from_sft.py`
- **Input**: 
  - Processed stock data
  - SFT model from Stage 3
- **Process**:
  - Initializes from the SFT model rather than the base model
  - Further optimizes using GRPO with reinforcement learning
  - Combines the benefits of supervised learning and RL
- **Output**: Second GRPO LoRA weights saved to `grpo-saved-lora-2nd`

### Model Comparison and Evaluation
- **Script**: `compare_models.py`
- **Input**: All three trained models (First GRPO, SFT, Second GRPO)
- **Process**:
  - Evaluates all models on test data
  - Compares prediction accuracy, financial metrics, and other performance indicators
  - Generates visualizations and comparison reports
- **Output**: Comprehensive evaluation results saved to the results directory

## Running the Pipeline

The entire pipeline can be executed using the `run_pipeline.sh` script:

```bash
./run_pipeline.sh
```

You can modify the configuration parameters at the top of the script:

```bash
# Model Configuration
STOCK="AAPL"
MODEL="Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH=2048
LORA_RANK=32
CUDA_DEVICES="1,0,2,3,4,5,6,7"
GPU_MEMORY_UTILIZATION=0.4
```

## Pipeline Configuration

The pipeline is highly configurable with the following key parameters:
- `STOCK`: Target stock symbol (e.g., "AAPL")
- `MODEL`: Base LLM to use (e.g., "meta-llama/Llama-3.2-1B-Instruct")
- `MAX_SEQ_LENGTH`: Maximum sequence length for the model
- `LORA_RANK`: Rank parameter for LoRA fine-tuning
- `CUDA_DEVICES`: GPU device mapping
- `GPU_MEMORY_UTILIZATION`: Memory utilization fraction

## Output Structure

All training outputs are organized in a timestamped directory structure:
```
results/
└── {STOCK}_{TIMESTAMP}/
    ├── stage1_grpo_training/
    ├── sft_data_generation/
    │   └── {STOCK}_sft_training_data.csv
    ├── stage2_sft_training/
    ├── stage3_grpo_from_sft/
    ├── model_comparison/
    └── pipeline.log
```

Model weights are stored separately:
```
models/
└── {MODEL}/
    └── {STOCK}/
        ├── grpo_lora_1st/
        ├── sft_lora/
        └── grpo_lora_2nd/
```

## Methodology

The GRPO approach combines the strengths of reinforcement learning and supervised learning:

1. **Initial RL Training**: Train the base model with reinforcement learning to optimize financial performance metrics
2. **Rejection Sampling**: Generate multiple predictions and select the best examples
3. **Supervised Fine-Tuning**: Train on high-quality examples with standard supervised learning
4. **Second RL Stage**: Further optimize the SFT model with reinforcement learning
5. **Comprehensive Evaluation**: Compare all models to determine the most effective approach

This innovative approach creates models that can effectively forecast financial time series data with improved accuracy and reliability.

## Prerequisites

- Python 3.8+
- PyTorch
- Unsloth
- vLLM
- TRL (Transformer Reinforcement Learning)
- Pandas, NumPy, Matplotlib
- Hugging Face Transformers

## Installation

```bash
pip install unsloth vllm pandas numpy matplotlib torch transformers trl datasets
```

## Usage

### Basic Usage

To run the complete pipeline with default settings:

```bash
./run_pipeline.sh
```

### Custom Configuration

You can customize the pipeline by passing command-line arguments:

```bash
./run_pipeline.sh \
  --stock AAPL \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-seq-length 2048 \
  --lora-rank 32 \
  --gpu-memory-utilization 0.4 \
  --cuda-devices 0,1,2,3
```

### Available Options

- `--stock`: Stock symbol (default: AAPL)
- `--model`: Hugging Face model name (default: meta-llama/meta-Llama-3.2-8B-Instruct)
- `--max-seq-length`: Maximum sequence length (default: 2048)
- `--lora-rank`: LoRA rank (default: 32)
- `--gpu-memory-utilization`: GPU memory utilization (default: 0.4)
- `--cuda-devices`: CUDA visible devices (default: 0,2,1,3,7,6,5,4)

### Running Individual Stages

You can also run individual stages separately:

```bash
# Example: Run Stage 1 only
python stage1_train_grpo.py \
  --stock AAPL \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-seq-length 2048 \
  --lora-rank 32 \
  --gpu-memory-utilization 0.4 \
  --train-x-path data/AAPL/train_x.csv \
  --train-y-path data/AAPL/train_y.csv \
  --test-x-path data/AAPL/test_x.csv \
  --test-y-path data/AAPL/test_y.csv \
  --full-x-path data/AAPL/full_x.csv \
  --full-y-path data/AAPL/full_y.csv \
  --grpo-saved-lora-1st models/grpo_lora_1st.pt \
  --grpo-saved-lora-2nd models/grpo_lora_2nd.pt \
  --sft-saved-lora models/sft_lora.pt \
  --results-dir results/AAPL
```

## Data Format

The pipeline expects CSV files with specific formats:

- `train_x.csv`, `test_x.csv`, `full_x.csv`: Input features including historical prices and technical indicators
- `train_y.csv`, `test_y.csv`, `full_y.csv`: Ground truth future prices

Each X-file should include columns for:
- `sequence`: Historical prices
- Technical indicators: `sma`, `ema`, `mom`, `rsi`, `macd`, `willr`, `cci`, `adx`, `bbands_upper`, `bbands_middle`, `bbands_lower`, `stoch_k`, `stoch_d`
- Statistics: `min`, `min_timestep`, `max`, `max_timestep`, `mean`, `median`, `median_timestep`

Each Y-file should include a column:
- `sequence`: Future prices (10 values representing the next 10 days)

## Directory Structure

```
.
├── grpo_utils.py                   # Utility functions shared across stages
├── run_pipeline.sh                 # Shell script to run the complete pipeline
├── stage1_train_grpo.py                 # Stage 1: Train first GRPO model
├── sft_data_generation.py                # Stage 1.5: Generate SFT data
├── stage2_train_sft.py                  # Stage 2: Train SFT model
├── stage3_train_grpo_from_sft.py        # Stage 3: Train second GRPO model from SFT
├── compare_models.py                    # Compare models (optional)
└── README.md                       # This file
```

## Results

The pipeline saves all results in a timestamped directory under `results/STOCK_TIMESTAMP/`. This includes:

- Trained model weights (LoRA adapters)
- Evaluation metrics and visualizations
- SFT dataset
- Detailed logs
- Comparison results

## Acknowledgments

This pipeline is based on research in RLHF, GRPO, and financial time series forecasting.

- Unsloth: https://github.com/unslothai/unsloth
- vLLM: https://github.com/vllm-project/vllm
- TRL: https://github.com/huggingface/trl 