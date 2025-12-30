#!/usr/bin/env python3
# Stage 1.5: Generate SFT data from the full dataset using the first GRPO model

import os
import torch
import pandas as pd
import sys

# Import utilities from grpo_utils
from grpo_utils import (
    get_argparser, 
    load_model, 
    add_lora_adapters, 
    get_stock_prediction_data,
    extract_xml_answer,
    extract_historical_data, 
    perform_rejection_sampling,
    evaluate_grpo_on_full_dataset,
    SYSTEM_PROMPT
)


def main():
    # Parse arguments
    parser = get_argparser()
    args = parser.parse_args()
    
    # Set up output directory
    results_dir = args.results_dir
    stage_dir = os.path.join(results_dir, "stage1_5_sft_data_generation")
    os.makedirs(stage_dir, exist_ok=True)
    
    print("="*80)
    print(f"STAGE 1.5: GENERATING SFT DATA FOR {args.stock}")
    print("="*80)
    
    print(f"Model: {args.model}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"LoRA weights: {args.grpo_saved_lora_1st}")
    print(f"Results will be saved to: {stage_dir}")
    
    # Verify CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)
    
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Current device index: {torch.cuda.current_device()}")
    
    # Load the model
    model, tokenizer = load_model(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Add LoRA adapters (required before loading LoRA weights)
    model = add_lora_adapters(model, args.lora_rank)
    
    # Load full dataset
    print("Loading full dataset...")
    train_dataset = get_stock_prediction_data(args.train_x_path, args.train_y_path)
    print(f"Loaded full dataset with {len(train_dataset)} examples")
    
    # Evaluate the GRPO model on the full dataset to generate SFT data
    print(f"Evaluating GRPO model with LoRA weights: {args.grpo_saved_lora_1st}")
    evaluation_df = evaluate_grpo_on_full_dataset(
        model=model,
        tokenizer=tokenizer,
        full_dataset=train_dataset,
        grpo_lora_path=args.grpo_saved_lora_1st,
        num_examples=5,
        batch_size=60,
        visualize=True,
        output_dir=stage_dir
    )
    
    # Perform rejection sampling to select the best examples for SFT
    print("Performing rejection sampling to select examples for SFT...")
    evaluation_stats_path = os.path.join(stage_dir, "evaluation_stats.csv")
    sft_dataset = perform_rejection_sampling(evaluation_stats_path, num_buckets=100, top_k=10)

    sft_data_path = args.sft_data_path
        
    sft_dataset.to_csv(args.sft_data_path, index=False)
    
    # Verify file was saved successfully
    if os.path.exists(sft_data_path):
        file_size_bytes = os.path.getsize(sft_data_path)
        print(f"SFT dataset successfully saved to: {sft_data_path} ({file_size_bytes} bytes)")
    else:
        print(f"ERROR: Failed to save SFT dataset to {sft_data_path}!")
    
    print("="*80)
    print("STAGE 1.5 COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Evaluation results saved to: {stage_dir}")
    print(f"SFT dataset saved to: {sft_data_path}")
    print(f"Selected {len(sft_dataset)} examples for SFT training")


if __name__ == "__main__":
    main() 


