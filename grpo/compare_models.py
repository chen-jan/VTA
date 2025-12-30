#!/usr/bin/env python3

import os
import torch
import numpy as np
import pandas as pd
import sys
import gc
from unsloth import FastLanguageModel, is_bfloat16_supported

# Import utilities from grpo_utils
from grpo_utils import (
    get_argparser, 
    get_stock_prediction_data,
    compare_all_models_on_test_set
)


def main():
    # Parse arguments
    parser = get_argparser()
    args = parser.parse_args()
    
    # Set up output directory
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_dir = args.results_dir or f"results/{args.stock}_{timestamp}"
    compare_dir = os.path.join(results_dir, "model_comparison")
    os.makedirs(compare_dir, exist_ok=True)
    
    print("="*80)
    print(f"COMPARING ALL MODELS FOR {args.stock}")
    print("="*80)
    
    print(f"Model: {args.model}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Results will be saved to: {compare_dir}")
    
    # Verify CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)
    
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Current device index: {torch.cuda.current_device()}")
    
    # Clear CUDA cache before loading
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load the base model directly using FastLanguageModel
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Add LoRA adapters (required to load LoRA weights)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        peft_type="lora",
    )
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = get_stock_prediction_data(args.test_x_path, args.test_y_path)
    print(f"Loaded test dataset with {len(test_dataset)} examples")
    
    # Set up LoRA paths dictionary
    lora_paths = {
        "grpo_1st": args.grpo_saved_lora_1st,
        "sft": args.sft_saved_lora,
        "grpo_2nd": args.grpo_saved_lora_2nd
    }
    
    # Compare all models on test set
    print("Comparing all models on test set...")
    results_df = compare_all_models_on_test_set(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        lora_paths=lora_paths,
        num_examples=5,
        batch_size=60,
        visualize=True,
        output_dir=compare_dir
    )
    
    print("="*80)
    print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Comparison results saved to: {compare_dir}")


if __name__ == "__main__":
    main()


