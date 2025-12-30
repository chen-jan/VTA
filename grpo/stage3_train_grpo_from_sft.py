#!/usr/bin/env python3
# Stage 3: Train the second GRPO model from SFT

import os
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
import sys
import gc

# Import utilities from grpo_utils
from grpo_utils import (
    get_argparser, 
    get_stock_prediction_data,
    correctness_reward_func,
    think_format_reward_func,
    answer_format_reward_func,
    xmlcount_reward_func,
    evaluate_grpo_on_full_dataset
)


def main():
    # Parse arguments
    parser = get_argparser()
    args = parser.parse_args()
    
    # Set up output directory
    results_dir = args.results_dir
    stage_dir = os.path.join(results_dir, "stage3_grpo_from_sft")
    os.makedirs(stage_dir, exist_ok=True)
    
    print("="*80)
    print(f"STAGE 3: TRAINING SECOND GRPO MODEL FROM SFT FOR {args.stock}")
    print("="*80)
    
    print(f"Model: {args.model}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"SFT LoRA weights: {args.sft_saved_lora}")
    print(f"Results will be saved to: {stage_dir}")
    
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
    
    # Load the model directly from the SFT LoRA path
    try:
        print(f"Loading model directly from SFT LoRA adapter path: {args.sft_saved_lora}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.sft_saved_lora,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
        print("Successfully loaded model with SFT LoRA adapters")
        print("Will continue GRPO training on the same adapters")
    except Exception as e:
        print(f"Error loading model from SFT LoRA path: {e}")
        print("Cannot proceed without SFT LoRA weights. Exiting.")
        print("Please ensure the SFT LoRA path is correct and the weights are properly saved.")
        print(f"Attempted to load from: {args.sft_saved_lora}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        sys.exit(1)
    
    # Load training dataset
    print("Loading training dataset...")
    dataset = get_stock_prediction_data(args.train_x_path, args.train_y_path)
    print(f"Loaded dataset with {len(dataset)} examples")
    
    # Set up training arguments
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_prompt_length=500,
        max_completion_length=500,
        max_steps=750,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=stage_dir,
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            think_format_reward_func,
            answer_format_reward_func,
            correctness_reward_func,
            xmlcount_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train the model
    print("Starting GRPO training from SFT model...")
    trainer.train()
    
    # Save the trained LoRA weights
    print(f"Saving trained LoRA weights to: {args.grpo_saved_lora_2nd}")
    model.save_lora(args.grpo_saved_lora_2nd)
    
    # Evaluate the GRPO model on the full dataset
    print("\n" + "="*80)
    print("EVALUATING GRPO MODEL ON FULL DATASET")
    print("="*80)
    
    # Load the full dataset
    print("Loading full dataset for evaluation...")
    full_dataset = get_stock_prediction_data(args.full_x_path, args.full_y_path)
    print(f"Loaded full dataset with {len(full_dataset)} examples")
    
    # Create output directory for evaluation results
    eval_output_dir = os.path.join(stage_dir, "full_dataset_evaluation")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Run evaluation
    print(f"Starting evaluation of GRPO model on full dataset...")
    
    # Clear CUDA cache before evaluation
    gc.collect()
    torch.cuda.empty_cache()
    
    results_df = evaluate_grpo_on_full_dataset(
        model=model,
        tokenizer=tokenizer,
        full_dataset=full_dataset,
        grpo_lora_path=args.grpo_saved_lora_2nd,
        num_examples=5,
        batch_size=60,
        visualize=True,
        output_dir=eval_output_dir
    )
    
    # Save GRPO2 full dataset predictions
    grpo2_full_data_path = os.path.join(results_dir, "grpo2_full_dataset_results.csv")
    results_df.to_csv(grpo2_full_data_path, index=False)
    print(f"Saved GRPO 2nd model full dataset predictions to: {grpo2_full_data_path}")
    
    print("="*80)
    print("STAGE 3 COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"LoRA weights saved to: {args.grpo_saved_lora_2nd}")
    print(f"Training logs and outputs saved to: {stage_dir}")
    print(f"Full dataset evaluation results saved to: {eval_output_dir}")
    print(f"GRPO 2nd model full dataset predictions saved to: {grpo2_full_data_path}")


if __name__ == "__main__":
    main() 


