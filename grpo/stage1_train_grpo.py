import os
import torch
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
import sys

from grpo_utils import (
    get_argparser, 
    load_model, 
    add_lora_adapters, 
    get_stock_prediction_data,
    correctness_reward_func,
    think_format_reward_func,
    answer_format_reward_func,
    xmlcount_reward_func
)

def main():
    parser = get_argparser()
    args = parser.parse_args()
    
    # output directory
    results_dir = args.results_dir
    stage1_dir = os.path.join(results_dir, "stage1_grpo_training")
    os.makedirs(stage1_dir, exist_ok=True)
    
    print("="*80)
    print(f"STAGE 1: TRAINING GRPO MODEL FOR {args.stock}")
    print("="*80)
    
    print(f"Model: {args.model}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Results will be saved to: {stage1_dir}")
    
    # CUDA is availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)
    
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Current device index: {torch.cuda.current_device()}")
    
    model, tokenizer = load_model(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # LoRA adapters
    model = add_lora_adapters(model, args.lora_rank)
    
    # training dataset
    print("Loading training dataset...")
    dataset = get_stock_prediction_data(args.train_x_path, args.train_y_path)
    print(f"Loaded dataset with {len(dataset)} examples")
    
    # training arguments
    training_args = GRPOConfig(
        use_vllm=True,  # use vLLM for fast inference
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
        save_steps=250,
        report_to="wandb",
        output_dir=stage1_dir,
    )
    
    # trainer initialization
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
    
    # model training
    print("Starting GRPO training...")
    trainer.train()
    
    # save LoRA weights
    print(f"Saving trained LoRA weights to: {args.grpo_saved_lora_1st}")
    model.save_lora(args.grpo_saved_lora_1st)
    
    print("="*80)
    print("STAGE 1 COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"LoRA weights saved to: {args.grpo_saved_lora_1st}")
    print(f"Training logs and outputs saved to: {stage1_dir}")

if __name__ == "__main__":
    main() 