#!/usr/bin/env python3
# Stage 2: Train the SFT model on selected examples from GRPO1

import os
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

# Import utilities from grpo_utils
from grpo_utils import get_argparser, SYSTEM_PROMPT


def main():
    # Parse arguments
    parser = get_argparser()
    args = parser.parse_args()
    
    # Set up output directory
    results_dir = args.results_dir
    stage_dir = os.path.join(results_dir, "stage2_sft_training")
    os.makedirs(stage_dir, exist_ok=True)

    # Use provided path or default
    sft_data_path = args.sft_data_path
    
    if not os.path.exists(sft_data_path):
        print(f"Error: SFT data not found at {sft_data_path}")
        print("Please run stage 1.5 first or specify a valid path with --sft-data-path")
        return
    
    print("="*80)
    print(f"STAGE 2: TRAINING SFT MODEL FOR {args.stock}")
    print("="*80)
    
    print(f"Model: {args.model}")
    print(f"SFT dataset: {sft_data_path}")
    print(f"Results will be saved to: {stage_dir}")
    
    # Verify CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Current device index: {torch.cuda.current_device()}")
    
    # Model configuration
    max_seq_length = args.max_seq_length
    lora_rank = args.lora_rank
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Load and prepare dataset
    print(f"Loading SFT dataset from: {sft_data_path}")
    df = pd.read_csv(sft_data_path)
    dataset = Dataset.from_pandas(df)
    print(f"Loaded dataset with {len(dataset)} examples")
    
    def formatting_prompts_func(examples):
        prompts = examples["prompt_question"]
        responses = examples["full_answer"]
        texts = []
        for prompt, response in zip(prompts, responses):
            text = f"{prompt}\n\nResponse:\n{response}{tokenizer.eos_token}"
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    
    # Create completion-only collator
    # Get token IDs directly from a sample of the dataset with context
    sample_text = f"{dataset[0]['prompt_question']}\n\nResponse:\n{dataset[0]['full_answer']}"
    sample_tokens = tokenizer.encode(sample_text)
    
    # Find the position of "\nResponse:\n" in the tokenized text
    response_token_ids = None
    for i in range(len(sample_tokens)):
        decoded = tokenizer.decode(sample_tokens[i:i+10])
        if "\nResponse:\n" in decoded:
            # Get exact number of tokens for the response template
            for j in range(i, len(sample_tokens)):
                subtext = tokenizer.decode(sample_tokens[i:j])
                if subtext.endswith("\nResponse:\n"):
                    response_token_ids = sample_tokens[i:j]
                    break
            break
    
    # Fallback if we couldn't find the response template
    if response_token_ids is None:
        print("Warning: Could not find response template in sample text.")
        print("Using tokenized response template directly.")
        response_token_ids = tokenizer.encode("\nResponse:\n", add_special_tokens=False)
    
    print(f"Using response template token IDs: {response_token_ids}")
    print(f"Decoded: {tokenizer.decode(response_token_ids)}")
    
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_token_ids,
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=stage_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="wandb",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=False,
        data_collator=collator,
        args=training_args,
    )
    
    # Training
    print("Starting SFT training...")
    trainer_stats = trainer.train()
    
    # Save the model
    print(f"Saving trained LoRA weights to: {args.sft_saved_lora}")
    model.save_pretrained(args.sft_saved_lora)
    tokenizer.save_pretrained(args.sft_saved_lora)
    
    # Show training stats
    training_time_seconds = trainer_stats.metrics['train_runtime']
    training_time_minutes = round(training_time_seconds/60, 2)
    print(f"{training_time_seconds} seconds used for training.")
    print(f"{training_time_minutes} minutes used for training.")
    
    print("="*80)
    print("STAGE 2 COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"LoRA weights saved to: {args.sft_saved_lora}")
    print(f"Training logs and outputs saved to: {stage_dir}")


if __name__ == "__main__":
    main() 


