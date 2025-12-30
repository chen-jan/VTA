import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
try:
    from vllm import SamplingParams
except ImportError:
    # Fallback if vllm is not available
    class SamplingParams:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
import argparse

# system prompt for stock prediction
SYSTEM_PROMPT = """
You are a helpful stock prediction assistant. You are given a 10-day series of standardized stock prices along with various statistics and technical indicators. Your goal is to think and reason about the data and predict the closing prices for the next 10 days. Your prediction should be a sequence of 10 values.

Respond in the following format:
<think>
...
</think>
<answer>
[value1, value2, value3, value4, value5, value6, value7, value8, value9, value10]
</answer>
"""

def get_argparser():
    """arg parser shared by grpo/sft stages"""
    parser = argparse.ArgumentParser(description='Run the GRPO-SFT pipeline')
    parser.add_argument('--stock', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--model', type=str, default='meta-llama/meta-Llama-3.2-8B-Instruct', help='Model name')
    parser.add_argument('--max-seq-length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--lora-rank', type=int, default=32, help='LoRA rank')
    parser.add_argument('--train-x-path', type=str, required=True, help='Path to training X data')
    parser.add_argument('--train-y-path', type=str, required=True, help='Path to training Y data')
    parser.add_argument('--test-x-path', type=str, required=True, help='Path to test X data')
    parser.add_argument('--test-y-path', type=str, required=True, help='Path to test Y data')
    parser.add_argument('--full-x-path', type=str, required=True, help='Path to full X data')
    parser.add_argument('--full-y-path', type=str, required=True, help='Path to full Y data')
    parser.add_argument('--grpo-saved-lora-1st', type=str, required=True, help='Path to save first GRPO LoRA weights')
    parser.add_argument('--grpo-saved-lora-2nd', type=str, required=True, help='Path to save second GRPO LoRA weights')
    parser.add_argument('--sft-saved-lora', type=str, required=True, help='Path to save SFT LoRA weights')
    parser.add_argument('--sft-data-path', type=str, help='Path to SFT dataset')
    parser.add_argument('--cuda-devices', type=str, default='0,2,1,3,7,6,5,4', help='CUDA visible devices')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.4, help='GPU memory utilization (0.0 to 1.0)')
    parser.add_argument('--results-dir', type=str, help='Base directory for saving results')
    return parser

def load_model(model_name, max_seq_length, lora_rank, gpu_memory_utilization, retry_higher=True):
    """Load the base model with error handling for memory issues"""
    print(f"Loading model: {model_name}")
    print(f"Requesting GPU memory utilization: {gpu_memory_utilization}")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=lora_rank,
            gpu_memory_utilization=gpu_memory_utilization
        )
        return model, tokenizer
    
    except ValueError as e:
        if "No available memory for the cache blocks" in str(e) and retry_higher:
            print("Initial utilization failed. Trying higher utilization...")
            higher_utilization = 0.8  # Try with higher utilization
            print(f"Retrying with GPU memory utilization: {higher_utilization}")
            torch.cuda.empty_cache()  # Clear cache before retry
            return load_model(model_name, max_seq_length, lora_rank, higher_utilization, retry_higher=False)
        else:
            raise e  # Re-raise other errors

def add_lora_adapters(model, lora_rank):
    """add lora adapters"""
    print("Initializing LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        peft_type="lora",
    )
    return model

def visual(true, pred, path):
    """plot pred vs truth"""
    plt.figure()
    plt.plot(true, label='Ground Truth', linewidth=2)
    plt.plot(pred, label='Prediction', linewidth=2)
    plt.legend()
    
    # Save the plot
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def extract_xml_answer(text: str) -> dict:
    """parse <answer>[...] into list[float] of len 10"""
    if "<answer>" not in text or "</answer>" not in text:
        return None
    
    answer_text = text.split("<answer>")[-1].split("</answer>")[0].strip()
    
    # Extract the sequence from the answer text
    # Looking for array format [val1, val2, val3, ...]
    sequence_match = re.search(r"\[(.*?)\]", answer_text)
    
    if not sequence_match:
        return None
        
    # Get the sequence string and split by commas
    sequence_str = sequence_match.group(1)
    try:
        # Try to parse as a list of floats
        sequence_values = [float(val.strip()) for val in sequence_str.split(',')]
        
        # Check if we have exactly 10 values
        if len(sequence_values) != 10:
            return None
            
        return {
            'sequence': sequence_values
        }
    except (ValueError, TypeError):
        return None

def format_array(arr):
    """format arrays/lists with compact floating output"""
    if arr is None:
        return "N/A"
    
    # Handle both numpy arrays and lists
    if isinstance(arr, list):
        arr = np.array(arr)
    
    # Format each number to a consistent format with 4 significant digits
    def format_number(x):
        return f"{x:.4g}"
    
    # Custom formatter with controlled width
    formatter = {'float_kind': format_number}
    with np.printoptions(formatter=formatter, threshold=np.inf, suppress=True):
        return np.array2string(arr, separator=', ')

def safe_parse_array(value):
    """parse '[a, b, ...]' -> list[float]; else passthrough"""
    if isinstance(value, str):
        if value.startswith('[') and value.endswith(']'):
            # Remove brackets and split by commas
            content = value.strip('[]')
            # Handle empty array
            if not content:
                return []
            # Split and convert to float
            try:
                return [float(x.strip()) for x in content.split(',')]
            except ValueError:
                # If parsing fails, return the original string
                return value
    return value

def safe_extract_float(value):
    """get float from value, handling strings like '[1.23]'"""
    if isinstance(value, str):
        # If it's a string with brackets like '[1.23]'
        if value.startswith('[') and value.endswith(']'):
            # Extract content between brackets
            content = value[1:-1].split(',')[0].strip()
            # Handle empty array
            if not content:
                return 0.0
            return float(content)
        # If it's a plain string number
        try:
            return float(value)
        except ValueError:
            return 0.0
    # If it's already a number
    elif isinstance(value, (int, float)):
        return float(value)
    # If it's something else (like a list with one element)
    elif isinstance(value, list) and len(value) > 0:
        return float(value[0])
    return 0.0

def safe_extract_int(value):
    """int wrapper around safe_extract_float"""
    return int(safe_extract_float(value))

def format_input_stats(stats_row, simplified=True):
    """Convert X statistics to natural language prompt with technical indicators.
    If simplified is True, only include the closing prices and the prediction instruction.
    Otherwise, include full statistics and technical analysis."""
    if simplified:
        prompt = """
Stock closing prices for the past 10 days:
{}

Based on these data, predict the closing prices for the next 10 days.
""".format(
            format_array(safe_parse_array(stats_row['sequence']))
        )
        return prompt
    else:
        prompt = """
Stock closing prices for the past 10 days:
{}

Stock statistics:
- Minimum price: {:.4g} on day {}
- Maximum price: {:.4g} on day {}
- Mean price: {:.4g}
- Median price: {:.4g} on day {}

Technical Analysis:
- Simple Moving Average: {}
- Exponential Moving Average: {}
- Momentum: {}
- Relative Strength Index: {}
- MACD Line: {}
- Williams %R: {}
- Commodity Channel Index: {}
- Average Directional Index: {}
- Bollinger Bands Upper: {}
- Bollinger Bands Middle: {}
- Bollinger Bands Lower: {}
- Stochastic %K: {}
- Stochastic %D: {}

Based on these data, predict the closing prices for the next 10 days.
""".format(
            format_array(safe_parse_array(stats_row['sequence'])),
            safe_extract_float(stats_row['min']), safe_extract_int(stats_row['min_timestep']),
            safe_extract_float(stats_row['max']), safe_extract_int(stats_row['max_timestep']),
            safe_extract_float(stats_row['mean']),
            safe_extract_float(stats_row['median']), safe_extract_int(stats_row['median_timestep']),
            format_array(safe_parse_array(stats_row['sma'])),
            format_array(safe_parse_array(stats_row['ema'])),
            format_array(safe_parse_array(stats_row['mom'])),
            format_array(safe_parse_array(stats_row['rsi'])),
            format_array(safe_parse_array(stats_row['macd'])),
            format_array(safe_parse_array(stats_row['willr'])),
            format_array(safe_parse_array(stats_row['cci'])),
            format_array(safe_parse_array(stats_row['adx'])),
            format_array(safe_parse_array(stats_row['bbands_upper'])),
            format_array(safe_parse_array(stats_row['bbands_middle'])),
            format_array(safe_parse_array(stats_row['bbands_lower'])),
            format_array(safe_parse_array(stats_row['stoch_k'])),
            format_array(safe_parse_array(stats_row['stoch_d']))
        )
        return prompt

def format_output_stats(stats_row):
    """format y to <answer>[...]"""
    return "<answer>[{}]</answer>".format(
        ", ".join(f"{x:.4f}" for x in safe_parse_array(stats_row['sequence']))
    )

def extract_historical_data(prompt_content):
    """pull historical closes from prompt string"""
    # Find the sequence in the prompt that contains historical prices
    sequence_match = re.search(r"Stock closing prices for the past 10 days:\n(.*?)\n\nStock statistics:", 
                             prompt_content, re.DOTALL)
    if sequence_match:
        sequence_str = sequence_match.group(1).strip()
        # Convert string representation of array to actual values
        try:
            # Remove brackets and split by commas
            values = [float(x.strip()) for x in sequence_str.strip('[]').split(',')]
            return np.array(values)
        except:
            return np.array([])
    return np.array([])

# load stock data from csv
def get_stock_prediction_data(x_path, y_path):
    """load x/y csv -> Dataset of {prompt, answer}"""
    try:
        # Load CSV files
        x_stats = pd.read_csv(x_path)
        y_stats = pd.read_csv(y_path)
        
        print(f"Loaded {len(x_stats)} rows from {x_path}")
        print(f"Loaded {len(y_stats)} rows from {y_path}")
        
        # keep within common length
        min_rows = min(len(x_stats), len(y_stats))
        if min_rows == 0:
            raise ValueError(f"No data found in one or both CSV files: {x_path}, {y_path}")
        
        # build dataset entries
        data = []
        for i in range(min_rows):
            try:
                # build user prompt
                prompt_content = format_input_stats(x_stats.iloc[i])
                
                # expected output
                answer = format_output_stats(y_stats.iloc[i])
                
                data.append({
                    'prompt': [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': prompt_content}
                    ],
                    'answer': answer
                })
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue
        
        print(f"Successfully created dataset with {len(data)} examples")
        return Dataset.from_list(data)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # empty dataset fallback
        return Dataset.from_list([])

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    
    # extract answers
    extracted_responses = []
    for r in responses:
        extracted = extract_xml_answer(r)
        extracted_responses.append(extracted)
    
    # extract ground truths
    ground_truths = []
    for a in answer:
        if "<answer>" in a and "</answer>" in a:
            ground_truth = extract_xml_answer(a)
            ground_truths.append(ground_truth)
        else:
            ground_truths.append(None)
    
    # debug: first example
    print('-'*20)
    print(f"Question:\n{q}")
    print(f"\nGround Truth:\n{answer[0]}")
    print(f"\nResponse:\n{responses[0]}")
    print(f"\nExtracted Response:\n{extracted_responses[0]}")
    print(f"\nExtracted Ground Truth:\n{ground_truths[0]}")
    
    # mse only if both parsed
    if extracted_responses[0] is not None and ground_truths[0] is not None:
        mse = np.mean([(p - t)**2 for p, t in zip(extracted_responses[0]['sequence'], ground_truths[0]['sequence'])])
        print(f"\nMSE:\n{mse}\n")
    else:
        print("\nCould not calculate MSE: Invalid response or ground truth format\n")
    
    # rewards
    rewards = []
    for resp, truth in zip(extracted_responses, ground_truths):
        if resp is None or truth is None:
            rewards.append(0.0)
            continue
        
        # reward via mse(pred, true)
        reward = 0.0
        try:
            # Get sequences
            pred_sequence = resp['sequence']
            true_sequence = truth['sequence']
            
            # Calculate MSE
            mse = np.mean([(p - t)**2 for p, t in zip(pred_sequence, true_sequence)])
            
            # Convert MSE to reward (lower MSE = higher reward)
            # Scale to 0-2 range (2 is perfect, 0 is bad)
            reward = 5 * 1 / (1 + 10 * mse)
        except (ValueError, TypeError, KeyError):
            # If any calculation fails, give zero reward
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards

def think_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the think-answer format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def answer_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the answer section has the correct format with a sequence of 10 values."""
    responses = [completion[0]["content"] for completion in completions]
    
    def check_format(response):
        if "<answer>" not in response or "</answer>" not in response:
            return 0.0
        
        answer_text = response.split("<answer>")[-1].split("</answer>")[0].strip()
        
        # Check if there's a sequence of 10 numbers
        sequence_match = re.search(r"\[(.*?)\]", answer_text)
        if not sequence_match:
            return 0.0
            
        sequence_str = sequence_match.group(1)
        try:
            values = [float(val.strip()) for val in sequence_str.split(',')]
            # Full score if exactly 10 values
            if len(values) == 10:
                return 1.0
            # Partial score based on how close to 10
            else:
                return max(0.0, 1.0 - abs(len(values) - 10) / 10)
        except (ValueError, TypeError):
            return 0.0
    
    return [check_format(r) for r in responses]

def count_xml(text) -> float:
    """count simple xml markers to nudge structure"""
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
    if text.count("\n</answer>") == 1:
        count += 0.125
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """reward: partial credit for xml markers presence"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def evaluate_model_on_dataset(model, tokenizer, dataset, lora_weights=None, num_examples=5, batch_size=60, visualize=False, output_dir=None):
    """evaluate model on dataset and log mse/mae; optional lora"""
    if output_dir is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"evaluation_results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # file outputs
    detailed_output_file = f"{output_dir}/evaluation_details.txt"
    stats_file = f"{output_dir}/evaluation_stats.csv"
    
    with open(detailed_output_file, "w") as f_out:
        def log_print(*args, **kwargs):
            print(*args, **kwargs)  # Print to console
            print(*args, **kwargs, file=f_out)  # Print to file
            
        log_print("\n" + "="*80)
        log_print("EVALUATING MODEL ON DATASET")
        log_print("="*80)
        
        # Convert dataset to list for easier batching
        dataset_list = [dataset[i] for i in range(len(dataset))]
        
        # sampling params
        sampling_params = SamplingParams(
            temperature=0.2,  # Lower temperature for more deterministic outputs
            top_p=0.95,
            max_tokens=2048,
        )
        
        # Load LoRA weights if provided
        if lora_weights:
            log_print(f"Loading LoRA weights: {lora_weights}")
            lora = model.load_lora(lora_weights)
        else:
            lora = None
        
        # metrics
        model_mse = []
        model_mae = []
        
        # collect per-example results
        all_results = []
        
        # iterate batches
        for batch_start in range(0, len(dataset_list), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset_list))
            batch = dataset_list[batch_start:batch_end]
            
            log_print(f"\nProcessing batch {batch_start//batch_size + 1}, examples {batch_start} to {batch_end}")
            
            # prep inputs
            batch_texts = []
            batch_ground_truths = []
            
            for example in batch:
                try:
                    prompt_content = example['prompt'][1]['content']
                    ground_truth = extract_xml_answer(example['answer'])
                    
                    if not ground_truth:
                        continue
                    
                    text = tokenizer.apply_chat_template([
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_content},
                    ], tokenize=False, add_generation_prompt=True)
                    
                    batch_texts.append(text)
                    batch_ground_truths.append(ground_truth)
                except Exception as e:
                    log_print(f"Error preparing example: {str(e)}")
                    continue
            
            if not batch_texts:
                log_print("No valid examples in batch")
                continue
            
            log_print(f"Processing {len(batch_texts)} valid examples in current batch")
            
            try:
            # batch gen
                outputs = model.fast_generate(
                    batch_texts,
                    sampling_params=sampling_params,
                    lora_request=lora,
                )
                
                # handle outputs
                for i, (output, ground_truth) in enumerate(zip(outputs, batch_ground_truths)):
                    try:
                        # Extract outputs
                        text = output.outputs[0].text
                        
                        # Extract prediction
                        extracted = extract_xml_answer(text)
                        
                        if extracted and 'sequence' in extracted:
                            sequence = extracted['sequence']
                            mse = np.mean([(p - t)**2 for p, t in zip(sequence, ground_truth['sequence'])])
                            mae = np.mean([abs(p - t) for p, t in zip(sequence, ground_truth['sequence'])])
                            model_mse.append(mse)
                            model_mae.append(mae)
                            
                            all_results.append({
                                'example_index': batch_start + i,
                                'ground_truth': ground_truth['sequence'],
                                'prediction': sequence,
                                'mse': mse,
                                'mae': mae
                            })
                            
                            # details for first few
                            if i < num_examples:
                                log_print("\n" + "-"*80)
                                log_print(f"EXAMPLE {batch_start + i}")
                                log_print("-"*80)
                                log_print(f"Ground Truth:\n{ground_truth['sequence']}")
                                log_print(f"\nModel Output:\n{text}")
                                log_print("\nExtracted Values:")
                                log_print(f"Ground Truth: {ground_truth['sequence']}")
                                log_print(f"Prediction: {sequence}")
                                log_print(f"MSE: {mse:.6f}")
                                log_print(f"MAE: {mae:.6f}")
                        else:
                            log_print(f"\nFAILED EXTRACTION for example {batch_start + i}:")
                            log_print(f"Output text:\n{text}\n")
                    except Exception as e:
                        log_print(f"Error processing result {i}: {e}")
                        continue
                
                # batch metrics
                batch_avg_mse = np.mean([r['mse'] for r in all_results[-len(batch_texts):] if 'mse' in r])
                batch_avg_mae = np.mean([r['mae'] for r in all_results[-len(batch_texts):] if 'mae' in r])
                running_avg_mse = np.mean(model_mse) if model_mse else float('nan')
                running_avg_mae = np.mean(model_mae) if model_mae else float('nan')
                
                log_print("\nBatch Metrics:")
                log_print(f"Batch MSE: {batch_avg_mse:.6f}")
                log_print(f"Batch MAE: {batch_avg_mae:.6f}")
                log_print(f"Running MSE: {running_avg_mse:.6f}")
                log_print(f"Running MAE: {running_avg_mae:.6f}")
                        
            except Exception as e:
                log_print(f"Error during model inference: {e}")
                continue
        
        # final metrics
        log_print("\n" + "="*80)
        log_print("FINAL METRICS")
        log_print("="*80)
        
        avg_mse = np.mean(model_mse) if model_mse else float('nan')
        avg_mae = np.mean(model_mae) if model_mae else float('nan')
        
        log_print(f"Average MSE: {avg_mse:.6f}")
        log_print(f"Average MAE: {avg_mae:.6f}")
        
        success_rate = len(model_mse) / len(dataset) * 100
        log_print(f"\nValid Prediction Rate: {success_rate:.2f}%")
    
    # save csv
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(stats_file, index=False)
    
    print(f"\nDetailed evaluation results saved to: {detailed_output_file}")
    print(f"Statistics saved to: {stats_file}")
    
    return results_df

def perform_rejection_sampling(evaluation_stats_path, num_buckets=100, top_k=10):
    """
    Perform rejection sampling on the evaluation results to create SFT dataset.
    
    Args:
        evaluation_stats_path: Path to the evaluation stats CSV file
        num_buckets: Number of buckets to split the data into
        top_k: Number of top examples to take from each bucket
        
    Returns:
        DataFrame containing the selected examples for SFT training
    """
    try:
        # read eval results
        df = pd.read_csv(evaluation_stats_path)
        print(f"Total records in evaluation results: {len(df)}")
        
        # bucket size
        bucket_size = len(df) // num_buckets
        
        # per-bucket holders
        processed_dfs = []
        
        # iterate buckets
        for i in range(num_buckets):
            start_idx = i * bucket_size
            end_idx = start_idx + bucket_size if i < num_buckets - 1 else len(df)
            
            # slice bucket
            bucket_df = df.iloc[start_idx:end_idx]
            
            # pick top_k by smallest grpo_mse
            top_k_df = bucket_df.nsmallest(top_k, 'grpo_mse')
            
            # Add to processed dataframes
            processed_dfs.append(top_k_df)
        
        # concat
        final_df = pd.concat(processed_dfs, ignore_index=True)
        
        # overall mse
        overall_mse = final_df['grpo_mse'].mean()
        print(f"Overall grpo_mse for the SFT dataset: {overall_mse}")
        
        return final_df
        
    except Exception as e:
        print(f"Error during rejection sampling: {e}")
        return None

def merge_sft_weights(model, sft_saved_lora):
    """load sft lora and merge into base model"""
    print(f"Loading SFT LoRA weights from: {sft_saved_lora}")
    
    try:
        # Load the SFT LoRA weights
        model.load_lora(sft_saved_lora)
        
        print("Merging SFT weights into base model...")
        model = model.merge_and_unload()  # This returns the SFT-merged base model
        print("Successfully merged SFT LoRA weights.")
        return model
    except Exception as e:
        print(f"Error loading or merging SFT LoRA weights: {e}")
        return model

def evaluate_grpo_on_full_dataset(model, tokenizer, full_dataset, grpo_lora_path, num_examples=5, batch_size=60, visualize=False, output_dir=None):
    """
    Evaluate only the GRPO model on the full dataset split, calculate metrics,
    and save data for supervised fine-tuning.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        full_dataset: The full dataset
        grpo_lora_path: Path to GRPO LoRA weights
        num_examples: Number of example generations to print
        batch_size: Number of prompts to process in parallel
        visualize: Whether to generate visualizations (default: False)
        output_dir: Directory to save results (if None, will create a timestamped directory)
    """
    # Create a dedicated output directory with timestamp and model name
    if output_dir is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"evaluation_results/{timestamp}_grpo_full_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # file outputs
    detailed_output_file = f"{output_dir}/evaluation_details.txt"
    stats_file = f"{output_dir}/evaluation_stats.csv"
    
    with open(detailed_output_file, "w") as f_out:
        def log_print(*args, **kwargs):
            print(*args, **kwargs)  # Print to console
            print(*args, **kwargs, file=f_out)  # Print to file
            
        log_print("\n" + "="*80)
        log_print("EVALUATING GRPO MODEL ON FULL DATASET")
        log_print("="*80)
        
        log_print(f"Loaded full dataset with {len(full_dataset)} examples")
        
        # Convert dataset to list for easier batching
        dataset_list = [full_dataset[i] for i in range(len(full_dataset))]
        
        # sampling params
        sampling_params = SamplingParams(
            temperature=0.2,  # Use slightly higher temperature for more variety
            top_p=0.95,
            max_tokens=2048,
        )
        
        # load lora once
        grpo_lora = model.load_lora(grpo_lora_path)
        
        # metrics
        grpo_model_mse = []
        grpo_model_mae = []
        mean_diff_mse_list = []
        mean_diff_mae_list = []
        
        # holders
        all_results = {} # Use dict for easier updates by index
        
        # arrays for downstream
        all_inputs_dict = {} # Use dict for easier updates by index
        all_preds_grpo_dict = {} # Use dict for easier updates by index
        all_trues_dict = {} # Use dict for easier updates by index

        # retry queue
        retry_queue = [] # List of (example_index, text_input, ground_truth, prompt_content)
        
        # stage 1: initial pass
        log_print("\n--- Stage 1: Initial Batch Processing ---")
        
        # iterate batches
        for batch_start in range(0, len(dataset_list), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset_list))
            batch = dataset_list[batch_start:batch_end]
            
            log_print(f"\nProcessing initial batch {batch_start//batch_size + 1}, examples {batch_start} to {batch_end}")
            
            # per-batch metrics
            batch_grpo_mse = []
            batch_grpo_mae = []
            batch_mean_diff_mse = []
            batch_mean_diff_mae = []
            
            # prep inputs
            batch_texts = []
            batch_ground_truths = []
            batch_prompt_contents = []  # Store for retries
            batch_example_indices = []  # Store example indices
            
            # valid counters
            batch_valid_grpo = 0
            batch_total = 0
            
            for i, example in enumerate(batch):
                try:
                    example_index = batch_start + i
                    prompt_content = example['prompt'][1]['content']
                    ground_truth = extract_xml_answer(example['answer'])
                    
                    if not ground_truth:
                        log_print(f"Skipping example {example_index}: Invalid ground truth format")
                        continue
                    
                    text = tokenizer.apply_chat_template([
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_content},
                    ], tokenize=False, add_generation_prompt=True)
                    
                    batch_texts.append(text)
                    batch_ground_truths.append(ground_truth)
                    batch_prompt_contents.append(prompt_content)
                    batch_example_indices.append(example_index)
                    batch_total += 1
                except Exception as e:
                    log_print(f"Error preparing example {batch_start + i}: {str(e)}")
                    continue
            
            if not batch_texts:
                log_print("No valid examples in batch")
                continue
            
            log_print(f"Processing {len(batch_texts)} valid examples in current batch")
            
            try:
            # batch gen
                grpo_outputs = model.fast_generate(
                    batch_texts,
                    sampling_params=sampling_params,
                    lora_request=grpo_lora,
                )
                
                # handle outputs
                for i, (grpo_output, example_index, ground_truth, prompt_content) in enumerate(
                    zip(grpo_outputs, batch_example_indices, batch_ground_truths, batch_prompt_contents)
                ):
                    try:
                        # Extract outputs
                        grpo_text = grpo_output.outputs[0].text
                        
                        # Extract GRPO model prediction
                        grpo_extracted = extract_xml_answer(grpo_text)
                        
                        # Check if extraction was successful
                        if grpo_extracted and 'sequence' in grpo_extracted:
                            grpo_sequence = grpo_extracted['sequence']
                            
                            # track valid
                            batch_valid_grpo += 1
                            
                            # metrics
                            grpo_mse = np.mean([(p - t)**2 for p, t in zip(grpo_sequence, ground_truth['sequence'])])
                            grpo_mae = np.mean([abs(p - t) for p, t in zip(grpo_sequence, ground_truth['sequence'])])
                            
                            # means
                            truth_mean = np.mean(ground_truth['sequence'])
                            pred_mean = np.mean(grpo_sequence)
                            
                            # mean diff metrics
                            mean_diff = truth_mean - pred_mean
                            mean_diff_mse = mean_diff ** 2
                            mean_diff_mae = abs(mean_diff)
                            
                            # store
                            grpo_model_mse.append(grpo_mse)
                            grpo_model_mae.append(grpo_mae)
                            mean_diff_mse_list.append(mean_diff_mse)
                            mean_diff_mae_list.append(mean_diff_mae)
                            
                            # add to batch
                            batch_grpo_mse.append(grpo_mse)
                            batch_grpo_mae.append(grpo_mae)
                            batch_mean_diff_mse.append(mean_diff_mse)
                            batch_mean_diff_mae.append(mean_diff_mae)
                            
                            # write result entry
                            all_results[example_index] = {
                                'example_index': example_index,
                                'ground_truth': ground_truth['sequence'],
                                'grpo_prediction': grpo_sequence,
                                'grpo_mse': grpo_mse,
                                'grpo_mae': grpo_mae,
                                'truth_mean': truth_mean,
                                'pred_mean': pred_mean,
                                'mean_diff_mse': mean_diff_mse,
                                'mean_diff_mae': mean_diff_mae,
                                'prompt_question': prompt_content,
                                'full_answer': grpo_text
                            }
                            
                            # Store data for visualization
                            historical_data = extract_historical_data(prompt_content)
                            all_inputs_dict[example_index] = np.array(historical_data)
                            all_preds_grpo_dict[example_index] = np.array(grpo_sequence)
                            all_trues_dict[example_index] = np.array(ground_truth['sequence'])
                            
                            # Generate visualization if requested
                            if visualize and i % 20 == 0:
                                x_hist = historical_data
                                y_true = ground_truth['sequence']
                                # Full sequences
                                full_gt = np.concatenate([x_hist, y_true])
                                full_pred_grpo = np.concatenate([x_hist, grpo_sequence])
                                # Save visualization
                                visual_path = os.path.join(output_dir, f'grpo_model_vis_{example_index}.pdf')
                                visual(full_gt, full_pred_grpo, visual_path)
                                
                                log_print(f"GRPO Model MSE: {grpo_mse:.6f}")
                        else:
                            # Initial extraction failed: store placeholder results and queue for retry
                            all_results[example_index] = {
                                'example_index': example_index,
                                'ground_truth': ground_truth['sequence'],
                                'grpo_prediction': [np.nan] * 10,
                                'grpo_mse': np.nan,
                                'grpo_mae': np.nan,
                                'truth_mean': np.nan,
                                'pred_mean': np.nan,
                                'mean_diff_mse': np.nan,
                                'mean_diff_mae': np.nan,
                                'prompt_question': prompt_content,
                                'full_answer': grpo_text
                            }
                            # Record inputs/outputs for aggregation with NaN placeholders
                            historical_data = extract_historical_data(prompt_content)
                            all_inputs_dict[example_index] = np.array(historical_data)
                            all_preds_grpo_dict[example_index] = np.array([np.nan] * 10)
                            all_trues_dict[example_index] = np.array(ground_truth['sequence'])
                            retry_queue.append((example_index, text, ground_truth, prompt_content))
                    except Exception as e:
                        log_print(f"Error processing result {i}: {e}")
                        # Error during initial pass: store placeholder results and queue for retry
                        all_results[example_index] = {
                            'example_index': example_index,
                            'ground_truth': ground_truth['sequence'],
                            'grpo_prediction': [np.nan] * 10,
                            'grpo_mse': np.nan,
                            'grpo_mae': np.nan,
                            'truth_mean': np.nan,
                            'pred_mean': np.nan,
                            'mean_diff_mse': np.nan,
                            'mean_diff_mae': np.nan,
                            'prompt_question': prompt_content,
                            'full_answer': 'ERROR DURING PROCESSING'
                        }
                        # Also need to populate placeholders here for dict consistency
                        historical_data = extract_historical_data(prompt_content)
                        all_inputs_dict[example_index] = np.array(historical_data)
                        all_preds_grpo_dict[example_index] = np.array([np.nan] * 10)
                        all_trues_dict[example_index] = np.array(ground_truth['sequence'])
                        retry_queue.append((example_index, text, ground_truth, prompt_content))
                
                # batch metrics
                batch_avg_grpo_mse = np.mean(batch_grpo_mse) if batch_grpo_mse else float('nan')
                batch_avg_grpo_mae = np.mean(batch_grpo_mae) if batch_grpo_mae else float('nan')
                batch_avg_mean_diff_mse = np.mean(batch_mean_diff_mse) if batch_mean_diff_mse else float('nan')
                batch_avg_mean_diff_mae = np.mean(batch_mean_diff_mae) if batch_mean_diff_mae else float('nan')
                
                running_avg_grpo_mse = np.mean(grpo_model_mse) if grpo_model_mse else float('nan')
                running_avg_grpo_mae = np.mean(grpo_model_mae) if grpo_model_mae else float('nan')
                running_avg_mean_diff_mse = np.mean(mean_diff_mse_list) if mean_diff_mse_list else float('nan')
                running_avg_mean_diff_mae = np.mean(mean_diff_mae_list) if mean_diff_mae_list else float('nan')
                
                # valid rate
                batch_grpo_rate = batch_valid_grpo / batch_total * 100 if batch_total > 0 else 0
                
                log_print("\nInitial Batch Metrics:")
                log_print(f"GRPO Model Valid Predictions: {batch_valid_grpo}/{batch_total} ({batch_grpo_rate:.2f}%)")
                log_print(f"GRPO Model Batch MSE: {batch_avg_grpo_mse:.6f}")
                log_print(f"GRPO Model Batch MAE: {batch_avg_grpo_mae:.6f}")
                log_print(f"Mean Diff Batch MSE: {batch_avg_mean_diff_mse:.6f}")
                log_print(f"Mean Diff Batch MAE: {batch_avg_mean_diff_mae:.6f}")
                log_print(f"Running GRPO Model MSE: {running_avg_grpo_mse:.6f}")
                log_print(f"Running GRPO Model MAE: {running_avg_grpo_mae:.6f}")
            except Exception as e:
                log_print(f"Error during model inference: {e}")
                continue
        
        # stage 2: retry failed examples
        log_print("\n" + "="*80)
        log_print("STAGE 2: RETRY FAILED EXAMPLES")
        log_print("="*80)

        max_retries = 5
        log_print(f"Retrying {len(retry_queue)} failed examples up to {max_retries} times with increasing temperature")

        for example_index, text_input, ground_truth, prompt_content in retry_queue:
            log_print(f"\nRetrying example {example_index}...")
            success = False
            for attempt in range(max_retries):
                temp = sampling_params.temperature + 0.1 * (attempt + 1)
                retry_sampling_params = SamplingParams(
                    temperature=temp,
                    top_p=sampling_params.top_p,
                    max_tokens=sampling_params.max_tokens,
                )
                try:
                    retry_output = model.fast_generate(
                        [text_input],
                        sampling_params=retry_sampling_params,
                        lora_request=grpo_lora,
                    )[0]
                    retry_text = retry_output.outputs[0].text
                    retry_extracted = extract_xml_answer(retry_text)
                    if retry_extracted and 'sequence' in retry_extracted:
                        log_print(f"Retry SUCCEEDED on attempt {attempt + 1} for example {example_index}")
                        grpo_sequence = retry_extracted['sequence']
                        grpo_mse = np.mean([(p - t)**2 for p, t in zip(grpo_sequence, ground_truth['sequence'])])
                        grpo_mae = np.mean([abs(p - t) for p, t in zip(grpo_sequence, ground_truth['sequence'])])
                        truth_mean = np.mean(ground_truth['sequence'])
                        pred_mean = np.mean(grpo_sequence)
                        mean_diff = truth_mean - pred_mean
                        mean_diff_mse = mean_diff ** 2
                        mean_diff_mae = abs(mean_diff)
                        all_results[example_index].update({
                            'grpo_prediction': grpo_sequence,
                            'grpo_mse': grpo_mse,
                            'grpo_mae': grpo_mae,
                            'truth_mean': truth_mean,
                            'pred_mean': pred_mean,
                            'mean_diff_mse': mean_diff_mse,
                            'mean_diff_mae': mean_diff_mae,
                            'prompt_question': prompt_content,
                            'full_answer': retry_text
                        })
                        all_preds_grpo_dict[example_index] = np.array(grpo_sequence)
                        success = True
                        break
                    else:
                        log_print(f"Retry attempt {attempt + 1} failed format check for example {example_index}")
                except Exception as e:
                    log_print(f"Error during retry attempt {attempt + 1} for example {example_index}: {e}")
            if not success:
                log_print(f"Example {example_index} FAILED all {max_retries} retry attempts; leaving NaN placeholders")
        
        # Calculate and print final metrics
        log_print("\n" + "="*80)
        log_print("FINAL METRICS")
        log_print("="*80)
        
        # final metrics
        final_grpo_avg_mse = np.nanmean(grpo_model_mse) if grpo_model_mse else float('nan')
        final_grpo_avg_mae = np.nanmean(grpo_model_mae) if grpo_model_mae else float('nan')
        final_mean_diff_avg_mse = np.nanmean(mean_diff_mse_list) if mean_diff_mse_list else float('nan')
        final_mean_diff_avg_mae = np.nanmean(mean_diff_mae_list) if mean_diff_mae_list else float('nan')
        
        log_print(f"GRPO Model Final Average MSE (over valid predictions): {final_grpo_avg_mse:.6f}")
        log_print(f"GRPO Model Final Average MAE (over valid predictions): {final_grpo_avg_mae:.6f}")
        log_print(f"Mean Difference Final Average MSE (over valid predictions): {final_mean_diff_avg_mse:.6f}")
        log_print(f"Mean Difference Final Average MAE (over valid predictions): {final_mean_diff_avg_mae:.6f}")
        
        # final success rate
        total_examples_processed = len(all_results)
        final_valid_predictions_count = sum(1 for idx, r in all_results.items() if r['grpo_mse'] is not np.nan and not pd.isna(r['grpo_mse']))
        final_grpo_success_rate = final_valid_predictions_count / total_examples_processed * 100 if total_examples_processed > 0 else 0
        
        log_print(f"\nTotal Examples Processed: {total_examples_processed}")
        log_print(f"Final Valid GRPO Predictions (after retries): {final_valid_predictions_count}")
        log_print(f"GRPO Model Final Valid Prediction Rate: {final_grpo_success_rate:.2f}% ")
    
    # Convert result dictionaries to lists in correct order for saving
    ordered_indices = sorted(all_results.keys())
    
    # Remove 'needs_retry' and 'retry_attempt' fields from each result dictionary
    for idx in ordered_indices:
        if 'needs_retry' in all_results[idx]:
            del all_results[idx]['needs_retry']
        if 'retry_attempt' in all_results[idx]:
            del all_results[idx]['retry_attempt']
    
    final_all_results_list = [all_results[i] for i in ordered_indices]
    final_all_inputs = np.array([all_inputs_dict[i] for i in ordered_indices])
    final_all_preds_grpo = np.array([all_preds_grpo_dict[i] for i in ordered_indices])
    final_all_trues = np.array([all_trues_dict[i] for i in ordered_indices])

    # save arrays
    np.save(os.path.join(output_dir, "inputs.npy"), final_all_inputs)
    np.save(os.path.join(output_dir, "preds_grpo.npy"), final_all_preds_grpo)
    np.save(os.path.join(output_dir, "true.npy"), final_all_trues)
    
    # save csv
    results_df = pd.DataFrame(final_all_results_list)
    results_df.to_csv(stats_file, index=False)
    
    print(f"\nDetailed evaluation results saved to: {detailed_output_file}")
    print(f"Statistics saved to: {stats_file}")
    
    return results_df

def compare_all_models_on_test_set(model, tokenizer, test_dataset, lora_paths, num_examples=1, batch_size=60, visualize=False, output_dir=None):
    """
    Compare all models side by side: base model, 1st GRPO model, SFT model, and 2nd GRPO model.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        test_dataset: The test dataset
        lora_paths: Dict containing paths to "grpo_1st", "sft", and "grpo_2nd" LoRA weights
        num_examples: Number of example generations to print
        batch_size: Number of prompts to process in parallel
        visualize: Whether to generate visualizations (default: False)
        output_dir: Directory to save results (if None, creates a timestamped directory)
    """
    # Create a dedicated output directory with timestamp and model name
    if output_dir is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"evaluation_results/{timestamp}_compare_all_models"
    os.makedirs(output_dir, exist_ok=True)
    
    # file outputs
    detailed_output_file = f"{output_dir}/comparison_details.txt"
    stats_file = f"{output_dir}/comparison_stats.csv"
    
    with open(detailed_output_file, "w") as f_out:
        def log_print(*args, **kwargs):
            print(*args, **kwargs)  # Print to console
            print(*args, **kwargs, file=f_out)  # Print to file
            
        log_print("\n" + "="*80)
        log_print("COMPARING ALL MODELS ON TEST SET")
        log_print("="*80)
        
        log_print(f"Loaded test dataset with {len(test_dataset)} examples")
        
        # Convert dataset to list for easier batching
        dataset_list = [test_dataset[i] for i in range(len(test_dataset))]
        
        # sampling params
        sampling_params = SamplingParams(
            temperature=0.2,  # Lower temperature for more deterministic outputs
            top_p=0.95,
            max_tokens=2048,
        )
        
        # Load all LoRA weights up front
        log_print("Loading all LoRA weights...")
        grpo_1st_lora = model.load_lora(lora_paths["grpo_1st"])
        sft_lora = model.load_lora(lora_paths["sft"])
        grpo_2nd_lora = model.load_lora(lora_paths["grpo_2nd"])
        
        # metrics
        base_model_mse = []
        grpo_1st_model_mse = []
        sft_model_mse = []
        grpo_2nd_model_mse = []
        
        # Store all predictions and metrics for CSV
        all_results = []
        
        # iterate batches
        for batch_start in range(0, len(dataset_list), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset_list))
            batch = dataset_list[batch_start:batch_end]
            
            log_print(f"\nProcessing batch {batch_start//batch_size + 1}, examples {batch_start} to {batch_end}")
            
            # prep inputs
            batch_texts = []
            batch_ground_truths = []
            batch_prompts = []
            
            # valid counters
            batch_valid_counts = {
                "base": 0,
                "grpo_1st": 0,
                "sft": 0,
                "grpo_2nd": 0
            }
            batch_total = 0
            
            for example in batch:
                try:
                    prompt_content = example['prompt'][1]['content']
                    ground_truth = extract_xml_answer(example['answer'])
                    
                    if not ground_truth:
                        continue
                    
                    text = tokenizer.apply_chat_template([
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_content},
                    ], tokenize=False, add_generation_prompt=True)
                    
                    batch_texts.append(text)
                    batch_ground_truths.append(ground_truth)
                    batch_prompts.append(prompt_content)
                    batch_total += 1
                except Exception as e:
                    log_print(f"Error preparing example: {str(e)}")
                    continue
            
            if not batch_texts:
                log_print("No valid examples in batch")
                continue
            
            log_print(f"Processing {len(batch_texts)} valid examples in current batch")
            
            try:
                # Run inference with all four models
                log_print("Running base model inference...")
                base_outputs = model.fast_generate(
                    batch_texts,
                    sampling_params=sampling_params,
                    lora_request=None,  # No LoRA for base model
                )
                
                log_print("Running 1st GRPO model inference...")
                grpo_1st_outputs = model.fast_generate(
                    batch_texts,
                    sampling_params=sampling_params,
                    lora_request=grpo_1st_lora,
                )
                
                log_print("Running SFT model inference...")
                sft_outputs = model.fast_generate(
                    batch_texts,
                    sampling_params=sampling_params,
                    lora_request=sft_lora,
                )
                
                log_print("Running 2nd GRPO model inference...")
                grpo_2nd_outputs = model.fast_generate(
                    batch_texts,
                    sampling_params=sampling_params,
                    lora_request=grpo_2nd_lora,
                )
                
                # handle outputs
                for i, (base_output, grpo_1st_output, sft_output, grpo_2nd_output, ground_truth, prompt_content) in enumerate(
                    zip(base_outputs, grpo_1st_outputs, sft_outputs, grpo_2nd_outputs, batch_ground_truths, batch_prompts)
                ):
                    try:
                        # example idx
                        example_index = batch_start + i
                        
                        # Extract outputs
                        base_text = base_output.outputs[0].text
                        grpo_1st_text = grpo_1st_output.outputs[0].text
                        sft_text = sft_output.outputs[0].text
                        grpo_2nd_text = grpo_2nd_output.outputs[0].text
                        
                        # parse answers
                        base_extracted = extract_xml_answer(base_text)
                        grpo_1st_extracted = extract_xml_answer(grpo_1st_text)
                        sft_extracted = extract_xml_answer(sft_text)
                        grpo_2nd_extracted = extract_xml_answer(grpo_2nd_text)
                        
                        # track validity
                        if base_extracted: batch_valid_counts["base"] += 1
                        if grpo_1st_extracted: batch_valid_counts["grpo_1st"] += 1
                        if sft_extracted: batch_valid_counts["sft"] += 1
                        if grpo_2nd_extracted: batch_valid_counts["grpo_2nd"] += 1
                        
                        # mse per model (if valid)
                        base_mse = np.mean([(p - t)**2 for p, t in zip(base_extracted['sequence'], ground_truth['sequence'])]) if base_extracted and 'sequence' in base_extracted and 'sequence' in ground_truth else None
                        grpo_1st_mse = np.mean([(p - t)**2 for p, t in zip(grpo_1st_extracted['sequence'], ground_truth['sequence'])]) if grpo_1st_extracted and 'sequence' in grpo_1st_extracted and 'sequence' in ground_truth else None
                        sft_mse = np.mean([(p - t)**2 for p, t in zip(sft_extracted['sequence'], ground_truth['sequence'])]) if sft_extracted and 'sequence' in sft_extracted and 'sequence' in ground_truth else None
                        grpo_2nd_mse = np.mean([(p - t)**2 for p, t in zip(grpo_2nd_extracted['sequence'], ground_truth['sequence'])]) if grpo_2nd_extracted and 'sequence' in grpo_2nd_extracted and 'sequence' in ground_truth else None
                        
                        # Store MSE values for overall metrics
                        if base_mse is not None: base_model_mse.append(base_mse)
                        if grpo_1st_mse is not None: grpo_1st_model_mse.append(grpo_1st_mse)
                        if sft_mse is not None: sft_model_mse.append(sft_mse)
                        if grpo_2nd_mse is not None: grpo_2nd_model_mse.append(grpo_2nd_mse)
                        
                        # record example
                        result_dict = {
                            'example_index': example_index,
                            'ground_truth': ground_truth['sequence'] if 'sequence' in ground_truth else None,
                            'base_prediction': base_extracted['sequence'] if base_extracted and 'sequence' in base_extracted else None,
                            'grpo_1st_prediction': grpo_1st_extracted['sequence'] if grpo_1st_extracted and 'sequence' in grpo_1st_extracted else None,
                            'sft_prediction': sft_extracted['sequence'] if sft_extracted and 'sequence' in sft_extracted else None,
                            'grpo_2nd_prediction': grpo_2nd_extracted['sequence'] if grpo_2nd_extracted and 'sequence' in grpo_2nd_extracted else None,
                            'base_mse': base_mse,
                            'grpo_1st_mse': grpo_1st_mse,
                            'sft_mse': sft_mse,
                            'grpo_2nd_mse': grpo_2nd_mse
                        }
                        all_results.append(result_dict)
                        
                        # visualization (optional)
                        if visualize and i % 20 == 0:
                            historical_data = extract_historical_data(prompt_content)
                            
                            # Only visualize if we have all predictions available
                            if base_extracted and grpo_1st_extracted and sft_extracted and grpo_2nd_extracted:
                                # Create a combined visualization showing all models
                                plt.figure(figsize=(12, 8))
                                # First plot: ground truth
                                ground_truth_data = np.concatenate([historical_data, ground_truth['sequence']])
                                plt.plot(ground_truth_data, label='Ground Truth', linewidth=2, color='black')
                                
                                # Plot model predictions
                                base_pred_data = np.concatenate([historical_data, base_extracted['sequence']])
                                grpo_1st_pred_data = np.concatenate([historical_data, grpo_1st_extracted['sequence']])
                                sft_pred_data = np.concatenate([historical_data, sft_extracted['sequence']])
                                grpo_2nd_pred_data = np.concatenate([historical_data, grpo_2nd_extracted['sequence']])
                                
                                plt.plot(base_pred_data, label=f'Base Model (MSE: {base_mse:.4f})', linestyle='--')
                                plt.plot(grpo_1st_pred_data, label=f'1st GRPO (MSE: {grpo_1st_mse:.4f})', linestyle='-.')
                                plt.plot(sft_pred_data, label=f'SFT (MSE: {sft_mse:.4f})', linestyle=':')
                                plt.plot(grpo_2nd_pred_data, label=f'2nd GRPO (MSE: {grpo_2nd_mse:.4f})', linestyle='-')
                                
                                plt.title(f'Comparison of All Models - Example {example_index}')
                                plt.xlabel('Time Steps')
                                plt.ylabel('Stock Price')
                                plt.legend()
                                plt.grid(True)
                                plt.tight_layout()
                                plt.savefig(os.path.join(output_dir, f'all_models_vis_{example_index}.pdf'))
                                plt.close()
                        
                        # details for first few
                        if i < num_examples:
                            log_print("\n" + "-"*80)
                            log_print(f"EXAMPLE {example_index}")
                            log_print("-"*80)
                            log_print(f"Ground Truth:\n{ground_truth['sequence'] if 'sequence' in ground_truth else 'N/A'}")
                            log_print(f"\nBase Model Output:\n{base_text}")
                            log_print(f"\n1st GRPO Model Output:\n{grpo_1st_text}")
                            log_print(f"\nSFT Model Output:\n{sft_text}")
                            log_print(f"\n2nd GRPO Model Output:\n{grpo_2nd_text}")
                            
                            log_print("\nExtracted Values:")
                            log_print(f"Ground Truth: {ground_truth['sequence'] if 'sequence' in ground_truth else 'N/A'}")
                            log_print(f"Base Model: {base_extracted['sequence'] if base_extracted and 'sequence' in base_extracted else 'Invalid format'}")
                            log_print(f"1st GRPO Model: {grpo_1st_extracted['sequence'] if grpo_1st_extracted and 'sequence' in grpo_1st_extracted else 'Invalid format'}")
                            log_print(f"SFT Model: {sft_extracted['sequence'] if sft_extracted and 'sequence' in sft_extracted else 'Invalid format'}")
                            log_print(f"2nd GRPO Model: {grpo_2nd_extracted['sequence'] if grpo_2nd_extracted and 'sequence' in grpo_2nd_extracted else 'Invalid format'}")
                            
                            if base_mse is not None:
                                log_print(f"Base Model MSE: {base_mse:.6f}")
                            if grpo_1st_mse is not None:
                                log_print(f"1st GRPO Model MSE: {grpo_1st_mse:.6f}")
                            if sft_mse is not None:
                                log_print(f"SFT Model MSE: {sft_mse:.6f}")
                            if grpo_2nd_mse is not None:
                                log_print(f"2nd GRPO Model MSE: {grpo_2nd_mse:.6f}")
                    
                    except Exception as e:
                        log_print(f"Error processing result {batch_start + i}: {e}")
                        continue
                
                # batch metrics
                log_print("\nBatch Metrics:")
                
                # valid rates
                for model_name in batch_valid_counts:
                    valid_rate = batch_valid_counts[model_name] / batch_total * 100 if batch_total > 0 else 0
                    log_print(f"{model_name.capitalize().replace('_', ' ')} Model Valid Predictions: {batch_valid_counts[model_name]}/{batch_total} ({valid_rate:.2f}%)")
                
                # batch avg mse
                batch_base_mse = [r['base_mse'] for r in all_results[-len(batch_texts):] if r['base_mse'] is not None]
                batch_grpo_1st_mse = [r['grpo_1st_mse'] for r in all_results[-len(batch_texts):] if r['grpo_1st_mse'] is not None]
                batch_sft_mse = [r['sft_mse'] for r in all_results[-len(batch_texts):] if r['sft_mse'] is not None]
                batch_grpo_2nd_mse = [r['grpo_2nd_mse'] for r in all_results[-len(batch_texts):] if r['grpo_2nd_mse'] is not None]
                
                batch_base_avg = np.mean(batch_base_mse) if batch_base_mse else float('nan')
                batch_grpo_1st_avg = np.mean(batch_grpo_1st_mse) if batch_grpo_1st_mse else float('nan')
                batch_sft_avg = np.mean(batch_sft_mse) if batch_sft_mse else float('nan')
                batch_grpo_2nd_avg = np.mean(batch_grpo_2nd_mse) if batch_grpo_2nd_mse else float('nan')
                
                log_print(f"Base Model Batch MSE: {batch_base_avg:.6f}")
                log_print(f"1st GRPO Model Batch MSE: {batch_grpo_1st_avg:.6f}")
                log_print(f"SFT Model Batch MSE: {batch_sft_avg:.6f}")
                log_print(f"2nd GRPO Model Batch MSE: {batch_grpo_2nd_avg:.6f}")
                
                # show improvements
                if not np.isnan(batch_base_avg):
                    if not np.isnan(batch_grpo_1st_avg):
                        imp_1 = (batch_base_avg - batch_grpo_1st_avg) / batch_base_avg * 100
                        log_print(f"1st GRPO Improvement over Base: {imp_1:.2f}%")
                    if not np.isnan(batch_sft_avg):
                        imp_2 = (batch_base_avg - batch_sft_avg) / batch_base_avg * 100
                        log_print(f"SFT Improvement over Base: {imp_2:.2f}%")
                    if not np.isnan(batch_grpo_2nd_avg):
                        imp_3 = (batch_base_avg - batch_grpo_2nd_avg) / batch_base_avg * 100
                        log_print(f"2nd GRPO Improvement over Base: {imp_3:.2f}%")
                
                if not np.isnan(batch_grpo_1st_avg) and not np.isnan(batch_grpo_2nd_avg) and batch_grpo_1st_avg > 0:
                    imp_4 = (batch_grpo_1st_avg - batch_grpo_2nd_avg) / batch_grpo_1st_avg * 100
                    log_print(f"2nd GRPO Improvement over 1st GRPO: {imp_4:.2f}%")
                
                if not np.isnan(batch_sft_avg) and not np.isnan(batch_grpo_2nd_avg) and batch_sft_avg > 0:
                    imp_5 = (batch_sft_avg - batch_grpo_2nd_avg) / batch_sft_avg * 100
                    log_print(f"2nd GRPO Improvement over SFT: {imp_5:.2f}%")
                
            except Exception as e:
                log_print(f"Error during model inference: {e}")
                continue
        
        # final metrics
        log_print("\n" + "="*80)
        log_print("FINAL METRICS")
        log_print("="*80)
        
        # average mse per model
        base_avg_mse = np.mean(base_model_mse) if base_model_mse else float('nan')
        grpo_1st_avg_mse = np.mean(grpo_1st_model_mse) if grpo_1st_model_mse else float('nan')
        sft_avg_mse = np.mean(sft_model_mse) if sft_model_mse else float('nan')
        grpo_2nd_avg_mse = np.mean(grpo_2nd_model_mse) if grpo_2nd_model_mse else float('nan')
        
        log_print(f"Base Model Average MSE: {base_avg_mse:.6f}")
        log_print(f"1st GRPO Model Average MSE: {grpo_1st_avg_mse:.6f}")
        log_print(f"SFT Model Average MSE: {sft_avg_mse:.6f}")
        log_print(f"2nd GRPO Model Average MSE: {grpo_2nd_avg_mse:.6f}")
        
        # improvements
        if not np.isnan(base_avg_mse):
            if not np.isnan(grpo_1st_avg_mse):
                imp_1 = (base_avg_mse - grpo_1st_avg_mse) / base_avg_mse * 100
                log_print(f"1st GRPO Improvement over Base: {imp_1:.2f}%")
            if not np.isnan(sft_avg_mse):
                imp_2 = (base_avg_mse - sft_avg_mse) / base_avg_mse * 100
                log_print(f"SFT Improvement over Base: {imp_2:.2f}%")
            if not np.isnan(grpo_2nd_avg_mse):
                imp_3 = (base_avg_mse - grpo_2nd_avg_mse) / base_avg_mse * 100
                log_print(f"2nd GRPO Improvement over Base: {imp_3:.2f}%")
        
        if not np.isnan(grpo_1st_avg_mse) and not np.isnan(grpo_2nd_avg_mse) and grpo_1st_avg_mse > 0:
            imp_4 = (grpo_1st_avg_mse - grpo_2nd_avg_mse) / grpo_1st_avg_mse * 100
            log_print(f"2nd GRPO Improvement over 1st GRPO: {imp_4:.2f}%")
        
        if not np.isnan(sft_avg_mse) and not np.isnan(grpo_2nd_avg_mse) and sft_avg_mse > 0:
            imp_5 = (sft_avg_mse - grpo_2nd_avg_mse) / sft_avg_mse * 100
            log_print(f"2nd GRPO Improvement over SFT: {imp_5:.2f}%")
        
        # valid rates
        total_examples = len(test_dataset)
        base_success_rate = len(base_model_mse) / total_examples * 100
        grpo_1st_success_rate = len(grpo_1st_model_mse) / total_examples * 100
        sft_success_rate = len(sft_model_mse) / total_examples * 100
        grpo_2nd_success_rate = len(grpo_2nd_model_mse) / total_examples * 100
        
        log_print(f"\nBase Model Valid Prediction Rate: {base_success_rate:.2f}%")
        log_print(f"1st GRPO Model Valid Prediction Rate: {grpo_1st_success_rate:.2f}%")
        log_print(f"SFT Model Valid Prediction Rate: {sft_success_rate:.2f}%")
        log_print(f"2nd GRPO Model Valid Prediction Rate: {grpo_2nd_success_rate:.2f}%")
        
        # summary table
        log_print("\n" + "-"*80)
        log_print("MODEL PERFORMANCE SUMMARY")
        log_print("-"*80)
        log_print(f"{'Model':<15} | {'Valid Rate':<10} | {'MSE':<10} | {'Improvement':<10}")
        log_print("-"*80)
        log_print(f"{'Base':<15} | {base_success_rate:<10.2f}% | {base_avg_mse:<10.6f} | {'N/A':<10}")
        log_print(f"{'1st GRPO':<15} | {grpo_1st_success_rate:<10.2f}% | {grpo_1st_avg_mse:<10.6f} | {imp_1 if not np.isnan(base_avg_mse) and not np.isnan(grpo_1st_avg_mse) else 'N/A':<10.2f}%")
        log_print(f"{'SFT':<15} | {sft_success_rate:<10.2f}% | {sft_avg_mse:<10.6f} | {imp_2 if not np.isnan(base_avg_mse) and not np.isnan(sft_avg_mse) else 'N/A':<10.2f}%")
        log_print(f"{'2nd GRPO':<15} | {grpo_2nd_success_rate:<10.2f}% | {grpo_2nd_avg_mse:<10.6f} | {imp_3 if not np.isnan(base_avg_mse) and not np.isnan(grpo_2nd_avg_mse) else 'N/A':<10.2f}%")
    
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(stats_file, index=False)
    
    print(f"\nDetailed comparison results saved to: {detailed_output_file}")
    print(f"Statistics saved to: {stats_file}")
    
    return results_df 