import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from utils.tools import Logger, save_hyperparameters, cleanup_gpu_memory
import warnings
import logging
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# Silence the GenerationMixin warning spam for custom models
warnings.filterwarnings(
    "ignore",
    message=r".*has generative capabilities, as `prepare_inputs_for_generation` is explicitly defined.*",
)
# Reduce transformers logger noise if present
logging.getLogger("transformers").setLevel(logging.ERROR)

def run_single_stock(args, timestamp, results_dir):
    """Process a single stock file"""
    # Set up logging
    original_stdout = sys.stdout
    log_file_path = os.path.join(results_dir, 'output_log.txt')
    sys.stdout = Logger(log_file_path)
    
    try:
        period = args.root_path.split(os.sep)[-1]
        print(f"Processing single stock: {args.data_path}")
        
        if args.eval_only:
            print("Running in EVALUATION ONLY mode - no training will be performed")
        
        results = pd.DataFrame(columns=[
            'Ticker', 'Period', 'MSE', 'MAE', 'RMSE', 
            'Accuracy', 'MCC', 'F1', 'Predicted StdDev', 'Volatility_Ratio',
            'Model_ID'
        ])
        
        # Create a file to store the model mapping
        model_mappings_file = os.path.join(results_dir, 'ticker_model_mappings.csv')
        model_mappings = pd.DataFrame(columns=['Ticker', 'Model_ID', 'Checkpoint_Path'])
        
        ticker = os.path.splitext(os.path.basename(args.data_path))[0]
        all_metrics = []
        

        if args.ref_sequence_csv_dir is not None:
            ticker_ref_path = os.path.join(args.ref_sequence_csv_dir, f"{ticker}.csv")
            if os.path.exists(ticker_ref_path):
                args.ref_sequence_csv_path = ticker_ref_path
                print(f"Using ticker-specific reference sequence: {ticker_ref_path}")
            else:
                print(f"Warning: No reference sequence found at {ticker_ref_path}, using default path")
        
        for ii in range(args.itr):
            setting = f'{args.model_id}'
            model_id = f"{setting}_{ticker}"  # Create model_id with ticker
            
            if not args.eval_only:
                print(f'>>>>>>>start training : {setting} for {ticker}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print(f'Model will be saved with ID: {model_id}')
            else:
                print(f'>>>>>>>loading model : {setting} for {ticker}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            
            exp = Exp_Long_Term_Forecast(args, results_dir)
            
            # Only train if not in eval mode
            if not args.eval_only:
                exp.train(setting)
            
            print(f'>>>>>>>testing : {setting} for {ticker}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            
            # For eval_only, try to determine the checkpoint path
            specific_model_path = None
            if args.eval_only:
                checkpoint_dir = os.path.join(args.checkpoints, model_id)
                checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
                
                if os.path.exists(checkpoint_path):
                    specific_model_path = checkpoint_path
                else:
                    orig_checkpoint = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
                    if os.path.exists(orig_checkpoint):
                        specific_model_path = orig_checkpoint
                        print(f"Using fallback checkpoint at: {orig_checkpoint}")
                    else:
                        print(f"WARNING: No checkpoint found at {checkpoint_path} or {orig_checkpoint}")
            
            # Perform testing and get metrics
            mae, mse, rmse, accuracy, mcc, f1, pred_std, vol_ratio = exp.test(setting, specific_model_path)
            
            all_metrics.append({
                'mae': mae, 
                'mse': mse, 
                'rmse': rmse,
                'accuracy': accuracy,
                'mcc': mcc,
                'f1': f1,
                'pred_std': pred_std,
                'vol_ratio': vol_ratio
            })
            
            # Record the mapping between ticker and model_id
            checkpoint_path = os.path.join(args.checkpoints, model_id)
            model_mappings = pd.concat([model_mappings, pd.DataFrame({
                'Ticker': [ticker],
                'Model_ID': [model_id],
                'Checkpoint_Path': [checkpoint_path]
            })], ignore_index=True)
            
            # Save model mappings after each iteration
            model_mappings.to_csv(model_mappings_file, index=False)
            
            # Calculate and store average metrics
            avg_metrics = {
                'Ticker': ticker,
                'Period': period,
                'MSE': np.mean([m['mse'] for m in all_metrics]),
                'MAE': np.mean([m['mae'] for m in all_metrics]),
                'RMSE': np.mean([m['rmse'] for m in all_metrics]),
                'Accuracy': np.mean([m['accuracy'] for m in all_metrics]),
                'MCC': np.mean([m['mcc'] for m in all_metrics]),
                'F1': np.mean([m['f1'] for m in all_metrics]),
                'Predicted StdDev': np.mean([m['pred_std'] for m in all_metrics]),
                'Volatility_Ratio': np.mean([m['vol_ratio'] for m in all_metrics]),
                'Model_ID': model_id
            }
            
            results = pd.concat([results, pd.DataFrame([avg_metrics])], ignore_index=True)
            
            # Save results after each iteration
            results.to_csv(os.path.join(results_dir, 'results.csv'), index=False)
            
            # Clean up GPU memory after each iteration
            cleanup_gpu_memory()
        
        # Preserve checkpoint
        print(f"Preserving checkpoint in {os.path.join(args.checkpoints, model_id)}")
        

            
    finally:
        # Restore original stdout
        sys.stdout = original_stdout

def run_multi_stock(args, timestamp, results_dir):
    """Process multiple stock files in a directory"""
    # Get list of CSV files in the directory
    csv_files = [f for f in os.listdir(args.root_path) if f.endswith('.csv')]
    
    print("\nFiles found in directory:")
    for f in csv_files:
        print(f"  - {f}")
    print()
    
    period = args.root_path.split(os.sep)[-1]
    print(f"Processing {len(csv_files)} files...")
    
    if args.eval_only:
        print("Running in EVALUATION ONLY mode - no training will be performed")
    
    # DataFrame for consolidated results of all stocks
    all_stocks_results = pd.DataFrame(columns=[
        'Ticker', 'Period', 'MSE', 'MAE', 'RMSE', 
        'Accuracy', 'MCC', 'F1', 'Predicted StdDev', 'Volatility_Ratio',
        'Model_ID'
    ])
    
    # Create a file to store the model mapping
    model_mappings_file = os.path.join(results_dir, 'ticker_model_mappings.csv')
    model_mappings = pd.DataFrame(columns=['Ticker', 'Model_ID', 'Checkpoint_Path'])
    
    for idx, csv_file in enumerate(csv_files):
        print(f'\nRunning on file {csv_file} ({idx+1}/{len(csv_files)})...')
        
        # Clean up GPU memory before each stock
        cleanup_gpu_memory()
        
        ticker = os.path.splitext(csv_file)[0]
        
        # Create ticker-specific directory
        ticker_dir = os.path.join(results_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Set up logging for this ticker
        original_stdout = sys.stdout
        log_file_path = os.path.join(ticker_dir, 'output_log.txt')
        sys.stdout = Logger(log_file_path)
        
        try:
            # Update args for this specific stock
            args.data_path = csv_file
            

            if args.ref_sequence_csv_dir is not None:
                ticker_ref_path = os.path.join(args.ref_sequence_csv_dir, f"{ticker}.csv")
                if os.path.exists(ticker_ref_path):
                    args.ref_sequence_csv_path = ticker_ref_path
                    print(f"Using ticker-specific reference sequence: {ticker_ref_path}")
                else:
                    print(f"Warning: No reference sequence found at {ticker_ref_path}, using default path")
            
            all_metrics = []
            
            for ii in range(args.itr):
                setting = f'{args.model_id}'
                model_id = f"{setting}_{ticker}"  # Create model_id with ticker
                
                if not args.eval_only:
                    print(f'>>>>>>>start training : {setting} for {ticker}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    print(f'Model will be saved with ID: {model_id}')
                else:
                    print(f'>>>>>>>loading model : {setting} for {ticker}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                
                exp = Exp_Long_Term_Forecast(args, ticker_dir)
                
                # Only train if not in eval mode
                if not args.eval_only:
                    exp.train(setting)
                
                print(f'>>>>>>>testing : {setting} for {ticker}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                
                # For eval_only, try to determine the checkpoint path
                specific_model_path = None
                if args.eval_only:
                    checkpoint_dir = os.path.join(args.checkpoints, model_id)
                    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
                    
                    if os.path.exists(checkpoint_path):
                        specific_model_path = checkpoint_path
                    else:
                        orig_checkpoint = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
                        if os.path.exists(orig_checkpoint):
                            specific_model_path = orig_checkpoint
                            print(f"Using fallback checkpoint at: {orig_checkpoint}")
                        else:
                            print(f"WARNING: No checkpoint found at {checkpoint_path} or {orig_checkpoint}")
                
                # Perform testing and get metrics
                mae, mse, rmse, accuracy, mcc, f1, pred_std, vol_ratio = exp.test(setting, specific_model_path)
                
                all_metrics.append({
                    'mae': mae, 
                    'mse': mse, 
                    'rmse': rmse,
                    'accuracy': accuracy,
                    'mcc': mcc,
                    'f1': f1,
                    'pred_std': pred_std,
                    'vol_ratio': vol_ratio
                })
                
                # Record the mapping between ticker and model_id
                checkpoint_path = os.path.join(args.checkpoints, model_id)
                model_mappings = pd.concat([model_mappings, pd.DataFrame({
                    'Ticker': [ticker],
                    'Model_ID': [model_id],
                    'Checkpoint_Path': [checkpoint_path]
                })], ignore_index=True)
                
                # Save model mappings after each iteration
                model_mappings.to_csv(model_mappings_file, index=False)
                
                # Clean up GPU memory after each iteration
                cleanup_gpu_memory()
            
            # Check if all_metrics is not empty before calculating averages
            if all_metrics:
                avg_metrics = {
                    'mse': np.mean([m['mse'] for m in all_metrics]),
                    'mae': np.mean([m['mae'] for m in all_metrics]),
                    'rmse': np.mean([m['rmse'] for m in all_metrics]),
                    'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
                    'mcc': np.mean([m['mcc'] for m in all_metrics]),
                    'f1': np.mean([m['f1'] for m in all_metrics]),
                    'pred_std': np.mean([m['pred_std'] for m in all_metrics]),
                    'vol_ratio': np.mean([m['vol_ratio'] for m in all_metrics])
                }
                
                # Add to consolidated results with model_id
                stock_result = pd.DataFrame({
                    'Ticker': [ticker],
                    'Period': [period],
                    'MSE': [avg_metrics['mse']],
                    'MAE': [avg_metrics['mae']],
                    'RMSE': [avg_metrics['rmse']],
                    'Accuracy': [avg_metrics['accuracy']],
                    'MCC': [avg_metrics['mcc']],
                    'F1': [avg_metrics['f1']],
                    'Predicted StdDev': [avg_metrics['pred_std']],
                    'Volatility_Ratio': [avg_metrics['vol_ratio']],
                    'Model_ID': [model_id]
                })
                all_stocks_results = pd.concat([all_stocks_results, stock_result], ignore_index=True)
                
                # Save ticker-specific results
                stock_result.to_csv(os.path.join(ticker_dir, 'metrics.csv'), index=False)
            else:
                print(f"Warning: No metrics collected for {ticker}")
            
            # Preserve checkpoint
            print(f"Preserving checkpoint in {os.path.join(args.checkpoints, model_id)}")
            
            print(f"Training completed successfully for {ticker}")
            

            
        finally:
            # Restore original stdout
            sys.stdout.close()
            sys.stdout = original_stdout
    
    # Save ticker-to-model mappings
    model_mappings.to_csv(model_mappings_file, index=False)
    print(f"Saved ticker-to-model mappings to: {model_mappings_file}")
    
    # Save consolidated results for all stocks
    all_stocks_results.to_csv(f'{results_dir}/all_stocks_metrics.csv', index=False)
    
    # Calculate and save average metrics across all stocks
    if not all_stocks_results.empty:
        avg_results = pd.DataFrame({
            'Ticker': ['AVERAGE'],
            'Period': [period],
            'MSE': [all_stocks_results['MSE'].mean()],
            'MAE': [all_stocks_results['MAE'].mean()],
            'RMSE': [all_stocks_results['RMSE'].mean()],
            'Accuracy': [all_stocks_results['Accuracy'].mean()],
            'MCC': [all_stocks_results['MCC'].mean()],
            'F1': [all_stocks_results['F1'].mean()],
            'Predicted StdDev': [all_stocks_results['Predicted StdDev'].mean()],
            'Volatility_Ratio': [all_stocks_results['Volatility_Ratio'].mean()],
            'Model_ID': ['AVERAGE']
        })
        avg_results.to_csv(f'{results_dir}/average_metrics_targetfeature.csv', index=False)
        
        print("\nAll stocks processed successfully")
        print(f"All stocks results saved to: {results_dir}/all_stocks_metrics.csv")
        print(f"Average metrics saved to: {results_dir}/average_metrics_targetfeature.csv")
    else:
        print("\nWarning: No stocks were successfully processed. No average metrics to save.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='backbone for Stock Forecasting')

    # Add the seed argument at the beginning
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='backbone',
                        help='model name, options: [backbone]')

    # Stock data loader
    parser.add_argument('--data', type=str, default='StockData_raw', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/raw_small', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='AAPL.csv', help='data file')
    parser.add_argument('--single_stock', type=int, default=0, help='whether to process single stock or directory (1=single, 0=multi)')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='Adj Close', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--scaler', type=str, default='price_standard', 
                        choices=['standard', 'price_standard', 'minmax', 'none'],
                        help='scaler to use for data normalization')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=10, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=10, help='prediction sequence length')
    parser.add_argument('--percent', type=int, default=100, help='percentage of data to use')

    # model define
    parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=6, help='output size')
    parser.add_argument('--d_model', type=int, default=768, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=768, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--timeenc', type=int, default=0, help='time encoding method')
    parser.add_argument('--word_embedding_path', type=str, default="wte_pca_500.pt")

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='COS', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--clip_grad', type=float, default=None, help='gradient clipping value')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # Lora parameters
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # GPT parameters
    parser.add_argument('--gpt_layers', type=int, default=6, help='number of hidden layers in gpt')

    # Visualization settings
    parser.add_argument('--visualization', type=int, default=1, help='enable visualization of predictions (0=disabled, 1=enabled)')
                   
    # Stats and alignment parameters
    parser.add_argument('--loss_stats', type=str, default='min,max', 
                   help='Comma-separated list of statistics to include in loss calculation (options: min,max,median,mean,trends)')
    parser.add_argument('--alignment_type', type=str, default='stats', 
                   choices=['stats', 'sequence'],
                   help='Type of alignment: statistical properties or reference sequence')
    parser.add_argument('--use_alignment', type=int, default=1,
                   help='Whether to use statistics alignment layer (1=use alignment, 0=disable alignment)')
    parser.add_argument('--ref_sequence_csv_dir', type=str, default=None,
                   help='Directory containing reference sequence CSV files named like <ticker>.csv')
    parser.add_argument('--ref_sequence_csv_path', type=str, default=None,
                   help='Path to the CSV file containing concatenated ground truth sequences for y')

    # Classifier-free guidance parameters
    parser.add_argument('--use_cfg', action='store_true', default=False,
                   help='Enable classifier-free guidance for conditionally guided generation')
    parser.add_argument('--p_uncond', type=float, default=0.1,
                   help='Probability of dropping conditioning during training (only used when use_cfg is enabled)')
    parser.add_argument('--guidance_scale', type=float, default=3.0,
                   help='Guidance scale for classifier-free guidance at inference time (higher values = stronger guidance)')

    # Testing only
    parser.add_argument('--eval_only', action='store_true', default=False,
                    help='run only evaluation without training using existing models')
                    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed_all(args.seed)

    print('Args in experiment:')
    print_args(args)

    # Generate a timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results/{args.model_id}_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created results directory at: {results_dir}")

    # Determine if we're processing a single stock or multiple stocks
    if args.single_stock:
        run_single_stock(args, timestamp, results_dir)
    else:
        run_multi_stock(args, timestamp, results_dir)
