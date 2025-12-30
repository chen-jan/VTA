import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import random
import talib
import glob
import sys
import time
from datetime import datetime
import argparse

# Data loader imports
from backbone.data_provider.data_factory import data_provider
from backbone.utils.technical_indicators import TechnicalIndicators
from backbone.utils.StandardNorm import Normalize

ROOT_PATH = os.path.join(os.path.dirname(__file__), 'backbone', 'data', 'stocknet')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'grpo', 'data', 'stocknet')
SEQ_LEN = 10
PRED_LEN = 10
LABEL_LEN = 0
BATCH_SIZE = 16
TARGET = 'Adj Close'
FEATURES = 'MS'
SCALER = 'price_standard'
SINGLE_STOCK = None
USE_GPU = True
NUM_WORKERS = 0
FREQ = 'd'
SEED = 2021

# Optional CLI overrides for portability
def _parse_cli_overrides():
    parser = argparse.ArgumentParser(description='VTA: Combined preprocessor for stock CSVs')
    parser.add_argument('--root_path', type=str, default=ROOT_PATH, help='Input CSV folder (e.g., backbone/data/stocknet)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Output folder for processed CSVs (e.g., grpo/data/stocknet)')
    parser.add_argument('--single_stock', type=str, default=SINGLE_STOCK, help='Optional single stock symbol (e.g., AAPL)')
    parser.add_argument('--seq_len', type=int, default=SEQ_LEN)
    parser.add_argument('--pred_len', type=int, default=PRED_LEN)
    parser.add_argument('--label_len', type=int, default=LABEL_LEN)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--target', type=str, default=TARGET)
    parser.add_argument('--features', type=str, default=FEATURES)
    parser.add_argument('--scaler', type=str, default=SCALER)
    parser.add_argument('--use_gpu', action='store_true', default=USE_GPU)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--freq', type=str, default=FREQ)
    parser.add_argument('--seed', type=int, default=SEED)
    # Only parse when run as script
    if __name__ == '__main__':
        return parser.parse_args()
    class _Empty:
        pass
    return _Empty()

_cli = _parse_cli_overrides()
if hasattr(_cli, 'root_path'):
    ROOT_PATH = os.path.abspath(getattr(_cli, 'root_path', ROOT_PATH))
    OUTPUT_DIR = os.path.abspath(getattr(_cli, 'output_dir', OUTPUT_DIR))
    SINGLE_STOCK = getattr(_cli, 'single_stock', SINGLE_STOCK)
    SEQ_LEN = getattr(_cli, 'seq_len', SEQ_LEN)
    PRED_LEN = getattr(_cli, 'pred_len', PRED_LEN)
    LABEL_LEN = getattr(_cli, 'label_len', LABEL_LEN)
    BATCH_SIZE = getattr(_cli, 'batch_size', BATCH_SIZE)
    TARGET = getattr(_cli, 'target', TARGET)
    FEATURES = getattr(_cli, 'features', FEATURES)
    SCALER = getattr(_cli, 'scaler', SCALER)
    USE_GPU = getattr(_cli, 'use_gpu', USE_GPU)
    NUM_WORKERS = getattr(_cli, 'num_workers', NUM_WORKERS)
    FREQ = getattr(_cli, 'freq', FREQ)
    SEED = getattr(_cli, 'seed', SEED)

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

SPLITS = ['train', 'val', 'test']

SELECTED_INDICATORS = [
    'sma',      # Simple Moving Average
    'ema',      # Weighted Moving Average
    'mom',      # Momentum
    'stoch',    # Stochastic K and D
    'rsi',      # Relative Strength Index
    'macd',     # MACD
    'willr',    # Williams %R
    'cci',      # Commodity Channel Index
    'adx',      # Average Directional Index
    'bbands'    # Bollinger Bands
]

seq_columns = ['sequence']

numeric_columns = [
    'min', 'min_timestep', 'max', 'max_timestep', 'mean', 'median', 
    'median_timestep', 'sma', 'ema', 'mom', 'stoch_k', 'stoch_d', 
    'rsi', 'macd', 'willr', 'cci', 'adx', 'bbands_upper', 
    'bbands_middle', 'bbands_lower'
]

columns = seq_columns + numeric_columns + ['batch_idx', 'stock', 'stock_symbol']

class StockDataProcessor:
    """
    Processes stock data, computes stats/indicators, and writes CSVs directly.
    """
    
    def __init__(self):
        """Initialize with global parameters."""
        # Create an args object to match the expected interface
        class DataArgs:
            def __init__(self):
                self.root_path = ROOT_PATH
                self.data_path = None  # Will be set for each stock
                self.seq_len = SEQ_LEN
                self.pred_len = PRED_LEN
                self.label_len = LABEL_LEN
                self.batch_size = BATCH_SIZE
                self.enc_in = 6  # Fixed for stock data (OHLCV + Adj Close)
                self.features = FEATURES
                self.target = TARGET
                self.scaler = SCALER
                self.data = 'StockData_raw'
                self.freq = FREQ
                self.num_workers = NUM_WORKERS
                self.use_gpu = USE_GPU
                self.top_k = 5  # For top lags calculation
                # Required by data_provider
                self.embed = 'timeF'
                self.eval_batch_size = BATCH_SIZE
        
        self.args = DataArgs()
        self.output_dir = OUTPUT_DIR
        self.single_stock = SINGLE_STOCK
        self.device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
        
        # Set up normalization
        self.normalize_layers = Normalize(self.args.enc_in, affine=False)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize dictionary to store data for each stock
        self.stock_data = {}
        
        # Keep track of processing time
        self.start_time = time.time()
        
        print(f"Initialized processor with device: {self.device}")
        print(f"Sequence length: {self.args.seq_len}, Prediction length: {self.args.pred_len}")
        print(f"Batch size: {self.args.batch_size}, Target: {self.args.target}")
        print(f"Output directory: {self.output_dir}")
        if self.single_stock:
            print(f"Processing single stock: {self.single_stock}")
    
    def load_data(self, flag='train'):
        """Load data using the data_provider."""
        generator = torch.Generator(device='cpu')
        data_set, data_loader = data_provider(self.args, flag, generator=generator)
        return data_set, data_loader
    
    def calculate_lags(self, x_enc):
        """
        Calculate autocorrelation lags using FFT for time series analysis.
        """
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.args.top_k, dim=-1)
        return lags
    
    def process_batch(self, batch_x, batch_y, batch_idx, stock_name):
        """
        Process a batch to extract statistics and indicators.
        Also perform normalization and get normalized statistics.
        
        Returns:
            stats: Dictionary of statistics for each sequence
        """
        # Store original batch data
        original_x = batch_x.clone()
        original_y = batch_y.clone()
        
        # Move data to appropriate device
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        
        # Normalize the data
        normalized_x = self.normalize_layers(batch_x, 'norm')
        normalized_y = self.normalize_layers._normalize(batch_y)
        
        # Extract last feature for statistics - Original data
        original_x_last = original_x[:, :, 5].unsqueeze(-1)  # Shape: [B, T, 1] - Adj Close
        original_y_last = original_y[:, :, 5].unsqueeze(-1)  # Shape: [B, T, 1] - Adj Close
        
        # Extract last feature for statistics - Normalized data
        normalized_x_last = normalized_x[:, :, 5].unsqueeze(-1)  # Shape: [B, T, 1] - Adj Close
        normalized_y_last = normalized_y[:, :, 5].unsqueeze(-1)  # Shape: [B, T, 1] - Adj Close
        
        # Calculate statistics for original input (original_x) - only last feature
        original_x_min_values, original_x_min_indices = torch.min(original_x_last, dim=1)
        original_x_max_values, original_x_max_indices = torch.max(original_x_last, dim=1)
        original_x_means = torch.mean(original_x_last, dim=1)
        original_x_medians, original_x_median_indices = torch.median(original_x_last, dim=1)
        original_x_lags = self.calculate_lags(original_x_last)
        original_x_trends = original_x_last.diff(dim=1).sum(dim=1)
        
        # Calculate statistics for original target (original_y) - only last feature
        original_y_min_values, original_y_min_indices = torch.min(original_y_last, dim=1)
        original_y_max_values, original_y_max_indices = torch.max(original_y_last, dim=1)
        original_y_means = torch.mean(original_y_last, dim=1)
        original_y_medians, original_y_median_indices = torch.median(original_y_last, dim=1)
        original_y_lags = self.calculate_lags(original_y_last)
        original_y_trends = original_y_last.diff(dim=1).sum(dim=1)
        
        # Extract the OHLC components from original_x
        open_prices_original_x = original_x[:, :, 0]  # First feature - Open
        high_prices_original_x = original_x[:, :, 1]  # Second feature - High 
        low_prices_original_x = original_x[:, :, 2]   # Third feature - Low
        close_prices_original_x = original_x[:, :, 3]  # Fourth feature - Close
        volume_original_x = original_x[:, :, 4]        # Fifth feature - Volume
        adj_close_original_x = original_x[:, :, 5]     # Sixth feature - Adj Close

        # Extract the OHLC components from original batch_y
        open_prices_original_y = original_y[:, :, 0]
        high_prices_original_y = original_y[:, :, 1]
        low_prices_original_y = original_y[:, :, 2]
        close_prices_original_y = original_y[:, :, 3]
        volume_original_y = original_y[:, :, 4]
        adj_close_original_y = original_y[:, :, 5]
        
        # Calculate indicators for original_x
        tech_ind_original_x = TechnicalIndicators(
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            open_prices=open_prices_original_x,
            high_prices=high_prices_original_x,
            low_prices=low_prices_original_x,
            close_prices=close_prices_original_x,
            volume=volume_original_x
        )
        tech_ind_original_x.set_selected_indicators(SELECTED_INDICATORS)
        raw_indicators_original_x = tech_ind_original_x.calculate_indicators()

        # Calculate indicators for original_y
        tech_ind_original_y = TechnicalIndicators(
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            open_prices=open_prices_original_y,
            high_prices=high_prices_original_y,
            low_prices=low_prices_original_y,
            close_prices=close_prices_original_y,
            volume=volume_original_y
        )
        tech_ind_original_y.set_selected_indicators(SELECTED_INDICATORS)
        raw_indicators_original_y = tech_ind_original_y.calculate_indicators()
        
        # Restructure indicators
        def restructure_indicators(raw_indicators, batch_size):
            indicators = {}
            for ind_type in SELECTED_INDICATORS:
                indicators[ind_type] = []
                for i in range(batch_size):
                    if f"{ind_type}_{i}" in raw_indicators:
                        indicators[ind_type].append(raw_indicators[f"{ind_type}_{i}"])
                    else:
                        # If indicator not found for this batch item, add None or empty array
                        indicators[ind_type].append(None)
                # Use object array instead of trying to create homogeneous array
                indicators[ind_type] = np.array(indicators[ind_type], dtype=object)
            return indicators

        batch_size = batch_x.size(0)
        indicators_original_x = restructure_indicators(raw_indicators_original_x, batch_size)
        indicators_original_y = restructure_indicators(raw_indicators_original_y, batch_size)
        
        # Create list of sequences for each sample in the batch
        sequences_original_x = []
        sequences_original_y = []
        
        for i in range(batch_size):
            # Only store the adjusted close values (last feature, index 5)
            sequences_original_x.append(original_x[i, :, 5].cpu().numpy())
            sequences_original_y.append(original_y[i, :, 5].cpu().numpy())
        
        # Prepare stats dictionary with all sets of statistics
        x_stats = {
            'sequence_data': sequences_original_x,
            'min': original_x_min_values.cpu().numpy(),
            'min_timestep': original_x_min_indices.cpu().numpy(),
            'max': original_x_max_values.cpu().numpy(),
            'max_timestep': original_x_max_indices.cpu().numpy(),
            'mean': original_x_means.cpu().numpy(),
            'median': original_x_medians.cpu().numpy(),
            'median_timestep': original_x_median_indices.cpu().numpy(),
            'trends': original_x_trends.cpu().numpy(),
        }
        
        y_stats = {
            'sequence_data': sequences_original_y,
            'min': original_y_min_values.cpu().numpy(),
            'min_timestep': original_y_min_indices.cpu().numpy(),
            'max': original_y_max_values.cpu().numpy(),
            'max_timestep': original_y_max_indices.cpu().numpy(),
            'mean': original_y_means.cpu().numpy(),
            'median': original_y_medians.cpu().numpy(),
            'median_timestep': original_y_median_indices.cpu().numpy(),
            'trends': original_y_trends.cpu().numpy(),
        }
        
        # Safely add technical indicators to stats dictionary
        for ind_type in SELECTED_INDICATORS:
            # Add technical indicators for all data types
            if ind_type in indicators_original_x:
                x_stats[ind_type] = indicators_original_x[ind_type]
            
            if ind_type in indicators_original_y:
                y_stats[ind_type] = indicators_original_y[ind_type]
            
            # Add special handling for multi-dimensional indicators
            if ind_type == 'stoch' and 'stoch' in indicators_original_x:
                try:
                    x_stats['stoch_k'] = np.array([x[0] if x is not None else None for x in indicators_original_x['stoch']], dtype=object)
                    x_stats['stoch_d'] = np.array([x[1] if x is not None else None for x in indicators_original_x['stoch']], dtype=object)
                except (IndexError, TypeError):
                    pass
                    
            if ind_type == 'stoch' and 'stoch' in indicators_original_y:
                try:
                    y_stats['stoch_k'] = np.array([x[0] if x is not None else None for x in indicators_original_y['stoch']], dtype=object)
                    y_stats['stoch_d'] = np.array([x[1] if x is not None else None for x in indicators_original_y['stoch']], dtype=object)
                except (IndexError, TypeError):
                    pass
                    
            # Add bbands handling
            if ind_type == 'bbands' and 'bbands' in indicators_original_x:
                try:
                    x_stats['bbands_upper'] = np.array([x[0] if x is not None else None for x in indicators_original_x['bbands']], dtype=object)
                    x_stats['bbands_middle'] = np.array([x[1] if x is not None else None for x in indicators_original_x['bbands']], dtype=object)
                    x_stats['bbands_lower'] = np.array([x[2] if x is not None else None for x in indicators_original_x['bbands']], dtype=object)
                except (IndexError, TypeError):
                    pass
                    
            if ind_type == 'bbands' and 'bbands' in indicators_original_y:
                try:
                    y_stats['bbands_upper'] = np.array([x[0] if x is not None else None for x in indicators_original_y['bbands']], dtype=object)
                    y_stats['bbands_middle'] = np.array([x[1] if x is not None else None for x in indicators_original_y['bbands']], dtype=object)
                    y_stats['bbands_lower'] = np.array([x[2] if x is not None else None for x in indicators_original_y['bbands']], dtype=object)
                except (IndexError, TypeError):
                    pass
                    
            # Add macd handling
            if ind_type == 'macd' and 'macd' in indicators_original_x:
                try:
                    x_stats['macd'] = np.array([x[0] if x is not None else None for x in indicators_original_x['macd']], dtype=object)
                except (IndexError, TypeError):
                    pass
                    
            if ind_type == 'macd' and 'macd' in indicators_original_y:
                try:
                    y_stats['macd'] = np.array([x[0] if x is not None else None for x in indicators_original_y['macd']], dtype=object)
                except (IndexError, TypeError):
                    pass
        
        # Initialize stock data structure if this is the first time seeing this stock
        if stock_name not in self.stock_data:
            self.stock_data[stock_name] = {}
            for split in SPLITS:
                self.stock_data[stock_name][split] = {
                    'original_x_stats': [],
                    'original_y_stats': []
                }
            # Add a "full" split to store all data combined
            self.stock_data[stock_name]['full'] = {
                'original_x_stats': [],
                'original_y_stats': []
            }
        
        # Add batch and stock info to each example
        for i in range(batch_size):
            # Create a row for x_stats
            x_row = {}
            
            # Add sequence data
            x_row['sequence'] = str(x_stats['sequence_data'][i].tolist())
            
            # Add other stats
            for col in numeric_columns:
                if col in x_stats:
                    value = x_stats[col][i]
                    if isinstance(value, np.ndarray):
                        x_row[col] = str(value.tolist())
                    else:
                        x_row[col] = str([float(value)] if isinstance(value, (int, float, np.number)) else value)
                else:
                    x_row[col] = str([])
            
            # Add metadata
            x_row['batch_idx'] = batch_idx
            x_row['stock'] = stock_name
            x_row['stock_symbol'] = stock_name
            
            # Add to stock data
            self.stock_data[stock_name][self.current_split]['original_x_stats'].append(x_row)
            
            # Also add to full dataset
            self.stock_data[stock_name]['full']['original_x_stats'].append(x_row)
            
            # Create a row for y_stats
            y_row = {}
            
            # Add sequence data
            y_row['sequence'] = str(y_stats['sequence_data'][i].tolist())
            
            # Add other stats
            for col in numeric_columns:
                if col in y_stats:
                    value = y_stats[col][i]
                    if isinstance(value, np.ndarray):
                        y_row[col] = str(value.tolist())
                    else:
                        y_row[col] = str([float(value)] if isinstance(value, (int, float, np.number)) else value)
                else:
                    y_row[col] = str([])
            
            # Add metadata
            y_row['batch_idx'] = batch_idx
            y_row['stock'] = stock_name
            y_row['stock_symbol'] = stock_name
            
            # Add to stock data
            self.stock_data[stock_name][self.current_split]['original_y_stats'].append(y_row)
            
            # Also add to full dataset
            self.stock_data[stock_name]['full']['original_y_stats'].append(y_row)
            
    def process_stock(self, stock_file):
        """Process all batches for a stock and add to stock data."""
        print(f"\nProcessing stock: {stock_file}")
        
        # Set the stock file
        self.args.data_path = stock_file
        stock_name = os.path.splitext(stock_file)[0]
        
        # Process each split
        for split in SPLITS:
            print(f"Processing {split} split...")
            self.current_split = split
            
            # Load data
            data_set, data_loader = self.load_data(flag=split)
            
            if data_loader is None:
                print(f"No data available for {split} split!")
                continue
            
            print(f"Processing {len(data_loader)} batches...")
            
            # Process all batches
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(data_loader)):
                self.process_batch(batch_x, batch_y, batch_idx, stock_name)
                
            print(f"Added {len(self.stock_data[stock_name][split]['original_x_stats'])} rows for {split} split.")
        
        # Write this stock's data to CSV after processing all splits
        self.write_stock_to_csv(stock_name)

    def write_stock_to_csv(self, stock_name):
        """Write data for a single stock to CSV files."""
        print(f"\nWriting data for {stock_name} to CSV files...")
        
        # Create a directory for this stock
        stock_dir = os.path.join(self.output_dir, stock_name)
        os.makedirs(stock_dir, exist_ok=True)
        
        # Combine train and val data
        train_data_x = self.stock_data[stock_name]['train']['original_x_stats'] + \
                      self.stock_data[stock_name]['val']['original_x_stats']
        train_data_y = self.stock_data[stock_name]['train']['original_y_stats'] + \
                      self.stock_data[stock_name]['val']['original_y_stats']
        
        # Create file names with stock name prefix
        train_x_filename = os.path.join(stock_dir, f"{stock_name}_train_x.csv")
        train_y_filename = os.path.join(stock_dir, f"{stock_name}_train_y.csv")
        test_x_filename = os.path.join(stock_dir, f"{stock_name}_test_x.csv")
        test_y_filename = os.path.join(stock_dir, f"{stock_name}_test_y.csv")
        full_x_filename = os.path.join(stock_dir, f"{stock_name}_full_x.csv")
        full_y_filename = os.path.join(stock_dir, f"{stock_name}_full_y.csv")
        
        # Get test data
        test_data_x = self.stock_data[stock_name]['test']['original_x_stats']
        test_data_y = self.stock_data[stock_name]['test']['original_y_stats']
        
        # Get full data
        full_data_x = self.stock_data[stock_name]['full']['original_x_stats']
        full_data_y = self.stock_data[stock_name]['full']['original_y_stats']
        
        # Write train data (combined train+val) to CSV
        if len(train_data_x) > 0:
            with open(train_x_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                writer.writerows(train_data_x)
            print(f"Exported {len(train_data_x)} rows to {train_x_filename}")
        
        if len(train_data_y) > 0:
            with open(train_y_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                writer.writerows(train_data_y)
            print(f"Exported {len(train_data_y)} rows to {train_y_filename}")
        
        # Write test data to CSV
        if len(test_data_x) > 0:
            with open(test_x_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                writer.writerows(test_data_x)
            print(f"Exported {len(test_data_x)} rows to {test_x_filename}")
        
        if len(test_data_y) > 0:
            with open(test_y_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                writer.writerows(test_data_y)
            print(f"Exported {len(test_data_y)} rows to {test_y_filename}")
        
        # Write full data to CSV
        if len(full_data_x) > 0:
            with open(full_x_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                writer.writerows(full_data_x)
            print(f"Exported {len(full_data_x)} rows to {full_x_filename}")
        
        if len(full_data_y) > 0:
            with open(full_y_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                writer.writerows(full_data_y)
            print(f"Exported {len(full_data_y)} rows to {full_y_filename}")
        
        # Clean up memory after writing
        del self.stock_data[stock_name]

    def write_to_csv(self):
        """Process any remaining stock data that hasn't been written yet."""
        print("\nWriting any remaining data to CSV files...")
        
        for stock_name in self.stock_data:
            self.write_stock_to_csv(stock_name)

    def print_summary(self):
        """Print a summary of the processing results."""
        print("\n" + "="*80)
        print("PROCESSING SUMMARY")
        print("="*80)
        
        print(f"Total processing time: {(time.time() - self.start_time)/60:.2f} minutes")
        print(f"Output directory: {self.output_dir}")
        print("="*80)
        
    def consolidate_all_data(self):
        """
        Consolidate data from all stocks into a single folder with combined CSV files.
        Creates an 'all_data' folder with consolidated train, test and full datasets.
        """
        print("\nConsolidating all stock data into a single folder...")
        
        # Create all_data directory
        all_data_dir = os.path.join(self.output_dir, 'all_data')
        os.makedirs(all_data_dir, exist_ok=True)
        
        # Initialize dictionaries to store combined data
        all_train_x = []
        all_train_y = []
        all_test_x = []
        all_test_y = []
        all_full_x = []
        all_full_y = []
        
        # Get list of all stock directories
        stock_dirs = [d for d in os.listdir(self.output_dir) 
                     if os.path.isdir(os.path.join(self.output_dir, d)) 
                     and d != 'all_data']
        
        # Loop through each stock directory and read its CSV data
        for stock_dir in tqdm(stock_dirs, desc="Processing stock directories"):
            stock_path = os.path.join(self.output_dir, stock_dir)
            stock_name = stock_dir
            
            # Read train files
            train_x_file = os.path.join(stock_path, f"{stock_name}_train_x.csv")
            train_y_file = os.path.join(stock_path, f"{stock_name}_train_y.csv")
            
            # Read test files
            test_x_file = os.path.join(stock_path, f"{stock_name}_test_x.csv")
            test_y_file = os.path.join(stock_path, f"{stock_name}_test_y.csv")
            
            # Read full files
            full_x_file = os.path.join(stock_path, f"{stock_name}_full_x.csv")
            full_y_file = os.path.join(stock_path, f"{stock_name}_full_y.csv")
            
            # Read and append train data
            if os.path.exists(train_x_file):
                train_x_data = pd.read_csv(train_x_file)
                all_train_x.append(train_x_data)
            
            if os.path.exists(train_y_file):
                train_y_data = pd.read_csv(train_y_file)
                all_train_y.append(train_y_data)
            
            # Read and append test data
            if os.path.exists(test_x_file):
                test_x_data = pd.read_csv(test_x_file)
                all_test_x.append(test_x_data)
            
            if os.path.exists(test_y_file):
                test_y_data = pd.read_csv(test_y_file)
                all_test_y.append(test_y_data)
            
            # Read and append full data
            if os.path.exists(full_x_file):
                full_x_data = pd.read_csv(full_x_file)
                all_full_x.append(full_x_data)
            
            if os.path.exists(full_y_file):
                full_y_data = pd.read_csv(full_y_file)
                all_full_y.append(full_y_data)
        
        # Concatenate all data frames
        print("Concatenating all data frames...")
        
        # Combine train data and save
        if all_train_x:
            combined_train_x = pd.concat(all_train_x, ignore_index=True)
            combined_train_x.to_csv(os.path.join(all_data_dir, "all_data_train_x.csv"), index=False)
            print(f"Exported {len(combined_train_x)} rows to all_data_train_x.csv")
        
        if all_train_y:
            combined_train_y = pd.concat(all_train_y, ignore_index=True)
            combined_train_y.to_csv(os.path.join(all_data_dir, "all_data_train_y.csv"), index=False)
            print(f"Exported {len(combined_train_y)} rows to all_data_train_y.csv")
        
        # Combine test data and save
        if all_test_x:
            combined_test_x = pd.concat(all_test_x, ignore_index=True)
            combined_test_x.to_csv(os.path.join(all_data_dir, "all_data_test_x.csv"), index=False)
            print(f"Exported {len(combined_test_x)} rows to all_data_test_x.csv")
        
        if all_test_y:
            combined_test_y = pd.concat(all_test_y, ignore_index=True)
            combined_test_y.to_csv(os.path.join(all_data_dir, "all_data_test_y.csv"), index=False)
            print(f"Exported {len(combined_test_y)} rows to all_data_test_y.csv")
        
        # Combine full data and save
        if all_full_x:
            combined_full_x = pd.concat(all_full_x, ignore_index=True)
            combined_full_x.to_csv(os.path.join(all_data_dir, "all_data_full_x.csv"), index=False)
            print(f"Exported {len(combined_full_x)} rows to all_data_full_x.csv")
        
        if all_full_y:
            combined_full_y = pd.concat(all_full_y, ignore_index=True)
            combined_full_y.to_csv(os.path.join(all_data_dir, "all_data_full_y.csv"), index=False)
            print(f"Exported {len(combined_full_y)} rows to all_data_full_y.csv")
        
        print(f"All stock data consolidated in {all_data_dir}")

def main():
    """Main function to run the data processing."""
    set_seed(SEED)
    
    print(f"Starting individual stock data processing")
    print(f"Using device: {'GPU' if USE_GPU and torch.cuda.is_available() else 'CPU'}")
    
    # Get list of stock files in root directory
    if not os.path.exists(ROOT_PATH):
        print(f"Error: Root path {ROOT_PATH} does not exist!")
        return
    
    # Get stock files
    if SINGLE_STOCK:
        stock_files = [f"{SINGLE_STOCK}.csv"]
        # Check if the file exists
        if not os.path.exists(os.path.join(ROOT_PATH, stock_files[0])):
            print(f"Error: Stock file for {SINGLE_STOCK} not found!")
            return
    else:
        stock_files = [f for f in os.listdir(ROOT_PATH) if f.endswith('.csv')]
    
    print(f"Found {len(stock_files)} CSV files to process")
    
    # Create processor
    processor = StockDataProcessor()
    
    # Process each stock file and write its data immediately
    for stock_file in stock_files:
        processor.process_stock(stock_file)
        
    # Write any remaining data to CSV (though this should be handled by process_stock now)
    processor.write_to_csv()
    
    # Consolidate all stock data into a single folder
    processor.consolidate_all_data()
    
    # Print summary
    processor.print_summary()

if __name__ == "__main__":
    main() 