import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class PriceStandardScaler:
    def __init__(self):
        print("PriceStandardScaler initialized")
        self.price_mean = 0.
        self.price_std = 1.
        self.volume_min = 0.
        self.volume_max = 1.
        self.epsilon = 1e-8  # Small constant to avoid division by zero
    
    def fit(self, data):
        # For price data, we want to scale based on the Adj Close price
        # In StockDataset_Raw, Adj Close is the last column in the price data
        adj_close_prices = data[:, -1]  # Get Adj Close prices
        self.price_mean = np.mean(adj_close_prices)
        self.price_std = np.std(adj_close_prices)
        
        # Scale volume between 0 and 1 using min-max normalization
        # In StockDataset_Raw, volume is the 5th column (index 4)
        volume = data[:, 4]
        self.volume_min = np.min(volume)
        self.volume_max = np.max(volume)
        
    def transform(self, data):
        # Create a copy to avoid modifying the original data
        transformed_data = data.copy()
        
        # Scale price columns (all except volume)
        # In StockDataset_Raw order: [open, high, low, close, volume, adj close]
        price_columns = [0, 1, 2, 3, 5]  # Open, High, Low, Close, Adj Close
        transformed_data[:, price_columns] = (transformed_data[:, price_columns] - self.price_mean) / self.price_std
        
        # Scale volume between 0 and 1
        transformed_data[:, 4] = (transformed_data[:, 4] - self.volume_min) / (self.volume_max - self.volume_min + self.epsilon)
        
        return transformed_data

    def inverse_transform(self, data):
        # Create a copy to avoid modifying the original data
        transformed_data = data.copy()
        
        # Inverse transform price columns
        # In StockDataset_Raw order: [open, high, low, close, volume, adj close]
        price_columns = [0, 1, 2, 3, 5]  # Open, High, Low, Close, Adj Close
        transformed_data[:, price_columns] = (transformed_data[:, price_columns] * self.price_std) + self.price_mean
        
        # Inverse transform volume from 0-1 scale
        transformed_data[:, 4] = (transformed_data[:, 4] * (self.volume_max - self.volume_min)) + self.volume_min
        
        return transformed_data

class MinMaxScaler:
    def __init__(self):
        self.min = 0.
        self.max = 1.
        self.epsilon = 1e-8  # Small constant to avoid division by zero
    
    def fit(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        
    def transform(self, data):
        return (data - self.min) / (self.max - self.min + self.epsilon)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min

class StandardScaler:
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        
    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class StockDataset_Raw(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', target='Adj Close', scale=True,
                 data_path=None, timeenc=0, freq='d', embed='timeF',
                 scaler='price_standard', seasonal_patterns=None, percent=100):
        # Initialize parameters
        self.root_path = root_path
        self.data_path = data_path
        
        self.flag = flag
        self.timeenc = timeenc
        self.embed = embed
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.features = features
        self.target = target
        self.freq = freq
        self.scale = scale
        self.percent = percent
        
        # Initialize scaler based on parameter
        if scaler == 'standard':
            self.scaler = StandardScaler()
        elif scaler == 'price_standard':
            self.scaler = PriceStandardScaler()
        elif scaler == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaler type: {scaler}")
        
        self.__read_data__()
        
    def __read_data__(self):
        # Add debug prints
        print(f"\n=== Raw Stock Data Loading ({self.flag}) ===")
        print(f"Root path: {self.root_path}")
        print(f"Data path: {self.data_path}")
        print(f"Target: {self.target}")
        
        # Read CSV file
        if self.data_path:
            file_path = os.path.join(self.root_path, self.data_path)
        else:
            # Use all files in the directory
            file_path = self.root_path
            
        print(f"Using data from: {file_path}")
        
        if os.path.isfile(file_path):
            df_raw = pd.read_csv(file_path)
            print(f"Single file loaded with {len(df_raw)} rows")
        else:
            # This means we're loading from a directory
            print(f"Error: Expected file but got directory {file_path}")
            raise ValueError(f"Expected file but got directory {file_path}")
        
        # Find column names case-insensitively
        column_mapping = {}
        for col in ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']:
            for actual_col in df_raw.columns:
                if actual_col.lower() == col.lower():
                    column_mapping[col] = actual_col
                    break
        
        print(f"Found columns: {column_mapping}")
        
        # Check if we have the date column
        if 'date' not in column_mapping:
            print(f"Warning: No date column found in {self.data_path}, creating synthetic dates")
            df_raw['date'] = pd.date_range(start='2000-01-01', periods=len(df_raw))
        else:
            # Convert to datetime properly
            df_raw['date'] = pd.to_datetime(df_raw[column_mapping['date']])
        
        # Process columns based on feature type
        if self.features == 'S':
            # Univariate case - only use target column
            cols = [self.target]
        elif self.features == 'MS' or self.features == 'M':
            # Multivariate case - use all financial features
            cols = []
            for col in ['open', 'high', 'low', 'close', 'adj close', 'volume']:
                if col in column_mapping:
                    cols.append(column_mapping[col])
            
            # For MS mode, ensure target column is the last column
            if self.features == 'MS':
                # Check if target column exists in dataframe
                if self.target not in df_raw.columns:
                    # Try to find it case-insensitively
                    for col in df_raw.columns:
                        if col.lower() == self.target.lower():
                            self.target = col
                            break
                    else:
                        print(f"Warning: Target column '{self.target}' not found. Using 'Adj Close' or first available column.")
                        self.target = 'Adj Close' if 'Adj Close' in df_raw.columns else df_raw.columns[1]
                
                # If target is already in cols, remove it first to avoid duplication
                if self.target in cols:
                    cols.remove(self.target)
                
                # Now add the target as the last column
                cols.append(self.target)
                print(f"Rearranged columns for MS mode. Target '{self.target}' is now the last column.")
        
        # Check if target column exists - This is now only needed for 'S' mode
        # since for 'MS' mode we already handled it above
        if self.features != 'MS' and self.target not in df_raw.columns:
            # Try to find it case-insensitively
            for col in df_raw.columns:
                if col.lower() == self.target.lower():
                    self.target = col
                    break
            else:
                print(f"Warning: Target column '{self.target}' not found. Using 'Adj Close' or first available column.")
                self.target = 'Adj Close' if 'Adj Close' in df_raw.columns else df_raw.columns[1]
        
        print(f"Using target column: {self.target}")
        print(f"Features: {cols}")
        
        # Calculate dataset splits
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        print(f"Split sizes - Train: {num_train}, Val: {num_vali}, Test: {num_test}")
        
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        
        train_indices = [0]  # train
        val_indices = [1]    # val
        test_indices = [2]   # test
        
        if self.flag == 'train':
            border1 = border1s[train_indices[0]]
            border2 = border2s[train_indices[0]]
        elif self.flag == 'val':
            border1 = border1s[val_indices[0]]
            border2 = border2s[val_indices[0]]
        elif self.flag == 'test':
            border1 = border1s[test_indices[0]]
            border2 = border2s[test_indices[0]]
        else:
            raise ValueError(f"Invalid flag: {self.flag}")
            
        print(f"Borders for current split ({self.flag}) - Start: {border1}, End: {border2}")
        
        # Get time features
        if self.timeenc == 0:
            # Handcrafted features
            df_stamp = pd.DataFrame({'date': df_raw['date'][border1:border2]})
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)

            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # Timeenc features
            data_stamp = time_features(pd.to_datetime(df_raw['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # Scale the data
        df_data = df_raw[cols]
        if self.scaler is not None and self.scale:
            train_data = df_data[border1s[0]:border2s[0]].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # Set class attributes
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2] if self.timeenc == 0 else data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        if self.scaler is not None and self.scale:
            return self.scaler.inverse_transform(data)
        return data
