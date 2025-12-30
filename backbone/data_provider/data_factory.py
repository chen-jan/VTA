from data_provider.data_loader import StockDataset_Raw
from torch.utils.data import DataLoader

data_dict = {
    'StockData_raw': StockDataset_Raw
}

def data_provider(args, flag, generator=None, vali=False):
    """
    Args:
        args: Configuration arguments
        flag: 'train', 'val', or 'test'
        generator: CUDA generator for data loader (optional)
        vali: Whether this is for validation data (legacy parameter)
    """
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    # Set defaults for missing arguments
    if not hasattr(args, 'freq'):
        args.freq = 'd'  # default to daily data
    if not hasattr(args, 'scaler'):
        args.scaler = 'price_standard'  # default to price_standard scaler for stock data

    # Set target to 'Adj Close' for stock data
    if not hasattr(args, 'target') or args.target == 'target':
        args.target = 'Adj Close'

    # Set batch sizes and shuffle flag based on the data split
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.eval_batch_size if hasattr(args, 'eval_batch_size') else args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
    
    # Create dataset
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=args.seasonal_patterns if hasattr(args, 'seasonal_patterns') else None,
        percent=args.percent if hasattr(args, 'percent') else 100,
        scaler=args.scaler
    )
    
    print(f"Dataset sizes - seq_len: {data_set.seq_len}, label_len: {data_set.label_len}, pred_len: {data_set.pred_len}")
    print(f"Dataset length for {flag}: {len(data_set)}")
    print(f"Using scaler: {args.scaler}")  # Debug print for scaler
    
    # Additional debug info for test dataset
    if flag == 'test':
        print("="*50)
        print("DEBUG: Test Dataset Length Calculation")
        print(f"data_x length: {len(data_set.data_x)}")
        print(f"seq_len: {data_set.seq_len}")
        print(f"pred_len: {data_set.pred_len}")
        print(f"Calculation: len(data_x) - seq_len - pred_len + 1 = {len(data_set.data_x)} - {data_set.seq_len} - {data_set.pred_len} + 1 = {len(data_set.data_x) - data_set.seq_len - data_set.pred_len + 1}")
        print("="*50)
    
    print(f"Target column: {args.target}")
    print(f"Features mode: {args.features}")
    
    if len(data_set) == 0:
        print(f"Warning: Empty dataset for {flag}!")
        return None, None

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        generator=generator
    )
    
    return data_set, data_loader
