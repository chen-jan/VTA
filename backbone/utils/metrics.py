import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean()


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def SMAPE(pred, true):
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))
    # return np.mean(200 * np.abs(pred - true) / (pred + true + 1e-8))


def ND(pred, true):
    return np.mean(np.abs(true - pred)) / np.mean(np.abs(true))


def calculate_directional_accuracy(pred, true):
    """Calculate directional accuracy (up/down movement prediction)"""
    # Calculate daily returns
    pred_returns = np.diff(pred, axis=0)
    true_returns = np.diff(true, axis=0)
    
    # Convert to binary signals (1 for up, 0 for down/same)
    pred_direction = (pred_returns > 0).astype(int)
    true_direction = (true_returns > 0).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(true_direction.flatten(), pred_direction.flatten())
    mcc = matthews_corrcoef(true_direction.flatten(), pred_direction.flatten())
    f1 = f1_score(true_direction.flatten(), pred_direction.flatten())
    
    return accuracy, mcc, f1


def calculate_std_dev(pred, true):
    """Calculate standard deviation of predictions and ground truth"""
    # Handle multi-feature case - use only target feature (last feature)
    pred = pred[..., -1]  
    true = true[..., -1] 
    
    # Calculate standard deviation across the second dimension (axis=1) and take mean across sequences
    pred_std = np.mean(np.std(pred, axis=1), axis=0)
    true_std = np.mean(np.std(true, axis=1), axis=0)
    
    # Calculate ratio of prediction volatility to true volatility
    vol_ratio = pred_std / np.maximum(true_std, 1e-10)
    
    return pred_std, true_std, vol_ratio


def metric(pred, true):
    """Calculate comprehensive set of metrics"""
    # Debug prints for metrics
    print("Metrics Debug Info:")
    print(f"Pred shape: {pred.shape}")
    print(f"True shape: {true.shape}")
    print(f"Pred min/max/mean: {np.min(pred):.4f}/{np.max(pred):.4f}/{np.mean(pred):.4f}")
    print(f"True min/max/mean: {np.min(true):.4f}/{np.max(true):.4f}/{np.mean(true):.4f}")
    
    # Traditional regression metrics
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    
    # Directional metrics for forecasting
    accuracy, mcc, f1 = calculate_directional_accuracy(pred, true)
    
    # Volatility metrics
    pred_std, true_std, vol_ratio = calculate_std_dev(pred, true)
    
    # Print all metrics
    print('Traditional Metrics:')
    print(f'MAE:\t{mae:.4f}')
    print(f'MSE:\t{mse:.4f}')
    print(f'RMSE:\t{rmse:.4f}')
    print(f'MAPE:\t{mape:.4f}')
    print(f'MSPE:\t{mspe:.4f}')
    print(f'RSE:\t{rse:.4f}')
    print(f'CORR:\t{corr:.4f}')
    
    print('\nDirectional Metrics:')
    print(f'Accuracy:\t{accuracy:.4f}')
    print(f'MCC:\t{mcc:.4f}')
    print(f'F1:\t{f1:.4f}')
    
    print('\nVolatility Metrics:')
    print(f'Pred StdDev:\t{pred_std:.4f}')
    print(f'True StdDev:\t{true_std:.4f}')
    print(f'Volatility Ratio:\t{vol_ratio:.4f}')
    
    return mae, mse, rmse, mape, mspe, rse, corr, accuracy, mcc, f1, pred_std, true_std, vol_ratio
