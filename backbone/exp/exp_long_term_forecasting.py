from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from models.backbone import Model
import json
import ast

warnings.filterwarnings('ignore')

# custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args, folder_path=None):
        self.full_ref_sequences = None
        self.train_set_size = 0
        self.val_set_size = 0
        self.test_set_size = 0
        
        super(Exp_Long_Term_Forecast, self).__init__(args)
        
        self.folder_path = folder_path or './results'
        os.makedirs(self.folder_path, exist_ok=True)
        
        # visualization
        self.visualization = 1 if not hasattr(args, 'visualization') else args.visualization
        if self.visualization:
            print("Visualization enabled - plots will be created during testing")
        
        # load reference sequences
        if hasattr(args, 'ref_sequence_csv_path') and args.ref_sequence_csv_path:
            if os.path.exists(args.ref_sequence_csv_path):
                print(f"Loading reference sequences from: {args.ref_sequence_csv_path}")
                try:
                    df_ref = pd.read_csv(args.ref_sequence_csv_path)
                    if 'grpo_prediction' in df_ref.columns:
                        parsed_sequences = df_ref['grpo_prediction'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32)).tolist()
                        self.full_ref_sequences = np.array(parsed_sequences)
                        print(f"Successfully loaded and parsed {len(self.full_ref_sequences)} reference sequences from 'grpo_prediction' column.")
                    else:
                        print(f"Error: Column 'grpo_prediction' not found in {args.ref_sequence_csv_path}")
                        self.full_ref_sequences = None
                except Exception as e:
                    print(f"Error loading or parsing reference sequence CSV: {e}")
                    self.full_ref_sequences = None
            else:
                print(f"Warning: Reference sequence CSV not found at {args.ref_sequence_csv_path}")

    def _build_model(self):
        model = Model(
            self.args,
            self.device,
            full_ref_sequences=self.full_ref_sequences,
            dataset_sizes={'train': self.train_set_size,
                          'val': self.val_set_size,
                          'test': self.test_set_size}
        )
        return model

    def _get_data(self, flag, vali_test=False):
        data_set, data_loader = data_provider(self.args, flag)
        
        dataset_size = len(data_set)
        if flag == 'train':
            self.train_set_size = dataset_size
            if hasattr(self.model, 'set_dataset_sizes'):
                self.model.set_dataset_sizes('train', self.train_set_size)
        elif flag == 'val':
            self.val_set_size = dataset_size
            if hasattr(self.model, 'set_dataset_sizes'):
                self.model.set_dataset_sizes('val', self.val_set_size)
        elif flag == 'test':
            self.test_set_size = dataset_size
            if hasattr(self.model, 'set_dataset_sizes'):
                self.model.set_dataset_sizes('test', self.test_set_size)
        
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return self._mse_loss_with_target
    
    def _mse_loss_with_target(self, outputs, batch_y, batch_idx=None, split=None):
        """
        MSE loss function that only calculates loss on the target feature.
        """
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        if outputs is None:
            raise ValueError("Received None for outputs in _mse_loss_with_target")
            
        predicted_steps = outputs.shape[1]
        batch_y_sliced = batch_y[:, -predicted_steps:, :]

        outputs_target = outputs[:, :, -1:]
        batch_y_target = batch_y_sliced[:, :, -1:]

        return F.mse_loss(outputs_target, batch_y_target)
    
    def _select_scheduler(self):
        if self.args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-8)
        else:
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                steps_per_epoch=len(self.train_loader),
                pct_start=self.args.pct_start if hasattr(self.args, 'pct_start') else 0.2,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate
            )
        return scheduler

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        
        # ticker name from data_path
        ticker = None
        if hasattr(self.args, 'data_path') and self.args.data_path:
            ticker = os.path.splitext(os.path.basename(self.args.data_path))[0]
            print(f"Training model for ticker: {ticker}")
        
        # model_id that includes the ticker
        model_id = setting
        if ticker and self.args.single_stock == 0:
            model_id = f"{setting}_{ticker}"
            print(f"Using model ID with ticker: {model_id}")
        
        path = os.path.join(self.args.checkpoints, model_id)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        self.model = self._build_model()
        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()
        self.scheduler = self._select_scheduler()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                self.optimizer.zero_grad()
                
                # move to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # current batch index and split for normalized stats retrieval
                if hasattr(self.model, 'current_batch_idx'):
                    self.model.current_batch_idx = i
                if hasattr(self.model, 'current_split'):
                    self.model.current_split = 'train'
                
                # forward pass
                if hasattr(self.model, 'forecast'):
                    model_output = self.model.forecast(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    outputs = model_output[0] if isinstance(model_output, tuple) else model_output
                else:
                    outputs = self.model(batch_x)
                
                # only use predicted steps for loss calculation
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                
                loss = self.criterion(outputs, batch_y)
                
                # backward and optimize
                loss.backward()
                if self.args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.optimizer.step()
                
                # record loss
                train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
            
            # update learning rate
            if self.args.lradj == 'COS':
                self.scheduler.step()
                print("lr = {:.10f}".format(self.optimizer.param_groups[0]['lr']))
            else:
                adjust_learning_rate(self.optimizer, epoch + 1, self.args)
                
            # validation
            vali_loss = self.vali(vali_data, vali_loader)
            epoch_train_loss = np.mean(train_loss)
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, epoch_train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, path, ticker)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # use the best model for testing
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, weights_only=False))
        
        return self.model

    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # current batch index and split for normalized stats retrieval
                if hasattr(self.model, 'current_batch_idx'):
                    self.model.current_batch_idx = i
                if hasattr(self.model, 'current_split'):
                    self.model.current_split = 'val'
                
                # forward pass
                model_output = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                outputs = model_output[0] if isinstance(model_output, tuple) else model_output
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                
                # Calculate loss
                loss = self.criterion(outputs, batch_y)
                total_loss.append(loss.item())
        
        return np.average(total_loss)

    def test(self, setting, specific_model_path=None):
        test_data, test_loader = self._get_data(flag='test')

        #clear gpu cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # use specific model path or construct default path
        if specific_model_path and os.path.exists(specific_model_path):
            path = specific_model_path
            print(f"Loading model from specified path: {path}")
        else:
            ticker = None
            if hasattr(self.args, 'data_path') and self.args.data_path:
                ticker = os.path.splitext(os.path.basename(self.args.data_path))[0]
            
            # create model_id
            model_id = setting
            if ticker and self.args.single_stock == 0: 
                model_id = f"{setting}_{ticker}"
            
            path = os.path.join(self.args.checkpoints, model_id, 'checkpoint.pth')
            print(f"Looking for model at: {path}")
        
        if not os.path.exists(path):
            print(f"ERROR: Model checkpoint file not found at {path}")
            if hasattr(self.args, 'eval_only') and self.args.eval_only:
                raise FileNotFoundError(f"Model checkpoint not found at {path}")
            else:
                print("Returning zeros for metrics due to missing checkpoint")
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # load pre-trained model
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if any(key.startswith('module.') for key in checkpoint.keys()):
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}  

        if not hasattr(self, 'model'):
            self.model = self._build_model() 

        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # folder for test results
        folder_path = os.path.join(self.folder_path, 'test_results')
        os.makedirs(folder_path, exist_ok=True)
        
        preds = []
        trues = []
        inputs = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if hasattr(self.model, 'current_batch_idx'):
                    self.model.current_batch_idx = i
                if hasattr(self.model, 'current_split'):
                    self.model.current_split = 'test'
                
                # forward pass
                model_output = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                outputs = model_output[0] if isinstance(model_output, tuple) else model_output
                
                # only use predicted steps for evaluation
                batch_y = batch_y[:, -self.args.pred_len:, :]
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                
                preds.append(outputs)
                trues.append(batch_y)
                inputs.append(batch_x)
                
                # generate visualizations
                if self.visualization and i % 5 == 0:
                    sample_idx = 0
                    if sample_idx < len(outputs):
                        target_feature = -1
                        x_hist = batch_x[sample_idx, :, target_feature]
                        y_future = batch_y[sample_idx, :, target_feature]
                        full_gt = np.concatenate([x_hist, y_future])
                        
                        # historical data + predictions
                        y_pred = outputs[sample_idx, :, target_feature]
                        full_pred = np.concatenate([x_hist, y_pred])
                        
                        visual(full_gt, full_pred, os.path.join(self.folder_path, f'{i}.pdf'))
                
                # memory cleanup
                if i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # concatenate batch results
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = np.concatenate(inputs, axis=0)
        
        # only keep the target feature for evaluation
        if self.args.features == 'MS':
            preds = preds[..., -1:] 
            trues = trues[..., -1:]

        # calculate metrics
        mae, mse, rmse, mape, mspe, rse, corr, accuracy, mcc, f1, pred_std, true_std, vol_ratio = metric(preds, trues)

        
        # metrics dictionary
        metrics_dict = {
            'Traditional Metrics': {
                'MAE': float(mae),
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAPE': float(mape),
                'MSPE': float(mspe),
                'RSE': float(rse),
                'CORR': float(corr)
            },
            'Directional Metrics': {
                'Accuracy': float(accuracy),
                'MCC': float(mcc),
                'F1': float(f1)
            },
            'Volatility Metrics': {
                'Pred StdDev': float(pred_std),
                'True StdDev': float(true_std),
                'Volatility Ratio': float(vol_ratio)
            }
        }
        
        # Save metrics, predictions, ground truth
        results_path = os.path.join(self.folder_path, 'results')
        os.makedirs(results_path, exist_ok=True)
        
        with open(os.path.join(results_path, 'metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4, cls=NumpyEncoder)
        
        np.save(os.path.join(results_path, 'metrics.npy'), 
               np.array([mae, mse, rmse, mape, mspe, rse, corr, accuracy, mcc, f1, 
                        pred_std, true_std, vol_ratio]))
        np.save(os.path.join(results_path, 'pred.npy'), preds)
        np.save(os.path.join(results_path, 'true.npy'), trues)
        
        # results
        print("\n=== Test Results ===")
        print("\nTraditional Metrics:")
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}, MSPE: {mspe:.4f}, RSE: {rse:.4f}")
        print(f"CORR: {corr:.4f}")
        
        print("\nDirectional Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nVolatility Metrics:")
        print(f"Pred StdDev: {pred_std:.4f}")
        print(f"True StdDev: {true_std:.4f}")
        print(f"Volatility Ratio: {vol_ratio:.4f}")
        
        print("\n=== Results saved successfully ===")
        
        # delete checkpoint file
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"Checkpoint deleted: {path}")
            else:
                print(f"Checkpoint not found at {path}")
        except Exception as e:
            print(f"Error deleting checkpoint: {e}")
        
        return mae, mse, rmse, accuracy, mcc, f1, pred_std, vol_ratio
