import numpy as np
import talib
import torch
import json
import os
from typing import Set, List

"""
Technical Indicators Explanations:

1. Commodity Channel Index (CCI):
   Purpose: Identifies price deviations from a moving average for cyclical trends
   Note: Good for cyclical asset analysis

2. Ichimoku Cloud:
   Purpose: Combines multiple moving averages to identify support, resistance, and trend direction
   Note: Used for identifying trends and potential breakout levels, works well with overbought/oversold thresholds

3. Relative Strength Index (RSI):
   Purpose: Identifies overbought/oversold conditions using momentum
   Note: Works well with overbought/oversold thresholds

4. Stochastic Oscillator:
   Purpose: Measures momentum by comparing current price to range of highs and lows
   Note: Crossovers of %K and %D provide trading signals

5. Williams %R:
   Purpose: Measures overbought/oversold conditions by comparing close to recent highs
   Note: Useful for spotting reversals

6. Rate of Change (ROC):
   Purpose: Measures the percentage change of price over a period
   Note: High ROC indicates strong price changes

7. Momentum Indicator:
   Purpose: Tracks the speed of price changes for trend momentum
   Note: Simple indicator for trend momentum

8. Chande Momentum Oscillator (CMO):
   Purpose: Measures momentum focusing on normalized gain/loss ratios
   Note: Good for shorter time periods

9. TRIX (Triple Exponential Average):
    Purpose: Measures momentum based on rate of change of a triple-smoothed EMA
    Note: Good for filtering noise and identifying long-term trends

10. Simple Moving Average (SMA):
    Purpose: Identifies trend by smoothing price data over a period
    Note: Can use SMA or EMA for smoother trends

11. Moving Average Convergence Divergence (MACD):
    Purpose: Shows momentum via difference between short and long EMAs
    Note: Good for identifying trend momentum shifts

12. Parabolic SAR:
    Purpose: Tracks price trends and reversal using stop-and-reverse points
    Note: Commonly used with trend-following systems. Above 25 suggests strong trend, below 20 is weak

13. ADX (Average Directional Index):
    Purpose: Measures trend strength using directional movement
    Note: Above 25 suggests strong trend, below 20 is weak

14. Exponential Moving Average (EMA):
    Purpose: Tracks the trend by smoothing price data with greater weight on recent prices
    Note: EMA reacts faster to recent price changes compared to SMA

15. Bollinger Bands:
    Purpose: Measures volatility using standard deviations around moving average
    Note: Upper and lower bands help identify breakout potential

16. Average True Range (ATR):
    Purpose: Measures the range of price movements over a period
    Note: Higher ATR indicates higher volatility

17. On-Balance Volume (OBV):
    Purpose: Cumulative volume indicator linked to price movement direction
    Note: Tracks volume to confirm trends

18. Money Flow Index (MFI):
    Purpose: Measures momentum using volume and price to identify extremes
    Note: Combines volume and price to identify trend extremes

19. Chaikin A/D Line:
    Purpose: Tracks the cumulative flow of money into or out of a security
    Note: Combines price and volume effectively
"""

class TechnicalIndicators:
    # Default configuration for indicators
    AVAILABLE_INDICATORS = {
        # Basic indicators
        'rsi': True,      # Relative Strength Index
        'cci': True,      # Commodity Channel Index
        
        # Moving averages
        'sma': True,      # Simple Moving Average
        'ema': True,      # Exponential Moving Average
        'wma': False,      # Weighted Moving Average
        'trix': False,     # Triple Exponential Average
        
        # Momentum indicators
        'roc': False,      # Rate of Change
        'mom': True,      # Momentum
        'cmo': False,      # Chande Momentum Oscillator
        
        # Volatility indicators
        'atr': False,      # Average True Range
        'bbands': True,   # Bollinger Bands
        
        # Volume indicators
        'obv': False,      # On Balance Volume
        'mfi': False,      # Money Flow Index
        'ad': False,       # Accumulation/Distribution Line
        
        # Trend indicators
        'adx': True,      # Average Directional Index
        'macd': True,     # MACD
        
        # Additional indicators
        'stoch': True,    # Stochastic
        'willr': True,    # Williams %R
        'sar': False       # Parabolic SAR
    }

    # Default timeperiods for indicators (standard industry values)
    DEFAULT_TIMEPERIODS = {
        'sma': 14,
        'ema': 14,
        'wma': 14,
        'trix': 14,
        'roc': 10,
        'mom': 10,
        'cmo': 14,
        'rsi': 14,
        'cci': 14,
        'atr': 14,
        'bbands': 20,
        'mfi': 14,
        'adx': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'stoch_k': 14,
        'stoch_d': 3,
        'willr': 14
    }
    
    # MACD standard ratios (to maintain proper relationships when scaling)
    MACD_RATIOS = {
        'fast_to_slow': 12/26,  # ~0.46
        'signal_to_slow': 9/26,  # ~0.35
    }

    def __init__(self, data=None, seq_len=None, pred_len=None, indicators_config=None, 
                 open_prices=None, high_prices=None, low_prices=None, 
                 close_prices=None, volume=None, debug=False, fill_nan=False,
                 timeperiods=None):
        """
        Initialize the TechnicalIndicators class with price data.
        
        Args:
            data: Legacy parameter for backward compatibility (single price input)
            seq_len: Sequence length for the data (will be used as timeperiod with caps)
            pred_len: Prediction length
            indicators_config: Path to a JSON file with indicator configuration
            open_prices: Open prices tensor/array of shape (batch_size, sequence_length)
            high_prices: High prices tensor/array of shape (batch_size, sequence_length)
            low_prices: Low prices tensor/array of shape (batch_size, sequence_length)
            close_prices: Close prices tensor/array of shape (batch_size, sequence_length)
            volume: Volume data tensor/array of shape (batch_size, sequence_length)
            debug: Debug mode flag (not used in this implementation)
            fill_nan: Fill NaN values flag (not used in this implementation)
            timeperiods: Dictionary of custom timeperiods for indicators (overrides defaults)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.selected_indicators = None
        
        # Handle legacy data input for backward compatibility
        self.data = data
        
        # Store the individual price components
        self.open_prices = open_prices
        self.high_prices = high_prices
        self.low_prices = low_prices
        self.close_prices = close_prices
        self.volume = volume
        
        # If we have the new format data but not the old format, use close prices as legacy data
        if self.data is None and self.close_prices is not None:
            self.data = self.close_prices
        
        # Initialize enabled_indicators with default values from AVAILABLE_INDICATORS
        self.enabled_indicators = self.AVAILABLE_INDICATORS.copy()
        
        # Initialize timeperiods based on sequence length with caps at default values
        self.timeperiods = self._calculate_timeperiods(seq_len)
        
        # Update timeperiods with custom values if provided
        if timeperiods is not None:
            self.timeperiods.update(timeperiods)
        
        # Load selected indicators from config if provided
        if indicators_config and os.path.exists(indicators_config):
            try:
                with open(indicators_config, 'r') as f:
                    config = json.load(f)
                    self.selected_indicators = config.get('selected_indicators', None)
                    print(f"Loaded RL-selected indicators: {self.selected_indicators}")
            except Exception as e:
                print(f"Error loading indicators config: {e}")
                self.selected_indicators = None
        
        # Initialize all indicators if we have data
        if self.data is not None or self.close_prices is not None:
            self.indicators = self.calculate_indicators()
    
    def _calculate_timeperiods(self, seq_len):
        """
        Calculate timeperiods based on sequence length with caps at default values.
        For MACD, maintain the standard ratios between fast, slow, and signal periods.
        
        Args:
            seq_len: Sequence length to use as base timeperiod
            
        Returns:
            dict: Dictionary of timeperiods
        """
        if seq_len is None:
            return self.DEFAULT_TIMEPERIODS.copy()
        
        timeperiods = {}
        
        # For each indicator, use min(seq_len, default_timeperiod)
        for key, default_value in self.DEFAULT_TIMEPERIODS.items():
            # Skip MACD parameters (will handle separately)
            if key.startswith('macd_'):
                continue
            
            # Use sequence length but cap at default value
            timeperiods[key] = min(seq_len - 5, default_value)
        
        # Handle MACD parameters to maintain proper ratios
        # If seq_len is less than the default slow period (26), scale all periods proportionally
        if seq_len < self.DEFAULT_TIMEPERIODS['macd_slow']:
            # Use seq_len as the slow period
            macd_slow = seq_len - 5
            # Calculate fast and signal periods based on standard ratios
            macd_fast = max(2, int(round(macd_slow * self.MACD_RATIOS['fast_to_slow'])))
            macd_signal = max(2, int(round(macd_slow * self.MACD_RATIOS['signal_to_slow'])))
        else:
            # Use default values if seq_len is large enough
            macd_slow = self.DEFAULT_TIMEPERIODS['macd_slow']
            macd_fast = self.DEFAULT_TIMEPERIODS['macd_fast']
            macd_signal = self.DEFAULT_TIMEPERIODS['macd_signal']
        
        timeperiods['macd_slow'] = macd_slow
        timeperiods['macd_fast'] = macd_fast
        timeperiods['macd_signal'] = macd_signal
        
        return timeperiods
    
    def get_enabled_indicators(self) -> Set[str]:
        """Returns set of currently enabled indicators."""
        return {k for k, v in self.enabled_indicators.items() if v}
    
    def enable_indicator(self, indicator: str):
        """Enable a specific indicator."""
        if indicator in self.enabled_indicators:
            self.enabled_indicators[indicator] = True
    
    def disable_indicator(self, indicator: str):
        """Disable a specific indicator."""
        if indicator in self.enabled_indicators:
            self.enabled_indicators[indicator] = False
    
    def calculate_indicators(self, data=None, open_prices=None, high_prices=None, 
                            low_prices=None, close_prices=None, volume=None,
                            timeperiods=None):
        """
        Calculate various technical indicators using the appropriate price data.
        Only returns the last value from each indicator array.
        
        Args:
            data: Legacy parameter for backward compatibility (single price input)
            open_prices: Open prices tensor/array
            high_prices: High prices tensor/array
            low_prices: Low prices tensor/array
            close_prices: Close prices tensor/array
            volume: Volume data tensor/array
            timeperiods: Dictionary of custom timeperiods for indicators
            
        Returns:
            dict: Dictionary containing calculated indicators (last values only)
        """
        # Use provided data or fall back to stored data
        open_data = open_prices if open_prices is not None else self.open_prices
        high_data = high_prices if high_prices is not None else self.high_prices
        low_data = low_prices if low_prices is not None else self.low_prices
        close_data = close_prices if close_prices is not None else self.close_prices
        volume_data = volume if volume is not None else self.volume
        
        # Use provided timeperiods or fall back to stored timeperiods
        tp = self.timeperiods.copy()
        if timeperiods is not None:
            tp.update(timeperiods)
        
        # For backward compatibility
        if data is not None:
            if isinstance(data, torch.Tensor):
                data_np = data.cpu().numpy()
            else:
                data_np = data
        else:
            data_np = None
        
        # Convert all input data to numpy arrays if they're torch tensors
        if open_data is not None and isinstance(open_data, torch.Tensor):
            open_data = open_data.cpu().numpy()
        if high_data is not None and isinstance(high_data, torch.Tensor):
            high_data = high_data.cpu().numpy()
        if low_data is not None and isinstance(low_data, torch.Tensor):
            low_data = low_data.cpu().numpy()
        if close_data is not None and isinstance(close_data, torch.Tensor):
            close_data = close_data.cpu().numpy()
        if volume_data is not None and isinstance(volume_data, torch.Tensor):
            volume_data = volume_data.cpu().numpy()
        
        # Use close_data as data_np for backward compatibility if available
        if data_np is None and close_data is not None:
            data_np = close_data
            
        # If we still don't have data, return empty dict
        if data_np is None:
            return {}
            
        indicators = {}
        
        # Determine number of batches from first available data
        num_batches = data_np.shape[0] if data_np is not None else 0
        if open_data is not None: num_batches = open_data.shape[0]
        if high_data is not None: num_batches = high_data.shape[0]
        if low_data is not None: num_batches = low_data.shape[0]
        if close_data is not None: num_batches = close_data.shape[0]
        
        for i in range(num_batches):
            # Get price data for this batch
            # Default to using data_np if specific price type isn't available
            # We keep all variables even if some are unused in certain indicators
            # to maintain consistent batch processing
            _ = open_data[i] if open_data is not None else data_np[i]
            high_seq = high_data[i] if high_data is not None else data_np[i]
            low_seq = low_data[i] if low_data is not None else data_np[i]
            close_seq = close_data[i] if close_data is not None else data_np[i]
            vol_seq = volume_data[i] if volume_data is not None else None
            
            # Ensure all sequences are 1D arrays as required by TA-Lib
            if high_seq is not None and len(high_seq.shape) > 1:
                high_seq = high_seq.flatten()
            if low_seq is not None and len(low_seq.shape) > 1:
                low_seq = low_seq.flatten()
            if close_seq is not None and len(close_seq.shape) > 1:
                close_seq = close_seq.flatten()
            if vol_seq is not None and len(vol_seq.shape) > 1:
                vol_seq = vol_seq.flatten()
            
            # Convert sequences to np.float64 for TA-Lib compatibility
            if high_seq is not None:
                high_seq = high_seq.astype(np.float64)
            if low_seq is not None:
                low_seq = low_seq.astype(np.float64)
            if close_seq is not None:
                close_seq = close_seq.astype(np.float64)
            if vol_seq is not None:
                vol_seq = vol_seq.astype(np.float64)
            
            # RSI - Uses close prices
            if self.enabled_indicators['rsi']:
                rsi = talib.RSI(close_seq, timeperiod=tp['rsi'])
                indicators[f'rsi_{i}'] = rsi[~np.isnan(rsi)]
            
            # CCI - Uses high, low, close prices
            if self.enabled_indicators['cci']:
                cci = talib.CCI(high_seq, low_seq, close_seq, timeperiod=tp['cci'])
                indicators[f'cci_{i}'] = cci[~np.isnan(cci)]
            
            # Moving averages - Use close prices
            if self.enabled_indicators['sma']:
                sma = talib.SMA(close_seq, timeperiod=tp['sma'])
                indicators[f'sma_{i}'] = sma[~np.isnan(sma)]
            
            if self.enabled_indicators['ema']:
                ema = talib.EMA(close_seq, timeperiod=tp['ema'])
                indicators[f'ema_{i}'] = ema[~np.isnan(ema)]
            
            if self.enabled_indicators['wma']:
                wma = talib.WMA(close_seq, timeperiod=tp['wma'])
                indicators[f'wma_{i}'] = wma[~np.isnan(wma)]
            
            if self.enabled_indicators['trix']:
                trix = talib.TRIX(close_seq, timeperiod=tp['trix'])
                indicators[f'trix_{i}'] = trix[~np.isnan(trix)]
            
            # Momentum indicators - Use close prices
            if self.enabled_indicators['roc']:
                roc = talib.ROC(close_seq, timeperiod=tp['roc'])
                indicators[f'roc_{i}'] = roc[~np.isnan(roc)]
            
            if self.enabled_indicators['mom']:
                mom = talib.MOM(close_seq, timeperiod=tp['mom'])
                indicators[f'mom_{i}'] = mom[~np.isnan(mom)]
            
            if self.enabled_indicators['cmo']:
                cmo = talib.CMO(close_seq, timeperiod=tp['cmo'])
                indicators[f'cmo_{i}'] = cmo[~np.isnan(cmo)]
            
            # Volatility indicators
            if self.enabled_indicators['atr']:
                atr = talib.ATR(high_seq, low_seq, close_seq, timeperiod=tp['atr'])
                indicators[f'atr_{i}'] = atr[~np.isnan(atr)]
            
            if self.enabled_indicators['bbands']:
                upper, middle, lower = talib.BBANDS(close_seq, timeperiod=tp['bbands'])
                indicators[f'bbands_{i}'] = (upper[~np.isnan(upper)], middle[~np.isnan(middle)], lower[~np.isnan(lower)])
            
            # Volume indicators
            if vol_seq is not None:
                if self.enabled_indicators['obv']:
                    obv = talib.OBV(close_seq, vol_seq)
                    indicators[f'obv_{i}'] = obv[~np.isnan(obv)]
                
                if self.enabled_indicators['mfi']:
                    mfi = talib.MFI(high_seq, low_seq, close_seq, vol_seq, timeperiod=tp['mfi'])
                    indicators[f'mfi_{i}'] = mfi[~np.isnan(mfi)]
                
                if self.enabled_indicators['ad']:
                    ad = talib.AD(high_seq, low_seq, close_seq, vol_seq)
                    indicators[f'ad_{i}'] = ad[~np.isnan(ad)]
            
            # Trend indicators
            if self.enabled_indicators['adx']:
                adx = talib.ADX(high_seq, low_seq, close_seq, timeperiod=tp['adx'])
                indicators[f'adx_{i}'] = adx[~np.isnan(adx)]
            
            if self.enabled_indicators['macd']:
                macd, signal, hist = talib.MACD(close_seq, 
                                              fastperiod=tp['macd_fast'],
                                              slowperiod=tp['macd_slow'],
                                              signalperiod=tp['macd_signal'])
                indicators[f'macd_{i}'] = (macd[~np.isnan(macd)], signal[~np.isnan(signal)], hist[~np.isnan(hist)])
            
            # Additional indicators
            if self.enabled_indicators['stoch']:
                slowk, slowd = talib.STOCH(high_seq, low_seq, close_seq)
                indicators[f'stoch_{i}'] = (slowk[~np.isnan(slowk)], slowd[~np.isnan(slowd)])
            
            if self.enabled_indicators['willr']:
                willr = talib.WILLR(high_seq, low_seq, close_seq, timeperiod=tp['willr'])
                indicators[f'willr_{i}'] = willr[~np.isnan(willr)]
            
            if self.enabled_indicators['sar']:
                sar = talib.SAR(high_seq, low_seq)
                indicators[f'sar_{i}'] = sar[~np.isnan(sar)]
            
        return indicators

    def format_indicator_prompt(self, indicators, batch_idx):
        """Format technical indicators into a natural language prompt string.
        
        Args:
            indicators (dict): Dictionary of calculated indicators
            batch_idx (int): Batch index to get indicators for
            
        Returns:
            str: Formatted prompt string with technical indicators
        """
        if not indicators:
            return ""
        
        # Store original enabled indicators if we need to modify them
        original_enabled = None
        use_selected = hasattr(self, 'selected_indicators') and self.selected_indicators
        
        if use_selected:
            # Temporarily store the original enabled indicators
            original_enabled = self.enabled_indicators.copy()
            
            # Create a temporary enabled_indicators dict with only selected indicators enabled
            temp_enabled = {ind: (ind in self.selected_indicators) for ind in self.AVAILABLE_INDICATORS}
            self.enabled_indicators = temp_enabled
        
        def format_array(arr):
            """Helper function to format numpy arrays"""
            if arr is None:
                return "N/A"
                
            # Format each number to a consistent total width
            def format_number(x):
                return f"{x:.4g}"  # 4 significant digits total
                
            # Custom formatter with controlled width
            formatter = {'float_kind': format_number}
            with np.printoptions(formatter=formatter, threshold=np.inf, suppress=True):
                return np.array2string(arr, separator=', ')

        # Check if we're dealing with the new indicator format (with batch suffixes)
        if any(f'{key}_{batch_idx}' in indicators for key in self.enabled_indicators):
            prompt = "\n"

            # Handle RSI
            if self.enabled_indicators['rsi'] and f'rsi_{batch_idx}' in indicators:
                rsi = indicators[f'rsi_{batch_idx}']
                prompt += f"- Relative Strength Index: {format_array(rsi)}\n"
            
            # Handle CCI
            if self.enabled_indicators['cci'] and f'cci_{batch_idx}' in indicators:
                cci = indicators[f'cci_{batch_idx}']
                prompt += f"- Commodity Channel Index: {format_array(cci)}\n"
            
            # Handle SMA
            if self.enabled_indicators['sma'] and f'sma_{batch_idx}' in indicators:
                sma = indicators[f'sma_{batch_idx}']
                prompt += f"- Simple Moving Average: {format_array(sma)}\n"
            
            # Handle EMA
            if self.enabled_indicators['ema'] and f'ema_{batch_idx}' in indicators:
                ema = indicators[f'ema_{batch_idx}']
                prompt += f"- Exponential Moving Average: {format_array(ema)}\n"
            
            # Handle WMA
            if self.enabled_indicators['wma'] and f'wma_{batch_idx}' in indicators:
                wma = indicators[f'wma_{batch_idx}']
                prompt += f"- Weighted Moving Average: {format_array(wma)}\n"
            
            # Handle ROC
            if self.enabled_indicators['roc'] and f'roc_{batch_idx}' in indicators:
                roc = indicators[f'roc_{batch_idx}']
                prompt += f"- Rate of Change: {format_array(roc)}\n"
            
            # Handle MOM
            if self.enabled_indicators['mom'] and f'mom_{batch_idx}' in indicators:
                mom = indicators[f'mom_{batch_idx}']
                prompt += f"- Momentum: {format_array(mom)}\n"
            
            # Handle CMO
            if self.enabled_indicators['cmo'] and f'cmo_{batch_idx}' in indicators:
                cmo = indicators[f'cmo_{batch_idx}']
                prompt += f"- Chande Momentum Oscillator: {format_array(cmo)}\n"
            
            # Handle ATR
            if self.enabled_indicators['atr'] and f'atr_{batch_idx}' in indicators:
                atr = indicators[f'atr_{batch_idx}']
                prompt += f"- Average True Range: {format_array(atr)}\n"
            
            # Handle Bollinger Bands with labels
            if self.enabled_indicators['bbands'] and f'bbands_{batch_idx}' in indicators:
                upper, middle, lower = indicators[f'bbands_{batch_idx}']
                prompt += f"- Bollinger Bands Upper: {format_array(upper)}\n"
                prompt += f"- Bollinger Bands Middle: {format_array(middle)}\n"
                prompt += f"- Bollinger Bands Lower: {format_array(lower)}\n"
            
            # Handle OBV
            if self.enabled_indicators['obv'] and f'obv_{batch_idx}' in indicators:
                obv = indicators[f'obv_{batch_idx}']
                prompt += f"- On-Balance Volume: {format_array(obv)}\n"
            
            # Handle MFI
            if self.enabled_indicators['mfi'] and f'mfi_{batch_idx}' in indicators:
                mfi = indicators[f'mfi_{batch_idx}']
                prompt += f"- Money Flow Index: {format_array(mfi)}\n"
            
            # Handle AD
            if self.enabled_indicators['ad'] and f'ad_{batch_idx}' in indicators:
                ad = indicators[f'ad_{batch_idx}']
                prompt += f"- Accumulation/Distribution Line: {format_array(ad)}\n"
            
            # Handle ADX
            if self.enabled_indicators['adx'] and f'adx_{batch_idx}' in indicators:
                adx = indicators[f'adx_{batch_idx}']
                prompt += f"- Average Directional Index: {format_array(adx)}\n"
            
            # Handle MACD with labels
            if self.enabled_indicators['macd'] and f'macd_{batch_idx}' in indicators:
                macd, signal, hist = indicators[f'macd_{batch_idx}']
                prompt += f"- MACD Line: {format_array(macd)}\n"
                prompt += f"- MACD Signal: {format_array(signal)}\n"
                prompt += f"- MACD Histogram: {format_array(hist)}\n"
            
            # Handle Stochastic with labels
            if self.enabled_indicators['stoch'] and f'stoch_{batch_idx}' in indicators:
                slowk, slowd = indicators[f'stoch_{batch_idx}']
                prompt += f"- Stochastic %K: {format_array(slowk)}\n"
                prompt += f"- Stochastic %D: {format_array(slowd)}\n"
            
            # Handle Williams %R
            if self.enabled_indicators['willr'] and f'willr_{batch_idx}' in indicators:
                willr = indicators[f'willr_{batch_idx}']
                prompt += f"- Williams %R: {format_array(willr)}\n"
            
            # Handle Parabolic SAR
            if self.enabled_indicators['sar'] and f'sar_{batch_idx}' in indicators:
                sar = indicators[f'sar_{batch_idx}']
                prompt += f"- Parabolic SAR: {format_array(sar)}\n"
            
        # Restore original enabled indicators if they were modified
        if use_selected and original_enabled is not None:
            self.enabled_indicators = original_enabled
            
        return prompt.rstrip("\n") + "\n"

    def set_selected_indicators(self, indicator_names):
        """
        Set which indicators are selected and disable all others.
        
        Args:
            indicator_names: List of indicator names to enable. All others will be disabled.
        """
        self.selected_indicators = indicator_names
        # Update enabled_indicators to match selected ones
        for indicator in self.AVAILABLE_INDICATORS:
            self.enabled_indicators[indicator] = indicator in indicator_names

    def save_selected_indicators(self, filepath):
        """Save the selected indicators to a JSON file."""
        if self.selected_indicators:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump({'selected_indicators': self.selected_indicators}, f)
            print(f"Saved selected indicators to {filepath}")
        else:
            print("No selected indicators to save")

    def get_available_indicators(self) -> List[str]:
        """Get list of all available indicators."""
        return list(self.AVAILABLE_INDICATORS.keys())

    def get_indicator_values(self, indicator_name, batch_idx=None):
        """Get the values for a specific indicator"""
        if batch_idx is not None:
            return self.indicators.get(f"{indicator_name}_{batch_idx}", None)
        else:
            return {k: v for k, v in self.indicators.items() if k.startswith(f"{indicator_name}_")}


