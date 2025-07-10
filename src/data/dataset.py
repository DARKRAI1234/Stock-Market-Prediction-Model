import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import random

from .preprocessing import compute_technical_indicators, normalize_features


class EnhancedStockDataset(Dataset):
    """
    Builds fixed-length windows across multiple stocks for
    training or validation.
    """

    def __init__(
        self,
        input_dir: str,
        valid_symbols: List[str],
        sector_map: Dict[str, int],
        window_size: int = 20,
        prediction_horizon: int = 1,
        split: str = "train",
        augment: bool = False,
    ):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.valid_symbols = valid_symbols if isinstance(valid_symbols, list) else [s.strip().upper() for s in valid_symbols]
        self.stock_data = {}
        self.sector_map = sector_map
        self.split = split
        self.augment = augment
        
        # Define the original features
        self.original_features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal',
            'UpperBB', 'LowerBB', 'BB_Width', 'ATR', 'Volume_MA', 'Volume_Ratio',
            'SMA50', 'Price_to_SMA50', 'ROC5', 'ROC10', 'ROC20', 'Stoch_K', 'Stoch_D',
            'ADX', 'Volatility', 'Streak', 'Return_Volatility_20'
        ]
        
        # Define the important features for sector averages
        self.important_features = ['Close', 'Volume', 'RSI', 'MACD', 'ATR', 'Volatility']
        
        self._load_data(input_dir)
    
    def _load_data(self, input_dir):
        print(f"Loading {self.split} data for {len(self.valid_symbols)} stocks...")
        
        for symbol in tqdm(self.valid_symbols, desc=f"Loading {self.split} stocks"):
            try:
                filepath = os.path.join(input_dir, f"{symbol}.csv")
                if not os.path.exists(filepath):
                    filepath = os.path.join(input_dir, f"{symbol}_processed.csv")
                    if not os.path.exists(filepath):
                        continue
                
                df = pd.read_csv(filepath)
                if 'Date' in df.columns:
                    df = df.set_index('Date', inplace=False)
                
                # Convert columns to numeric and handle invalid entries
                for col in df.columns:
                    df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
                
                # Replace inf with NaN early
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Interpolate for time-series continuity
                df = df.interpolate(method='linear', limit_direction='both')
                
                # Fill remaining NaN with column median
                for col in df.columns:
                    if df[col].isna().any():
                        median_val = df[col].median() if not df[col].isna().all() else 0
                        df[col] = df[col].fillna(median_val)
                
                if len(df) < self.window_size + self.prediction_horizon:
                    continue
                
                # Compute technical indicators
                df = compute_technical_indicators(df)
                
                # Split data for train/valid
                if self.split != 'test':
                    valid_start = int(len(df) * 0.8)
                    if self.split == 'train':
                        df = df.iloc[:valid_start]
                    else:  # valid
                        df = df.iloc[valid_start:]
                
                if len(df) < self.window_size + self.prediction_horizon:
                    continue
                
                # Get features and future returns
                features = df[self.original_features].values.astype(np.float32)
                future_returns = df[f'Return_{self.prediction_horizon}d'].values.astype(np.float32)
                
                if str(symbol) not in self.sector_map:
                    continue
                
                self.stock_data[symbol] = {
                    'features': normalize_features(features),
                    'returns': future_returns,
                    'sector': self.sector_map[str(symbol)],
                }
            except Exception as e:
                print(f"⛔ Failed to load {self.split} data for {symbol}: {str(e)}")
                continue
        
        print(f"Successfully loaded {self.split} data for {len(self.stock_data)} stocks")
        self._compute_sector_averages()
        self._create_windows()
        
    def _compute_sector_averages(self):
        sector_features = defaultdict(list)
        important_indices = [self.original_features.index(feat) for feat in self.important_features]
        
        # Collect important features for each stock
        for symbol, data in self.stock_data.items():
            sector = data['sector']
            important_feats = data['features'][:, important_indices]
            sector_features[sector].append(important_feats)
        
        # Compute and apply sector averages
        for sector, features_list in sector_features.items():
            # Stack all features into a single array and average across all days and stocks
            all_features = np.vstack(features_list)  # Shape: (total_days_across_stocks, 6)
            sector_avg = np.mean(all_features, axis=0)  # Shape: (6,)
            for symbol in [s for s, d in self.stock_data.items() if d['sector'] == sector]:
                orig_features = self.stock_data[symbol]['features']  # Shape: (num_days, 25)
                num_days = orig_features.shape[0]
                # Tile the sector average to match the stock's number of days
                expanded_sector_avg = np.tile(sector_avg, (num_days, 1))  # Shape: (num_days, 6)
                self.stock_data[symbol]['features'] = np.concatenate(
                    [orig_features, expanded_sector_avg], axis=1  # Shape: (num_days, 31)
                )
    
    def _create_windows(self):
        self.windows = []
        dates = {}
        max_len = max(len(data['features']) for data in self.stock_data.values())
        
        for t in range(max_len - self.window_size - self.prediction_horizon + 1):
            dates[t] = {
                'features': [],
                'returns': [],
                'sectors': [],
                'symbols': []
            }
            
            for symbol, data in self.stock_data.items():
                if t + self.window_size + self.prediction_horizon <= len(data['features']):
                    features = data['features'][t:t+self.window_size]
                    future_return = data['returns'][t+self.window_size-1]
                    
                    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                        print(f"Skipping window for {symbol} at t={t} due to invalid features")
                        continue
                    if np.isnan(future_return) or np.isinf(future_return):
                        print(f"Skipping window for {symbol} at t={t} due to invalid return")
                        continue
                    
                    dates[t]['features'].append(features)
                    dates[t]['returns'].append(future_return)
                    dates[t]['sectors'].append(data['sector'])
                    dates[t]['symbols'].append(symbol)
        
        for t, date_data in dates.items():
            if len(date_data['features']) >= 5:
                self.windows.append({
                    'features': torch.tensor(np.array(date_data['features']), dtype=torch.float32),
                    'returns': torch.tensor(np.array(date_data['returns']), dtype=torch.float32),
                    'sectors': torch.tensor(np.array(date_data['sectors']), dtype=torch.long),
                    'symbols': date_data['symbols']
                })
        
        print(f"Created {len(self.windows)} {self.split} windows")
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        if self.augment and random.random() < 0.5:
            # Data augmentation: randomly mask 10-20% of feature values
            mask_rate = random.uniform(0.1, 0.2)
            features = window['features'].clone()
            mask = torch.rand_like(features) < mask_rate
            features[mask] = 0
            
            return {
                'features': features,
                'returns': window['returns'],
                'sectors': window['sectors'],
                'symbols': window['symbols']
            }
        
        return window


class TestStockDataset(Dataset):
    """
    Dataset for predicting all days with padding and masking.
    """
    
    def __init__(self, input_dir, valid_symbols, sector_map, window_size=20, prediction_horizon=1, min_history=1):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.min_history = min_history
        self.valid_symbols = valid_symbols if isinstance(valid_symbols, list) else [s.strip().upper() for s in valid_symbols]
        self.stock_data = {}
        self.sector_map = sector_map
        self.test_windows = []
        
        # Define the original 25 features
        self.original_features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal',
            'UpperBB', 'LowerBB', 'BB_Width', 'ATR', 'Volume_MA', 'Volume_Ratio',
            'SMA50', 'Price_to_SMA50', 'ROC5', 'ROC10', 'ROC20', 'Stoch_K', 'Stoch_D',
            'ADX', 'Volatility', 'Streak', 'Return_Volatility_20'
        ]
        
        # Define important features for sector averages
        self.important_features = ['Close', 'Volume', 'RSI', 'MACD', 'ATR', 'Volatility']
        
        self._load_test_data(input_dir)
    
    def _load_test_data(self, input_dir):
        print(f"Loading test data for {len(self.valid_symbols)} stocks...")
        
        for symbol in tqdm(self.valid_symbols, desc="Loading test stocks"):
            try:
                filepath = os.path.join(input_dir, f"{symbol}.csv")
                if not os.path.exists(filepath):
                    filepath = os.path.join(input_dir, f"{symbol}_processed.csv")
                    if not os.path.exists(filepath):
                        continue
                
                # Load data with Date as index
                df = pd.read_csv(filepath, parse_dates=['Date'] if 'Date' in pd.read_csv(filepath, nrows=1).columns else False)
                if 'Date' in df.columns:
                    df = df.set_index('Date', inplace=False)
                
                # Convert to numeric
                for col in df.columns:
                    df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.ffill().bfill()
                
                if len(df) < self.min_history + self.prediction_horizon:
                    continue
                
                # Compute technical indicators and returns
                df = compute_technical_indicators(df)
                features = df[self.original_features].values.astype(np.float32)
                returns = df[f'Return_{self.prediction_horizon}d'].values.astype(np.float32)
                
                if str(symbol) not in self.sector_map:
                    continue
                
                self.stock_data[symbol] = {
                    'features': normalize_features(features),
                    'prices': df['Close'].values.astype(np.float32),
                    'returns': returns,  # Actual returns for evaluation
                    'sector': self.sector_map[str(symbol)],
                    'dates': df.index.tolist()  # Store dates for day-wise mapping
                }
            except Exception as e:
                print(f"⛔ Failed to load test data for {symbol}: {str(e)}")
                continue
        
        print(f"Successfully loaded test data for {len(self.stock_data)} stocks")
        self._compute_sector_averages()
        self._create_test_windows()
    
    def _compute_sector_averages(self):
        sector_features = defaultdict(list)
        important_indices = [self.original_features.index(feat) for feat in self.important_features]
        
        for symbol, data in self.stock_data.items():
            sector = data['sector']
            important_feats = data['features'][:, important_indices]
            sector_features[sector].append(important_feats)
        
        for sector, features_list in sector_features.items():
            all_features = np.vstack(features_list)
            sector_avg = np.mean(all_features, axis=0)
            for symbol in [s for s, d in self.stock_data.items() if d['sector'] == sector]:
                orig_features = self.stock_data[symbol]['features']
                num_days = orig_features.shape[0]
                expanded_sector_avg = np.tile(sector_avg, (num_days, 1))
                self.stock_data[symbol]['features'] = np.concatenate(
                    [orig_features, expanded_sector_avg], axis=1
                )
    
    def _create_test_windows(self):
        self.test_windows = []
        
        if not self.stock_data:
            print("No stock data available to create windows")
            return
        
        ref_symbol = list(self.stock_data.keys())[0]
        dates = self.stock_data[ref_symbol]['dates']
        max_len = len(dates)
        
        # Create a window for every day from 0 to max_len - 1
        for t in range(max_len):
            window_date = dates[t]
            window_data = {
                'date': window_date,
                'features': [],
                'masks': [],
                'sectors': [],
                'symbols': [],
                'actual_returns': []
            }
            
            for symbol, data in self.stock_data.items():
                if t < len(data['features']):
                    # Calculate available history
                    history_start = max(0, t - self.window_size + 1)
                    history_days = t - history_start + 1
                    
                    # Only include if minimum history is met
                    if history_days >= self.min_history:
                        features = data['features'][history_start:t+1]
                        if history_days < self.window_size:
                            pad_length = self.window_size - history_days
                            pad = np.zeros((pad_length, features.shape[1]))
                            features = np.vstack([pad, features])
                            mask = np.ones((self.window_size,))
                            mask[:pad_length] = 0
                        else:
                            mask = np.ones((self.window_size,))
                        
                        # Actual return (NaN if future data incomplete)
                        actual_return = data['returns'][t] if t + self.prediction_horizon <= len(data['returns']) else np.nan
                        
                        window_data['features'].append(features)
                        window_data['masks'].append(mask)
                        window_data['sectors'].append(data['sector'])
                        window_data['symbols'].append(symbol)
                        window_data['actual_returns'].append(actual_return)
            
            if len(window_data['features']) >= 5:  # Minimum stocks per window
                self.test_windows.append({
                    'date': window_data['date'],
                    'features': torch.tensor(np.array(window_data['features']), dtype=torch.float32),
                    'masks': torch.tensor(np.array(window_data['masks']), dtype=torch.float32),
                    'sectors': torch.tensor(np.array(window_data['sectors']), dtype=torch.long),
                    'symbols': window_data['symbols'],
                    'actual_returns': torch.tensor(np.array(window_data['actual_returns']), dtype=torch.float32)
                })
        
        print(f"Created {len(self.test_windows)} test windows")
    
    def __len__(self):
        return len(self.test_windows)
    
    def __getitem__(self, idx):
        window_data = self.test_windows[idx]
        return {
            'date': window_data['date'],
            'features': window_data['features'],
            'masks': window_data['masks'],
            'sectors': window_data['sectors'],
            'symbols': window_data['symbols'],
            'actual_returns': window_data['actual_returns']
        }
