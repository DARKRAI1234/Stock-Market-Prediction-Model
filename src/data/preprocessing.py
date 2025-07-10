"""
Utility functions for computing technical indicators and
robust feature scaling used across datasets.
"""

import numpy as np
import pandas as pd


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators in-place and return the DataFrame."""
    
    # RSI (14)
    if 'RSI' not in df.columns:
        delta = df['Close'].diff().fillna(0)  # Fill NaN from diff immediately
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean().fillna(0)
        avg_loss = loss.rolling(window=14).mean().fillna(0)
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].replace([np.inf, -np.inf], np.nan).fillna(50)
    
    # MACD
    if 'MACD' not in df.columns:
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD'] = df['MACD'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['MACD_Signal'] = df['MACD_Signal'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Bollinger Bands
    if 'UpperBB' not in df.columns:
        sma20 = df['Close'].rolling(window=20).mean().fillna(df['Close'])
        std20 = df['Close'].rolling(window=20).std().fillna(0)
        df['UpperBB'] = sma20 + (std20 * 2)
        df['LowerBB'] = sma20 - (std20 * 2)
        df['BB_Width'] = (df['UpperBB'] - df['LowerBB']) / sma20.replace(0, np.finfo(float).eps)
        df['UpperBB'] = df['UpperBB'].replace([np.inf, -np.inf], np.nan).fillna(df['Close'])
        df['LowerBB'] = df['LowerBB'].replace([np.inf, -np.inf], np.nan).fillna(df['Close'])
        df['BB_Width'] = df['BB_Width'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # ATR (14)
    if 'ATR' not in df.columns:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift().fillna(df['Close']))
        low_close = np.abs(df['Low'] - df['Close'].shift().fillna(df['Close']))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1).fillna(0)
        df['ATR'] = true_range.rolling(14).mean().fillna(0)
    
    # Volume indicators
    if 'Volume_MA' not in df.columns:
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean().fillna(df['Volume'])
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA'].replace(0, np.finfo(float).eps)
        df['Volume_MA'] = df['Volume_MA'].replace([np.inf, -np.inf], np.nan).fillna(df['Volume'])
        df['Volume_Ratio'] = df['Volume_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Moving Averages and relative price
    if 'SMA50' not in df.columns:
        df['SMA50'] = df['Close'].rolling(window=50).mean().fillna(df['Close'])
        df['Price_to_SMA50'] = df['Close'] / df['SMA50'].replace(0, np.finfo(float).eps)
        df['SMA50'] = df['SMA50'].replace([np.inf, -np.inf], np.nan).fillna(df['Close'])
        df['Price_to_SMA50'] = df['Price_to_SMA50'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Rate of Change
    if 'ROC5' not in df.columns:
        df['ROC5'] = df['Close'].pct_change(5).fillna(0)
        df['ROC10'] = df['Close'].pct_change(10).fillna(0)
        df['ROC20'] = df['Close'].pct_change(20).fillna(0)
    
    # Stochastic Oscillator
    if 'Stoch_K' not in df.columns:
        low_14 = df['Low'].rolling(window=14).min().fillna(df['Low'])
        high_14 = df['High'].rolling(window=14).max().fillna(df['High'])
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14).replace(0, np.finfo(float).eps))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean().fillna(50)
        df['Stoch_K'] = df['Stoch_K'].replace([np.inf, -np.inf], np.nan).fillna(50)
        df['Stoch_D'] = df['Stoch_D'].replace([np.inf, -np.inf], np.nan).fillna(50)
    
    # ADX
    if 'ADX' not in df.columns:
        plus_dm = df['High'].diff().fillna(0)
        minus_dm = df['Low'].diff().multiply(-1).fillna(0)
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = pd.DataFrame({
            'tr1': np.abs(df['High'] - df['Low']),
            'tr2': np.abs(df['High'] - df['Close'].shift().fillna(df['Close'])),
            'tr3': np.abs(df['Low'] - df['Close'].shift().fillna(df['Close']))
        }).max(axis=1).fillna(0)
        atr = tr.rolling(window=14).mean().fillna(0)
        plus_di = 100 * (plus_dm.rolling(window=14).mean().fillna(0) / atr.replace(0, np.finfo(float).eps))
        minus_di = 100 * (minus_dm.rolling(window=14).mean().fillna(0) / atr.replace(0, np.finfo(float).eps))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.finfo(float).eps)
        df['ADX'] = dx.rolling(window=14).mean().fillna(25)
    
    # Volatility
    if 'Volatility' not in df.columns:
        df['Volatility'] = (df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)).fillna(0)
    
    # Price Streaks
    if 'Streak' not in df.columns:
        direction = np.sign(df['Close'].diff().fillna(0))
        streak = [0]
        for i in range(1, len(direction)):
            if direction.iloc[i] == direction.iloc[i-1]:
                streak.append(streak[-1] + direction.iloc[i])
            else:
                streak.append(direction.iloc[i])
        df['Streak'] = streak
    
    # Return Volatility
    if 'Return_Volatility_20' not in df.columns:
        df['Return_Volatility_20'] = df['Close'].pct_change().rolling(window=20).std().fillna(0)
    
    # Forward returns
    df['Return_1d'] = df['Close'].pct_change(1).shift(-1).fillna(0)
    df['Return_5d'] = df['Close'].pct_change(5).shift(-5).fillna(0)
    df['Return_10d'] = df['Close'].pct_change(10).shift(-10).fillna(0)
    
    return df


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Min-max scale each column to [0,1] with IQR outlier clipping."""
    
    # Ensure input is a numpy array
    features = np.asarray(features, dtype=np.float32)
    
    # Replace inf with NaN for consistent handling
    features = np.where(np.isinf(features), np.nan, features)
    
    # Normalize each feature column
    for j in range(features.shape[1]):
        col = features[:, j]
        
        # Handle NaN: fill with median, or 0 if entire column is NaN
        if np.all(np.isnan(col)):
            col = np.zeros_like(col)  # If all NaN, set to zero
        else:
            median_val = np.nanmedian(col)
            col = np.where(np.isnan(col), median_val, col)
        
        # Clip outliers using IQR method
        q1, q3 = np.percentile(col, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            # If no variation (IQR=0), avoid clipping to preserve data
            lower_bound = np.min(col)
            upper_bound = np.max(col)
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
        col = np.clip(col, lower_bound, upper_bound)
        
        # Min-max normalization to [0, 1]
        col_min, col_max = np.min(col), np.max(col)
        if col_max > col_min:
            features[:, j] = (col - col_min) / (col_max - col_min)
        else:
            features[:, j] = 0  # If no variation, set to 0
    
    # Final check for any remaining NaN or inf (should be rare)
    features = np.where(np.isinf(features), 0, features)
    features = np.where(np.isnan(features), 0, features)
    
    return features
