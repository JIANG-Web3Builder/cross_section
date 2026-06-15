"""
Factor Calculation Script for TFT Model
Calculates all factors from 因子计算.md with Rolling Z-Score normalization (90-day window)
to avoid look-ahead bias. Output: parquet files for BTC, ETH, SOL, BNB (2021-01 to 2026-01)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============== Configuration ==============
DATA_DIR = Path(r"D:\strategy\TFT\factor")
COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
ZSCORE_WINDOW = 90 * 24  # 90 days in hours
START_DATE = "2021-01-01"
END_DATE = "2026-02-01"

# ============== Helper Functions ==============
def rolling_zscore(series, window):
    """Apply rolling z-score normalization to avoid look-ahead bias"""
    rolling_mean = series.rolling(window=window, min_periods=max(1, window//10)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(1, window//10)).std()
    return (series - rolling_mean) / (rolling_std + 1e-8)

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def rsi(series, period=14):
    """Calculate RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def bollinger_bands(close, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    ma = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower

def macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def mfi(high, low, close, volume, period=14):
    """Money Flow Index"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0.0)
    negative_flow = money_flow.where(delta < 0, 0.0)
    positive_mf = positive_flow.rolling(window=period, min_periods=1).sum()
    negative_mf = negative_flow.rolling(window=period, min_periods=1).sum()
    mfi_ratio = positive_mf / (negative_mf + 1e-8)
    return 100 - (100 / (1 + mfi_ratio))

def obv(close, volume):
    """On-Balance Volume"""
    sign = np.sign(close.diff())
    sign.iloc[0] = 0
    return (sign * volume).cumsum()

def trix(close, period=15):
    """Triple Exponential Average"""
    ema1 = ema(close, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return ((ema3 - ema3.shift(1)) / (ema3.shift(1) + 1e-8)) * 100

def stochastic(high, low, close, k_period=14, d_period=3):
    """KDJ / Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-8) * 100
    k = rsv.rolling(window=d_period, min_periods=1).mean()
    d = k.rolling(window=d_period, min_periods=1).mean()
    j = 3 * k - 2 * d
    return k, d, j

def choppiness_index(high, low, close, period=14):
    """Choppiness Index - measures market trendiness vs choppiness"""
    atr_val = atr(high, low, close, 1)
    atr_sum = atr_val.rolling(window=period, min_periods=1).sum()
    highest_high = high.rolling(window=period, min_periods=1).max()
    lowest_low = low.rolling(window=period, min_periods=1).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-8)) / np.log10(period)
    return chop

def parkinson_volatility(high, low, window=24):
    """Parkinson Volatility using High-Low range"""
    hl_ratio = np.log(high / (low + 1e-8))
    return np.sqrt((hl_ratio ** 2).rolling(window=window, min_periods=1).mean() / (4 * np.log(2)))

def hurst_exponent(series, max_lag=100):
    """Simplified Hurst Exponent estimation using R/S analysis"""
    lags = range(2, min(max_lag, len(series) // 10))
    if len(lags) < 2:
        return pd.Series(0.5, index=series.index)
    
    result = []
    for i in range(len(series)):
        if i < max_lag:
            result.append(0.5)
            continue
        window = series.iloc[max(0, i-max_lag):i+1].values
        if len(window) < max_lag // 2:
            result.append(0.5)
            continue
        try:
            rs_values = []
            for lag in [10, 20, 50]:
                if lag >= len(window):
                    continue
                chunks = [window[j:j+lag] for j in range(0, len(window)-lag, lag)]
                if not chunks:
                    continue
                rs_list = []
                for chunk in chunks:
                    if len(chunk) < 2:
                        continue
                    mean_c = np.mean(chunk)
                    cumdev = np.cumsum(chunk - mean_c)
                    r = np.max(cumdev) - np.min(cumdev)
                    s = np.std(chunk, ddof=1) + 1e-8
                    rs_list.append(r / s)
                if rs_list:
                    rs_values.append((np.log(lag), np.log(np.mean(rs_list))))
            if len(rs_values) >= 2:
                x = np.array([v[0] for v in rs_values])
                y = np.array([v[1] for v in rs_values])
                h = np.polyfit(x, y, 1)[0]
                result.append(np.clip(h, 0, 1))
            else:
                result.append(0.5)
        except:
            result.append(0.5)
    return pd.Series(result, index=series.index)

def connors_rsi(close, rsi_period=3, streak_period=2, roc_period=100):
    """Connors RSI - composite RSI for short-term reversals"""
    # Component 1: Short-term RSI
    rsi_val = rsi(close, rsi_period)
    
    # Component 2: Streak RSI (up/down streak)
    streak = pd.Series(0, index=close.index, dtype=float)
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            streak.iloc[i] = streak.iloc[i-1] + 1 if streak.iloc[i-1] > 0 else 1
        elif close.iloc[i] < close.iloc[i-1]:
            streak.iloc[i] = streak.iloc[i-1] - 1 if streak.iloc[i-1] < 0 else -1
        else:
            streak.iloc[i] = 0
    streak_rsi = rsi(streak, streak_period)
    
    # Component 3: Percent Rank of ROC
    roc = (close - close.shift(1)) / (close.shift(1) + 1e-8)
    pct_rank = roc.rolling(window=roc_period, min_periods=1).apply(
        lambda x: (x[-1] > x[:-1]).sum() / len(x[:-1]) * 100 if len(x) > 1 else 50, raw=True
    )
    
    return (rsi_val + streak_rsi + pct_rank) / 3

def alpha006(open_price, volume, window=10):
    """Alpha 006: -1 * correlation(open, volume, 10)"""
    return -1 * open_price.rolling(window=window, min_periods=1).corr(volume)

def alpha053(high, low, close, delta_period=9):
    """Alpha 053: -1 * delta((((close - low) - (high - close)) / (close - low)), 9)"""
    inner = ((close - low) - (high - close)) / (close - low + 1e-8)
    return -1 * (inner - inner.shift(delta_period))

def alpha101(open_price, high, low, close, volume):
    """Alpha 101: Complex momentum-reversal hybrid"""
    # Simplified Alpha 101: (close - open) / ((high - low) + 0.001)
    return (close - open_price) / (high - low + 1e-8)

# ============== Main Factor Calculation ==============
def calculate_all_factors(df):
    """Calculate all factors for a single coin dataframe"""
    factors = pd.DataFrame(index=df.index)
    factors['timestamp'] = df['open_time']
    
    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # ========== A. 基础市场数据 (Base) ==========
    # 1-4. Log returns (Momentum)
    factors['log_return_1h'] = np.log(close / close.shift(1).replace(0, np.nan))
    factors['log_return_4h'] = np.log(close / close.shift(4).replace(0, np.nan))
    factors['log_return_24h'] = np.log(close / close.shift(24).replace(0, np.nan))
    factors['log_return_168h'] = np.log(close / close.shift(168).replace(0, np.nan))
    
    # 5. Volume Change Ratio (Volume / 24h MA)
    vol_ma_24 = volume.rolling(window=24, min_periods=1).mean()
    factors['volume_change_ratio'] = (volume + 1) / (vol_ma_24 + 1)
    
    # 6. Volatility Structure (24h vs 168h)
    # Using 1h log returns standard deviation
    factors['volatility_24h'] = factors['log_return_1h'].rolling(window=24, min_periods=20).std()
    factors['volatility_168h'] = factors['log_return_1h'].rolling(window=168, min_periods=100).std()
    factors['volatility_term_structure'] = factors['volatility_24h'] / (factors['volatility_168h'] + 1e-8)
    factors['momentum_gap_24h_168h'] = factors['log_return_24h'] - (factors['log_return_168h'] / 7)
    factors['return_skew_24h'] = factors['log_return_1h'].rolling(window=24, min_periods=12).skew()
    factors['amihud_illiquidity_24h'] = (
        factors['log_return_1h'].abs() / (volume.replace(0, np.nan) + 1e-8)
    ).rolling(window=24, min_periods=12).mean()
    log_volume = np.log(volume.replace(0, np.nan) + 1)
    factors['volume_volatility_24h'] = log_volume.diff().rolling(window=24, min_periods=12).std()
    rolling_high_24h = high.rolling(window=24, min_periods=12).max()
    rolling_low_24h = low.rolling(window=24, min_periods=12).min()
    factors['price_range_24h'] = (rolling_high_24h - rolling_low_24h) / (close + 1e-8)
    path_length_24h = factors['log_return_1h'].abs().rolling(window=24, min_periods=12).sum()
    factors['price_efficiency_24h'] = factors['log_return_24h'].abs() / (path_length_24h + 1e-8)
    
    # ========== B. 移除零售指标 (Cleaned) ==========
    # Removed: RSI, MACD, BB, ATR, MFI, OBV, KDJ, etc. per instructions.
    
    return factors

def apply_rolling_zscore(factors_df, window=ZSCORE_WINDOW):
    """Apply rolling z-score to all factor columns except timestamp"""
    result = factors_df.copy()
    factor_cols = [col for col in factors_df.columns if col != 'timestamp']
    
    print(f"  Applying Rolling Z-Score (window={window} hours = {window//24} days)...")
    for col in factor_cols:
        result[col] = rolling_zscore(factors_df[col], window)
    
    return result

def process_coin(coin):
    """Process a single coin: load data, calculate factors, normalize, save"""
    print(f"\n{'='*50}")
    print(f"Processing {coin}...")
    print(f"{'='*50}")
    
    # Load data
    csv_path = DATA_DIR / f"{coin}_1h.csv"
    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found!")
        return None
    
    print(f"  Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)
    print(f"  Loaded {len(df)} rows, date range: {df['open_time'].min()} to {df['open_time'].max()}")
    
    # Calculate factors
    print("  Calculating factors...")
    factors = calculate_all_factors(df)
    
    # Apply rolling z-score normalization
    factors_normalized = apply_rolling_zscore(factors)
    
    # Filter date range
    print(f"  Filtering to date range: {START_DATE} to {END_DATE}...")
    mask = (factors_normalized['timestamp'] >= START_DATE) & (factors_normalized['timestamp'] < END_DATE)
    factors_final = factors_normalized[mask].copy()
    print(f"  After filtering: {len(factors_final)} rows")
    
    # Handle inf/nan
    factors_final = factors_final.replace([np.inf, -np.inf], np.nan)
    
    # Save to parquet
    output_path = DATA_DIR / f"{coin}_factors.parquet"
    factors_final.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")
    
    # Print summary
    factor_cols = [col for col in factors_final.columns if col != 'timestamp']
    print(f"\n  Factor Summary ({len(factor_cols)} factors):")
    print(f"  Columns: {', '.join(factor_cols[:10])}...")
    
    return factors_final

def main():
    print("="*60)
    print("TFT Factor Calculation Script")
    print("="*60)
    print(f"Coins: {COINS}")
    print(f"Z-Score Window: {ZSCORE_WINDOW} hours ({ZSCORE_WINDOW//24} days)")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    
    results = {}
    for coin in COINS:
        results[coin] = process_coin(coin)
    
    print("\n" + "="*60)
    print("COMPLETED!")
    print("="*60)
    print("\nGenerated files:")
    for coin in COINS:
        output_path = DATA_DIR / f"{coin}_factors.parquet"
        if output_path.exists():
            df = pd.read_parquet(output_path)
            print(f"  {output_path.name}: {len(df)} rows, {len(df.columns)} columns")

if __name__ == "__main__":
    main()
