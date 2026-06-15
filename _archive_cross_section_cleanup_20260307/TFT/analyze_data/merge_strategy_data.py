
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Configuration
DATA_DIR = r"D:\strategy\TFT\factor"
OUTPUT_FILE = os.path.join(DATA_DIR, "merged_tft_data.parquet")
REPORT_FILE = os.path.join(DATA_DIR, "merge_report.json")

def load_and_prep_ohlcv(filepath, prefix):
    print(f"Loading OHLCV: {filepath}")
    df = pd.read_csv(filepath)
    # Parse dates
    df['timestamp'] = pd.to_datetime(df['open_time'])
    df = df.set_index('timestamp').sort_index()
    # Select relevant cols and rename
    cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[cols].copy()
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df

def load_and_prep_metrics(filepath, prefix):
    print(f"Loading Metrics: {filepath}")
    df = pd.read_parquet(filepath)
    # Check if index is datetime, if not look for valid col
    if not isinstance(df.index, pd.DatetimeIndex):
        # Fallback if parquet didn't save index properly
        pass 
    
    # Rename all columns with prefix
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df

def load_and_prep_factors(filepath, prefix):
    print(f"Loading Factors: {filepath}")
    df = pd.read_parquet(filepath)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
    
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df

def load_and_prep_global(filepath, index_col=None):
    print(f"Loading Global: {filepath}")
    df = pd.read_parquet(filepath)
    
    if index_col and index_col in df.columns:
        df[index_col] = pd.to_datetime(df[index_col])
        df = df.set_index(index_col).sort_index()
        
    # Remove timezone info if present to align with others (usually naive in crypto data)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
        
    return df

def add_time_features(df):
    print("Generating time features...")
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    return df

def main():
    dfs = []
    
    # 1. Load BTC OHLCV (Primary Spine)
    btc_ohlcv = load_and_prep_ohlcv(os.path.join(DATA_DIR, "BTCUSDT_1h.csv"), "BTC")
    dfs.append(btc_ohlcv)
    
    # 2. Load ETH OHLCV
    eth_ohlcv = load_and_prep_ohlcv(os.path.join(DATA_DIR, "ETHUSDT_1h.csv"), "ETH")
    dfs.append(eth_ohlcv)
    
    # 3. Load Metrics
    btc_metrics = load_and_prep_metrics(os.path.join(DATA_DIR, "BTCUSDT_combined_1H.parquet"), "BTC")
    dfs.append(btc_metrics)
    
    eth_metrics = load_and_prep_metrics(os.path.join(DATA_DIR, "ETHUSDT_combined_1H.parquet"), "ETH")
    dfs.append(eth_metrics)
    
    # 4. Load Factors
    btc_factors = load_and_prep_factors(os.path.join(DATA_DIR, "BTCUSDT_factors.parquet"), "BTC")
    dfs.append(btc_factors)
    
    eth_factors = load_and_prep_factors(os.path.join(DATA_DIR, "ETHUSDT_factors.parquet"), "ETH")
    dfs.append(eth_factors)
    
    # 5. Load Global Data
    # Dominance (Index is already TS)
    dom_df = load_and_prep_global(os.path.join(DATA_DIR, "btc_dominance_1h.parquet"))
    dfs.append(dom_df)
    
    # Correlation (TS is column)
    corr_df = load_and_prep_global(os.path.join(DATA_DIR, "btc_spx_correlation.parquet"), index_col="timestamp")
    dfs.append(corr_df)
    
    # 6. Merge All
    print("Merging all dataframes...")
    # Join outer to keep all history, then we can trim
    full_df = pd.concat(dfs, axis=1)
    
    # 7. Post-Merge Processing
    # Sort index
    full_df = full_df.sort_index()
    
    # Generate time features dynamically (replacing known_time.parquet)
    full_df = add_time_features(full_df)
    
    # Handle Missing Values (Forward fill then Backward fill for edges)
    # Limit ffill to avoid filling huge gaps if data stopped, but here we assume continuous history
    print("Handling missing values...")
    
    # Record missing before fill
    missing_before = full_df.isnull().sum().to_dict()
    
    full_df = full_df.ffill()
    # Optional: Fill remaining NaNs (start of data) with 0 or drop?
    # For TFT, it's often better to drop the initial rows where indicators aren't calculated
    # Let's drop rows where BTC_close is NaN (market didn't exist or file issue)
    full_df = full_df.dropna(subset=['BTC_close'])
    
    # Record missing after basic fill
    missing_after = full_df.isnull().sum().to_dict()

    # Optimization: Downcast float64 to float32 to save space (approx 50% reduction)
    print("Optimizing memory usage (float64 -> float32)...")
    float64_cols = full_df.select_dtypes(include=['float64']).columns
    full_df[float64_cols] = full_df[float64_cols].astype('float32')
    
    # 8. Save
    print(f"Saving to {OUTPUT_FILE}...")
    # Use snappy compression (usually default) but ensure it's on
    full_df.to_parquet(OUTPUT_FILE, compression='snappy')
    
    # 9. Generate Report
    report = {
        "start_date": str(full_df.index.min()),
        "end_date": str(full_df.index.max()),
        "total_rows": len(full_df),
        "total_columns": len(full_df.columns),
        "columns": list(full_df.columns),
        "missing_stats": {
            "before_fill": missing_before,
            "after_fill": missing_after
        }
    }
    
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=4)
        
    print("Done.")

if __name__ == "__main__":
    main()
