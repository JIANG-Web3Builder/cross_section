import pandas as pd
import numpy as np
from config_tft import TFT_VARIABLES, FINAL_DIR, TARGET_NAME, refresh_tft_variables

def check_data():
    refresh_tft_variables()
    data_path = FINAL_DIR / 'tft_training_dataset.parquet'
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return

    df = pd.read_parquet(data_path)
    print(f"Loaded data shape: {df.shape}")
    
    print(f"  Symbols: {df['symbol'].unique()}")
    
    expected_symbols = ['BTC', 'ETH']
    if set(df['symbol'].unique()) == set(expected_symbols):
        print("✓ Symbols are correct (BTC, ETH).")
    else:
        print(f"✗ Unexpected symbols: {df['symbol'].unique()}")

    # Check relevant columns
    relevant_cols = (
        TFT_VARIABLES['static_categoricals'] +
        TFT_VARIABLES['time_varying_known_reals'] +
        TFT_VARIABLES['time_varying_unknown_reals'] +
        [TARGET_NAME, 'time_idx']
    )
    
    # Explicitly check for new derivative features
    deriv_features = ['funding_rate', 'open_interest_value', 'top_trader_ls_ratio', 'taker_buy_sell_ratio']
    print("\nChecking for new derivative features:")
    for feat in deriv_features:
        if feat in df.columns:
            print(f"  ✓ Found {feat}")
        else:
            print(f"  ✗ MISSING {feat}")

    print("\nChecking for NaNs/Infs in relevant columns:")
    has_error = False
    for col in relevant_cols:
        if col not in df.columns:
            print(f"MISSING COLUMN: {col}")
            has_error = True
            continue
            
        n_nans = df[col].isnull().sum()
        n_infs = np.isinf(df[col]).sum() if np.issubdtype(df[col].dtype, np.number) else 0
        
        if n_nans > 0 or n_infs > 0:
            print(f"  {col}: NaNs={n_nans}, Infs={n_infs}")
            has_error = True
            
    if not has_error:
        print("✓ No NaNs or Infs found in relevant columns.")
    else:
        print("✗ Found issues in relevant columns.")

if __name__ == "__main__":
    check_data()
