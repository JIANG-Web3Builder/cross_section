"""
Module 1: TFT Data Engineering
合并factor目录下数据文件为TFT训练宽表 (Multi-Symbol Support)

Supported Symbols: BTC, ETH
Architecture:
- Stacked DataFrame (Long Format)
- Asset-Specific Features (Generic Names): log_return, volatility, etc.
- Shared Macro Features: dominance, correlation, cross-asset ratios.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config_tft import (
    FACTOR_DIR, FINAL_DIR, TARGET_FORWARD_HOURS, TARGET_NAME, 
    FFILL_LIMIT, START_DATE, END_DATE,
    load_selected_factor_features, load_selected_factor_directions,
    refresh_tft_variables
)


def compute_triple_barrier_labels(
    df: pd.DataFrame, 
    close_col: str = 'close',
    vol_span: int = 24,
    horizon: int = 24,
    upper_width: float = 0.5,
    lower_width: float = 0.5
) -> pd.DataFrame:
    """
    Tier-1 Labeling: 三重势垒法 (Applied per asset)
    """
    df = df.copy()
    
    # Check if close column exists
    if close_col not in df.columns:
        raise ValueError(f"Close column '{close_col}' not found in dataframe")

    # 1. Calculate dynamic volatility
    df['returns'] = df[close_col].pct_change()
    df['volatility'] = df['returns'].ewm(span=vol_span).std().shift(1)
    
    labels = []
    closes = df[close_col].values
    vols = df['volatility'].values
    
    n = len(df)
    
    for t in range(n - horizon):
        current_p = closes[t]
        current_vol = vols[t]
        
        if np.isnan(current_vol) or current_vol < 1e-4:
            current_vol = 0.005
        
        upper_barrier = current_p * (1 + current_vol * upper_width)
        lower_barrier = current_p * (1 - current_vol * lower_width)
        
        future_path = closes[t+1 : t+1+horizon]
        
        hit_upper_mask = future_path >= upper_barrier
        hit_lower_mask = future_path <= lower_barrier
        
        first_upper_idx = np.argmax(hit_upper_mask) if hit_upper_mask.any() else 9999
        first_lower_idx = np.argmax(hit_lower_mask) if hit_lower_mask.any() else 9999
        
        label = 0
        if first_upper_idx == 9999 and first_lower_idx == 9999:
            label = 0
        elif first_upper_idx < first_lower_idx:
            label = 1
        elif first_lower_idx < first_upper_idx:
            label = 2
        
        labels.append(label)
    
    labels.extend([np.nan] * horizon)
    df['target_class'] = labels
    
    return df


class TFTDataProcessor:
    """TFT Data Engineering Pipeline - Multi-Symbol"""
    
    def __init__(self):
        refresh_tft_variables()
        self.factor_dir = FACTOR_DIR
        self.output_dir = FINAL_DIR
        self.selected_factor_features = load_selected_factor_features()
        self.factor_directions = load_selected_factor_directions()
        
    def load_parquet(self, filepath: Path) -> pd.DataFrame:
        """Load parquet and set timestamp index"""
        print(f"  Loading {filepath.name}...")
        df = pd.read_parquet(filepath)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
        elif not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                pass
        
        # Remove timezone
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
            
        return df

    def generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate cyclic time features"""
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_of_week'] = df.index.dayofweek
        return df

    def prepare_macro_features(self) -> pd.DataFrame:
        """
        Load and prepare shared macro features:
        1. BTC Dominance
        2. SPX Correlation
        3. ETH/BTC Ratio (Derived)
        """
        print("\n" + "=" * 40)
        print("PREPARING MACRO FEATURES")
        print("=" * 40)
        
        # 1. BTC Dominance
        dom_df = self.load_parquet(self.factor_dir / "btc_dominance_1h.parquet")
        if 'btcdom_change_zscore' in dom_df.columns:
            dom_df = dom_df.rename(columns={'btcdom_change_zscore': 'btc_dominance_change'})
        dom_cols = ['btc_dominance_change']
        dom_df = dom_df[ [c for c in dom_cols if c in dom_df.columns] ]
        
        # 2. SPX Correlation
        corr_df = self.load_parquet(self.factor_dir / "btc_spx_correlation.parquet")
        if 'corr_7d' in corr_df.columns:
            corr_df = corr_df.rename(columns={'corr_7d': 'spx_correlation_rolling_7d'})
        corr_cols = ['spx_correlation_rolling_7d']
        corr_df = corr_df[ [c for c in corr_cols if c in corr_df.columns] ]
        
        # 3. ETH/BTC Ratio
        # Need close prices for both
        btc_ohlcv = pd.read_csv(self.factor_dir / "BTCUSDT_1h.csv")
        eth_ohlcv = pd.read_csv(self.factor_dir / "ETHUSDT_1h.csv")
        
        for df in [btc_ohlcv, eth_ohlcv]:
            df['timestamp'] = pd.to_datetime(df['open_time'])
            df.set_index('timestamp', inplace=True)
            if df.index.tz is not None: df.index = df.index.tz_convert(None)
            
        common_idx = btc_ohlcv.index.intersection(eth_ohlcv.index)
        ratio_df = pd.DataFrame(index=common_idx)
        ratio_df['ratio'] = eth_ohlcv.loc[common_idx, 'close'] / (btc_ohlcv.loc[common_idx, 'close'] + 1e-8)
        ratio_df['eth_btc_ratio_change'] = np.log(ratio_df['ratio'] / ratio_df['ratio'].shift(1).replace(0, np.nan))
        
        # Merge all macro
        macro_df = pd.concat([dom_df, corr_df, ratio_df[['eth_btc_ratio_change']]], axis=1)
        macro_df = macro_df.sort_index()
        
        # Fill macro NaNs (forward fill then 0)
        macro_df = macro_df.ffill().fillna(0.0)
        
        print(f"  Macro features prepared: {macro_df.shape}")
        print(f"  Columns: {list(macro_df.columns)}")
        
        return macro_df

    def process_asset(self, symbol: str, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process single asset:
        1. Load Factors (Generic)
        2. Load Derivative Data (Funding, OI, etc.)
        3. Load Close Price
        4. Calculate Derived Factors (Vol Term Structure)
        5. Calculate Target
        6. Merge Macro
        """
        print(f"\nProcessing Asset: {symbol}...")
        
        # 1. Load Factors (Generic)
        factor_file = self.factor_dir / f"{symbol}USDT_factors.parquet"
        if not factor_file.exists():
            raise FileNotFoundError(f"Factor file not found: {factor_file}")
            
        factors_df = self.load_parquet(factor_file)
        available_factor_features = [
            col for col in self.selected_factor_features if col in factors_df.columns
        ]
        if not available_factor_features:
            raise ValueError(f"No selected factor columns found in {factor_file.name}")

        factors_df = factors_df[available_factor_features].copy()
        for feature in available_factor_features:
            direction = self.factor_directions.get(feature, 1)
            factors_df[feature] = factors_df[feature] * direction
        
        # 2. Load Derivative Data
        deriv_file = self.factor_dir / f"{symbol}USDT_combined_1H.parquet"
        if deriv_file.exists():
            print(f"  Loading derivatives: {deriv_file.name}")
            deriv_df = self.load_parquet(deriv_file)
            
            # Rename to generic names
            rename_map = {
                'last_funding_rate': 'funding_rate',
                'sum_open_interest_value': 'open_interest_value',
                'sum_toptrader_long_short_ratio': 'top_trader_ls_ratio',
                'sum_taker_long_short_vol_ratio': 'taker_buy_sell_ratio'
            }
            deriv_df = deriv_df.rename(columns=rename_map)
            
            # Select only needed columns
            needed_cols = list(rename_map.values())
            available_cols = [c for c in needed_cols if c in deriv_df.columns]
            deriv_df = deriv_df[available_cols]
            
            # Merge into factors (align by index)
            # Use join to keep factors_df index (which should be the master time index usually)
            # But factors_df might have gaps? Better to join carefully.
            # Assuming factors_df is the anchor.
            factors_df = factors_df.join(deriv_df, how='left')
        else:
            print(f"  WARNING: Derivative file not found: {deriv_file}")

        # 3. Load OHLCV for Close price (Target calculation)
        ohlcv_file = self.factor_dir / f"{symbol}USDT_1h.csv"
        ohlcv_df = pd.read_csv(ohlcv_file)
        ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['open_time'])
        ohlcv_df.set_index('timestamp', inplace=True)
        if ohlcv_df.index.tz is not None: ohlcv_df.index = ohlcv_df.index.tz_convert(None)
        
        # Merge Close price into factors
        factors_df['close'] = ohlcv_df['close']
        
        # 4. Derived Factors
        if 'volatility_term_structure' not in factors_df.columns and 'volatility_24h' in factors_df.columns and 'volatility_168h' in factors_df.columns:
            factors_df['volatility_term_structure'] = factors_df['volatility_24h'] / (factors_df['volatility_168h'] + 1e-8)
            
        # 5. Calculate Target
        factors_df = compute_triple_barrier_labels(
            factors_df, 
            close_col='close',
            horizon=TARGET_FORWARD_HOURS,
            upper_width=0.5,
            lower_width=0.5
        )
        
        # Drop rows with NaN target (before stacking)
        factors_df = factors_df.dropna(subset=['target_class'])
        
        # Convert target to int
        factors_df[TARGET_NAME] = factors_df['target_class'].astype(int)
        
        # 5. Merge Macro
        # Align timestamps
        common_idx = factors_df.index.intersection(macro_df.index)
        factors_df = factors_df.loc[common_idx]
        factors_df = pd.concat([factors_df, macro_df.loc[common_idx]], axis=1)
        
        # 6. Add Symbol
        factors_df['symbol'] = symbol
        
        # 7. Time Features
        factors_df = self.generate_time_features(factors_df)
        
        # 8. Handle NaNs (Per Asset)
        # Backfill initial lags then fill with 0
        feature_cols = [c for c in factors_df.columns if c not in [TARGET_NAME, 'symbol', 'close', 'target_class']]
        factors_df[feature_cols] = factors_df[feature_cols].ffill(limit=FFILL_LIMIT).bfill().fillna(0.0)
        
        print(f"  Processed {symbol}: {factors_df.shape}")
        return factors_df

    def run_pipeline(self) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("TFT DATA PIPELINE (Multi-Symbol)")
        print("=" * 60)
        
        # 1. Macro Features
        macro_df = self.prepare_macro_features()
        
        # 2. Process Assets
        assets = ['BTC', 'ETH']
        dfs = []
        for asset in assets:
            try:
                df_asset = self.process_asset(asset, macro_df)
                dfs.append(df_asset)
            except Exception as e:
                print(f"Error processing {asset}: {e}")
        
        if not dfs:
            raise ValueError("No data processed!")
            
        # 3. Stack
        print("\nStacking DataFrames...")
        full_df = pd.concat(dfs, axis=0)
        
        # 4. Global Filtering
        print(f"Filtering date range: {START_DATE} to {END_DATE}")
        full_df = full_df[(full_df.index >= START_DATE) & (full_df.index < END_DATE)]
        
        # 5. Add time_idx (Global, per group handled by TFT internally via relative_time_idx usually, 
        # but pytorch-forecasting needs continuous time_idx for the whole dataset?
        # Actually for multiple groups, time_idx should be absolute time index shared across groups)
        # We define time_idx based on timestamp rank
        
        # Create a map from timestamp to integer
        unique_timestamps = sorted(full_df.index.unique())
        ts_map = {ts: i for i, ts in enumerate(unique_timestamps)}
        full_df['time_idx'] = full_df.index.map(ts_map)
        
        # Sort
        full_df = full_df.sort_values(['symbol', 'time_idx'])
        
        # 6. Validate
        print("\nValidating...")
        print(f"  Shape: {full_df.shape}")
        print(f"  Symbols: {full_df['symbol'].unique()}")
        print(f"  Target NaNs: {full_df[TARGET_NAME].isnull().sum()}")
        print(f"  Feature NaNs: {full_df.isnull().sum().sum()}")
        
        # 7. Save
        output_path = self.output_dir / 'tft_training_dataset.parquet'
        # Reset index to save timestamp
        save_df = full_df.reset_index()
        save_df.to_parquet(output_path, index=False)
        
        print(f"\n✓ Saved to {output_path}")
        print(f"  Columns: {list(full_df.columns)}")
        
        return full_df

def main():
    processor = TFTDataProcessor()
    processor.run_pipeline()

if __name__ == "__main__":
    main()
