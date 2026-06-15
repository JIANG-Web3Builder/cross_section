"""
Data loading and preprocessing module for ETH/BTC pairs trading
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Load and preprocess market data for pairs trading"""
    
    def __init__(self, data_dir: str, asset_y: str, asset_x: str, timeframe: str):
        self.data_dir = Path(data_dir)
        self.asset_y = asset_y
        self.asset_x = asset_x
        self.timeframe = timeframe
        
    def load_data(self, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load and merge ETH and BTC data
        
        Returns:
            DataFrame with columns: eth_close, btc_close, eth_volume, btc_volume
        """
        # Load ETH data
        eth_file = self.data_dir / f"{self.asset_y}_{self.timeframe}.csv"
        eth_df = pd.read_csv(eth_file)
        eth_df['open_time'] = pd.to_datetime(eth_df['open_time'])
        eth_df = eth_df.set_index('open_time')
        eth_df = eth_df.rename(columns={
            'close': 'eth_close',
            'volume': 'eth_volume',
            'open': 'eth_open',
            'high': 'eth_high',
            'low': 'eth_low'
        })
        
        # Load BTC data
        btc_file = self.data_dir / f"{self.asset_x}_{self.timeframe}.csv"
        btc_df = pd.read_csv(btc_file)
        btc_df['open_time'] = pd.to_datetime(btc_df['open_time'])
        btc_df = btc_df.set_index('open_time')
        btc_df = btc_df.rename(columns={
            'close': 'btc_close',
            'volume': 'btc_volume',
            'open': 'btc_open',
            'high': 'btc_high',
            'low': 'btc_low'
        })
        
        # Merge on timestamp (inner join to ensure alignment)
        df = eth_df.join(btc_df, how='inner', rsuffix='_btc')
        
        # Filter by date range if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        # Remove any NaN values
        df = df.dropna()
        
        print(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")
        print(f"ETH price range: ${df['eth_close'].min():.2f} - ${df['eth_close'].max():.2f}")
        print(f"BTC price range: ${df['btc_close'].min():.2f} - ${df['btc_close'].max():.2f}")
        
        return df
    
    def add_log_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log-transformed prices for statistical analysis"""
        df['log_eth'] = np.log(df['eth_close'])
        df['log_btc'] = np.log(df['btc_close'])
        return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns"""
        df['eth_return'] = df['log_eth'].diff()
        df['btc_return'] = df['log_btc'].diff()
        return df
    
    def add_volatility_features(self, df: pd.DataFrame, windows: list = [24, 168]) -> pd.DataFrame:
        """Add rolling volatility features"""
        for window in windows:
            df[f'eth_vol_{window}h'] = df['eth_return'].rolling(window).std() * np.sqrt(window)
            df[f'btc_vol_{window}h'] = df['btc_return'].rolling(window).std() * np.sqrt(window)
            df[f'vol_ratio_{window}h'] = df[f'eth_vol_{window}h'] / df[f'btc_vol_{window}h']
        return df
    
    def prepare_full_dataset(self, start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """Load and prepare complete dataset with all features"""
        df = self.load_data(start_date, end_date)
        df = self.add_log_prices(df)
        df = self.calculate_returns(df)
        df = self.add_volatility_features(df)
        
        # Drop initial NaN rows from rolling calculations
        df = df.dropna()
        
        print(f"\nFinal dataset: {len(df)} rows")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
