"""
Configuration file for ETH/BTC Statistical Arbitrage Strategy
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class StrategyConfig:
    """Strategy parameters configuration"""
    
    # Data paths
    DATA_DIR: str = r"D:\strategy\data\processed"
    OUTPUT_DIR: str = r"D:\strategy\套利BTC_ETH\output"
    
    # Trading pair
    ASSET_Y: str = "ETHUSDT"  # Dependent variable
    ASSET_X: str = "BTCUSDT"  # Independent variable
    TIMEFRAME: str = "1h"
    
    # Cointegration test parameters
    ADF_SIGNIFICANCE_LEVEL: float = 0.15  # Relaxed for crypto volatility
    LOOKBACK_COINTEGRATION: int = 2000  # Hours for full cointegration test
    
    # Dynamic hedge ratio parameters
    ROLLING_WINDOW: int = 720  # 30 days for rolling OLS
    KALMAN_TRANSITION_COV: float = 0.0001  # Process noise
    KALMAN_OBSERVATION_COV: float = 1.0  # Measurement noise
    USE_KALMAN: bool = True  # Use Kalman Filter vs Rolling OLS
    
    # Signal generation parameters
    ZSCORE_WINDOW: int = 336  # 14 days for Z-score calculation (more stable)
    ZSCORE_ENTRY: float = 3.5  # Entry threshold (wider to reduce noise)
    ZSCORE_EXIT: float = 0.3  # Exit threshold (tighter to lock profits)
    ZSCORE_STOP: float = 4.0  # Stop loss threshold
    
    # Half-life parameters
    HALFLIFE_MIN: int = 6  # Minimum acceptable half-life (hours) - increased
    HALFLIFE_MAX: int = 120  # Maximum acceptable half-life (hours) - decreased
    HALFLIFE_WINDOW: int = 720  # Window for half-life calculation
    
    # Risk management
    MAX_POSITION_SIZE: float = 0.5  # Maximum position size (50% to reduce risk)
    POSITION_SIZING_METHOD: str = "fixed"  # Use fixed sizing for now
    KELLY_FRACTION: float = 0.25  # Conservative Kelly fraction
    MAX_LEVERAGE: float = 2.0  # Maximum leverage (reduced)
    
    # Transaction costs
    MAKER_FEE: float = 0.0002  # 0.02% maker fee
    TAKER_FEE: float = 0.0004  # 0.04% taker fee
    SLIPPAGE_BPS: float = 5.0  # 5 basis points slippage
    
    # Funding rate simulation
    FUNDING_INTERVAL_HOURS: int = 8  # Binance funding every 8 hours
    AVG_FUNDING_RATE_ETH: float = 0.0001  # Average funding rate
    AVG_FUNDING_RATE_BTC: float = 0.0001
    FUNDING_VOLATILITY: float = 0.0002  # Funding rate volatility
    
    # Backtesting parameters
    INITIAL_CAPITAL: float = 100000.0  # USD
    START_DATE: Optional[str] = "2024-01-01"  # Test most recent year only
    END_DATE: Optional[str] = None
    
    # Live trading (for future implementation)
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    TESTNET: bool = True
    
    # Performance monitoring
    REBALANCE_THRESHOLD: float = 0.1  # 10% deviation triggers rebalance
    MAX_DRAWDOWN_STOP: float = 0.20  # 20% max drawdown stop
    
    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

# Global config instance
config = StrategyConfig()
