"""
Signal Generation Module for Statistical Arbitrage
Generates trading signals based on Z-score and mean reversion indicators
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SignalGenerator:
    """
    Generate trading signals for pairs trading strategy
    
    Signal Logic:
    - Z > +entry_threshold: Short Spread (Short ETH, Long BTC)
    - Z < -entry_threshold: Long Spread (Long ETH, Short BTC)
    - Z returns to exit_threshold: Close position
    - |Z| > stop_threshold: Stop loss
    """
    
    def __init__(self, 
                 z_entry: float = 2.0,
                 z_exit: float = 0.5,
                 z_stop: float = 4.0,
                 zscore_window: int = 168,
                 halflife_min: int = 2,
                 halflife_max: int = 240):
        """
        Args:
            z_entry: Z-score threshold for entry
            z_exit: Z-score threshold for exit
            z_stop: Z-score threshold for stop loss
            zscore_window: Rolling window for Z-score calculation (hours)
            halflife_min: Minimum acceptable half-life (hours)
            halflife_max: Maximum acceptable half-life (hours)
        """
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.zscore_window = zscore_window
        self.halflife_min = halflife_min
        self.halflife_max = halflife_max
        
    def calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """
        Calculate rolling Z-score of spread
        
        Z = (spread - rolling_mean) / rolling_std
        """
        rolling_mean = spread.rolling(window=self.zscore_window, min_periods=1).mean()
        rolling_std = spread.rolling(window=self.zscore_window, min_periods=1).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        zscore = (spread - rolling_mean) / rolling_std
        
        return zscore
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion
        """
        from hedge_ratio import SpreadAnalyzer
        return SpreadAnalyzer.calculate_half_life(spread)
    
    def rolling_half_life(self, spread: pd.Series, window: int = 720) -> pd.Series:
        """
        Calculate rolling half-life
        """
        from hedge_ratio import SpreadAnalyzer
        return SpreadAnalyzer.rolling_half_life(spread, window)
    
    def generate_signals(self, zscore: pd.Series, 
                        half_life: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate trading signals based on Z-score
        
        Returns:
            DataFrame with columns: signal, position, regime
            - signal: 1 (Long Spread), -1 (Short Spread), 0 (Flat)
            - position: Cumulative position state
            - regime: 'tradeable' or 'non_tradeable' based on half-life
        """
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        signals['signal'] = 0
        signals['position'] = 0
        signals['regime'] = 'tradeable'
        
        # Check half-life regime if provided
        if half_life is not None:
            # Align half_life with signals index
            common_idx = signals.index.intersection(half_life.index)
            signals = signals.loc[common_idx]
            half_life_aligned = half_life.loc[common_idx]
            signals['half_life'] = half_life_aligned
            
            # Mark non-tradeable regimes
            non_tradeable_mask = (half_life_aligned < self.halflife_min) | (half_life_aligned > self.halflife_max)
            signals.loc[non_tradeable_mask, 'regime'] = 'non_tradeable'
        
        position = 0  # Current position state
        
        for i, idx in enumerate(signals.index):
            z = signals.loc[idx, 'zscore']
            
            # Skip if in non-tradeable regime
            if signals.loc[idx, 'regime'] == 'non_tradeable':
                if position != 0:
                    # Force close position
                    signals.loc[idx, 'signal'] = -position
                    position = 0
                signals.loc[idx, 'position'] = position
                continue
            
            # Stop loss logic
            if abs(z) > self.z_stop:
                if position != 0:
                    signals.loc[idx, 'signal'] = -position  # Close position
                    position = 0
            
            # Entry logic (when flat)
            elif position == 0:
                if z > self.z_entry:
                    signals.loc[idx, 'signal'] = -1  # Short Spread
                    position = -1
                elif z < -self.z_entry:
                    signals.loc[idx, 'signal'] = 1  # Long Spread
                    position = 1
            
            # Exit logic (when in position)
            elif position == -1:  # Currently short spread
                if z <= self.z_exit:
                    signals.loc[idx, 'signal'] = 1  # Close (buy back)
                    position = 0
            
            elif position == 1:  # Currently long spread
                if z >= -self.z_exit:
                    signals.loc[idx, 'signal'] = -1  # Close (sell)
                    position = 0
            
            signals.loc[idx, 'position'] = position
        
        return signals
    
    def generate_adaptive_signals(self, zscore: pd.Series, 
                                  volatility: pd.Series,
                                  half_life: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Advanced signal generation with volatility-adjusted thresholds
        
        In high volatility regimes, widen the entry threshold to avoid false signals
        """
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        signals['volatility'] = volatility
        signals['signal'] = 0
        signals['position'] = 0
        signals['regime'] = 'tradeable'
        
        # Normalize volatility
        vol_mean = volatility.rolling(window=720, min_periods=1).mean()
        vol_ratio = volatility / vol_mean
        
        # Adjust thresholds based on volatility
        # High vol -> wider thresholds
        signals['z_entry_adj'] = self.z_entry * vol_ratio
        signals['z_exit_adj'] = self.z_exit * vol_ratio
        
        # Check half-life regime
        if half_life is not None:
            signals['half_life'] = half_life
            non_tradeable = (half_life < self.halflife_min) | (half_life > self.halflife_max)
            signals.loc[non_tradeable, 'regime'] = 'non_tradeable'
        
        position = 0
        
        for i, idx in enumerate(zscore.index):
            z = zscore.loc[idx]
            z_entry_adj = signals.loc[idx, 'z_entry_adj']
            z_exit_adj = signals.loc[idx, 'z_exit_adj']
            
            # Skip if in non-tradeable regime
            if signals.loc[idx, 'regime'] == 'non_tradeable':
                if position != 0:
                    signals.loc[idx, 'signal'] = -position
                    position = 0
                signals.loc[idx, 'position'] = position
                continue
            
            # Stop loss
            if abs(z) > self.z_stop:
                if position != 0:
                    signals.loc[idx, 'signal'] = -position
                    position = 0
            
            # Entry
            elif position == 0:
                if z > z_entry_adj:
                    signals.loc[idx, 'signal'] = -1
                    position = -1
                elif z < -z_entry_adj:
                    signals.loc[idx, 'signal'] = 1
                    position = 1
            
            # Exit
            elif position == -1:
                if z <= z_exit_adj:
                    signals.loc[idx, 'signal'] = 1
                    position = 0
            
            elif position == 1:
                if z >= -z_exit_adj:
                    signals.loc[idx, 'signal'] = -1
                    position = 0
            
            signals.loc[idx, 'position'] = position
        
        return signals
    
    def calculate_signal_quality_metrics(self, signals: pd.DataFrame) -> dict:
        """
        Calculate metrics to assess signal quality
        """
        # Count trades
        trades = signals[signals['signal'] != 0]
        n_trades = len(trades)
        
        # Entry/exit breakdown
        entries = signals[signals['signal'].abs() == 1]
        n_entries = len(entries[entries['position'].shift(1) == 0])
        
        # Average holding period
        position_changes = signals['position'].diff().fillna(0)
        entry_indices = signals[position_changes != 0].index
        
        holding_periods = []
        for i in range(len(entry_indices) - 1):
            if signals.loc[entry_indices[i], 'position'] != 0:
                holding_periods.append((entry_indices[i+1] - entry_indices[i]).total_seconds() / 3600)
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # Time in market
        time_in_market = (signals['position'] != 0).sum() / len(signals)
        
        # Regime analysis
        if 'regime' in signals.columns:
            tradeable_pct = (signals['regime'] == 'tradeable').sum() / len(signals)
        else:
            tradeable_pct = 1.0
        
        metrics = {
            'total_signals': n_trades,
            'entry_signals': n_entries,
            'avg_holding_period_hours': avg_holding_period,
            'time_in_market_pct': time_in_market * 100,
            'tradeable_regime_pct': tradeable_pct * 100
        }
        
        return metrics


class PositionSizer:
    """
    Calculate position sizes based on risk management rules
    """
    
    def __init__(self, method: str = 'fixed', max_leverage: float = 3.0):
        """
        Args:
            method: 'fixed', 'kelly', 'volatility'
            max_leverage: Maximum allowed leverage
        """
        self.method = method
        self.max_leverage = max_leverage
    
    def fixed_size(self, capital: float, leverage: float = 1.0) -> float:
        """Fixed position size"""
        return capital * min(leverage, self.max_leverage)
    
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float,
                       kelly_fraction: float = 0.25) -> float:
        """
        Kelly Criterion for optimal position sizing
        
        f* = (p * b - q) / b
        where:
        - p = win rate
        - q = 1 - p
        - b = avg_win / avg_loss
        
        Use fractional Kelly for safety
        """
        if avg_loss == 0:
            return 0
        
        b = avg_win / abs(avg_loss)
        q = 1 - win_rate
        
        kelly_pct = (win_rate * b - q) / b
        kelly_pct = max(0, min(kelly_pct, 1))  # Clamp to [0, 1]
        
        # Apply fractional Kelly
        return kelly_pct * kelly_fraction
    
    def volatility_based(self, spread_volatility: float, target_volatility: float = 0.02) -> float:
        """
        Size position inversely to volatility
        Target a constant volatility exposure
        """
        if spread_volatility == 0:
            return 0
        
        size = target_volatility / spread_volatility
        return min(size, self.max_leverage)
    
    def calculate_position_size(self, capital: float, signal: int, 
                               spread_vol: Optional[float] = None,
                               **kwargs) -> float:
        """
        Calculate position size based on selected method
        
        Returns:
            Position size in USD
        """
        if signal == 0:
            return 0
        
        if self.method == 'fixed':
            leverage = kwargs.get('leverage', 1.0)
            return self.fixed_size(capital, leverage)
        
        elif self.method == 'kelly':
            win_rate = kwargs.get('win_rate', 0.5)
            avg_win = kwargs.get('avg_win', 0.01)
            avg_loss = kwargs.get('avg_loss', 0.01)
            kelly_fraction = kwargs.get('kelly_fraction', 0.25)
            
            kelly_pct = self.kelly_criterion(win_rate, avg_win, avg_loss, kelly_fraction)
            return capital * kelly_pct
        
        elif self.method == 'volatility':
            if spread_vol is None:
                raise ValueError("spread_vol required for volatility-based sizing")
            
            target_vol = kwargs.get('target_volatility', 0.02)
            vol_multiplier = self.volatility_based(spread_vol, target_vol)
            return capital * vol_multiplier
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
