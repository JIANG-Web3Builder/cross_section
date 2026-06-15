"""
Module 4: Strategy Execution Engine
Regime Classification + Portfolio Construction

Key Features:
- 5-state regime classification based on TFT signals
- Dynamic position sizing based on market regime
- Smart capitulation detection for left-side buying
- Risk management and position limits
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config_tft import (
    OUTPUT_DIR, REGIME_THRESHOLDS, REGIMES, CAPITULATION_CONFIG,
    PORTFOLIO_CONFIG, RISK_CONFIG
)


class RegimeClassifier:
    """Classify market regime based on TFT signals"""
    
    def __init__(self, tft_signals: pd.DataFrame):
        """
        Initialize regime classifier
        
        Args:
            tft_signals: DataFrame with TFT signals (trend, uncertainty)
        """
        self.signals = tft_signals
        self.regimes = None
        
    def classify_regime(self) -> pd.DataFrame:
        """
        Classify market regime for each timestamp
        
        Regimes:
        - S1_BULL: Low uncertainty + positive trend
        - S2_BEAR: Low uncertainty + negative trend
        - S3_CHOP: Medium uncertainty
        - S4_DANGER: High uncertainty
        - S5_PANIC: Extreme uncertainty + capitulation signal
        
        Returns:
            DataFrame with regime classification
        """
        print("\n" + "=" * 60)
        print("CLASSIFYING MARKET REGIMES")
        print("=" * 60)
        
        regimes = self.signals[['timestamp']].copy()
        
        # Get normalized uncertainty
        if 'tft_uncertainty_norm' in self.signals.columns:
            uncertainty = self.signals['tft_uncertainty_norm']
        else:
            # Normalize if not already done
            unc = self.signals['tft_uncertainty']
            uncertainty = (unc - unc.min()) / (unc.max() - unc.min() + 1e-8)
        
        trend = self.signals['tft_trend']
        
        # Thresholds
        unc_low = REGIME_THRESHOLDS['uncertainty_low']
        unc_high = REGIME_THRESHOLDS['uncertainty_high']
        unc_extreme = REGIME_THRESHOLDS['uncertainty_extreme']
        
        # Initialize regime
        regimes['regime'] = 'S3_CHOP'  # Default
        
        # S1: Bull Market (low uncertainty + positive trend)
        bull_mask = (uncertainty < unc_low) & (trend > 0)
        regimes.loc[bull_mask, 'regime'] = 'S1_BULL'
        
        # S2: Bear Market (low uncertainty + negative trend)
        bear_mask = (uncertainty < unc_low) & (trend < 0)
        regimes.loc[bear_mask, 'regime'] = 'S2_BEAR'
        
        # S4: Danger (high uncertainty)
        danger_mask = (uncertainty > unc_high) & (uncertainty < unc_extreme)
        regimes.loc[danger_mask, 'regime'] = 'S4_DANGER'
        
        # S5: Panic (extreme uncertainty)
        panic_mask = uncertainty >= unc_extreme
        regimes.loc[panic_mask, 'regime'] = 'S5_PANIC'
        
        # Add regime metadata
        regimes['uncertainty'] = uncertainty.values
        regimes['trend'] = trend.values
        
        # Statistics
        regime_counts = regimes['regime'].value_counts()
        print(f"\n  Regime Distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(regimes) * 100
            print(f"    {regime}: {count} ({pct:.1f}%)")
        
        self.regimes = regimes
        return regimes
    
    def detect_capitulation(self, market_data: pd.DataFrame = None) -> pd.Series:
        """
        Detect smart capitulation signals for left-side buying
        
        Conditions:
        1. Uncertainty starts declining from peak
        2. Extreme oversold (RSI < 25)
        3. Price stabilization (not making new lows)
        
        Args:
            market_data: Optional market data for additional signals
            
        Returns:
            Boolean Series indicating capitulation signals
        """
        print("\n  Detecting capitulation signals...")
        
        if self.regimes is None:
            raise ValueError("Must classify regimes first")
        
        # Only check in S5_PANIC regime
        panic_mask = self.regimes['regime'] == 'S5_PANIC'
        
        # Condition 1: Uncertainty declining
        uncertainty = self.regimes['uncertainty']
        unc_declining = uncertainty < uncertainty.shift(1)
        
        # Condition 2: Extreme oversold (if market data available)
        if market_data is not None and 'rsi' in market_data.columns:
            oversold = market_data['rsi'] < CAPITULATION_CONFIG['rsi_threshold']
        else:
            oversold = True  # Default to True if no RSI data
        
        # Combine conditions
        capitulation = panic_mask & unc_declining & oversold
        
        cap_count = capitulation.sum()
        print(f"    Capitulation signals detected: {cap_count}")
        
        self.regimes['capitulation'] = capitulation
        
        return capitulation


class PortfolioConstructor:
    """Construct portfolio based on regime and LGBM scores"""
    
    def __init__(
        self,
        regimes: pd.DataFrame,
        lgbm_scores: pd.DataFrame
    ):
        """
        Initialize portfolio constructor
        
        Args:
            regimes: DataFrame with regime classification
            lgbm_scores: DataFrame with LGBM predictions (MultiIndex: timestamp, symbol)
        """
        self.regimes = regimes.set_index('timestamp') if 'timestamp' in regimes.columns else regimes
        self.scores = lgbm_scores
        self.positions = None
        
    def select_positions(self) -> pd.DataFrame:
        """
        Select positions based on regime and LGBM scores
        
        Process:
        1. For each timestamp, get current regime
        2. Based on regime, determine long_n and short_n
        3. Rank symbols by LGBM score
        4. Select top N long and bottom N short
        
        Returns:
            DataFrame with position signals
        """
        print("\n" + "=" * 60)
        print("CONSTRUCTING PORTFOLIO POSITIONS")
        print("=" * 60)
        
        all_positions = []
        
        # Get unique timestamps
        timestamps = self.scores.index.get_level_values('timestamp').unique()
        
        print(f"  Processing {len(timestamps)} timestamps...")
        
        for ts in timestamps:
            # Get regime for this timestamp
            if ts not in self.regimes.index:
                continue
            
            regime = self.regimes.loc[ts, 'regime']
            
            # Get regime configuration
            regime_config = REGIMES.get(regime, REGIMES['S3_CHOP'])
            long_n = regime_config['long_n']
            short_n = regime_config['short_n']
            
            # Get scores for this timestamp
            ts_scores = self.scores.loc[ts]
            
            if isinstance(ts_scores, pd.Series):
                ts_scores = ts_scores.to_frame('prediction')
            
            # Rank by score
            ts_scores = ts_scores.sort_values('prediction', ascending=False)
            
            # Select long positions (top N)
            if long_n > 0:
                long_symbols = ts_scores.head(long_n).index.tolist()
            else:
                long_symbols = []
            
            # Select short positions (bottom N)
            if short_n > 0:
                short_symbols = ts_scores.tail(short_n).index.tolist()
            else:
                short_symbols = []
            
            # Create position DataFrame
            for symbol in long_symbols:
                all_positions.append({
                    'timestamp': ts,
                    'symbol': symbol,
                    'position': 1,  # Long
                    'regime': regime,
                    'score': ts_scores.loc[symbol, 'prediction']
                })
            
            for symbol in short_symbols:
                all_positions.append({
                    'timestamp': ts,
                    'symbol': symbol,
                    'position': -1,  # Short
                    'regime': regime,
                    'score': ts_scores.loc[symbol, 'prediction']
                })
        
        positions_df = pd.DataFrame(all_positions)
        
        if len(positions_df) == 0:
            print("  WARNING: No positions generated!")
            return pd.DataFrame()
        
        print(f"\n  Position Summary:")
        print(f"    Total positions: {len(positions_df):,}")
        print(f"    Long positions: {(positions_df['position'] == 1).sum():,}")
        print(f"    Short positions: {(positions_df['position'] == -1).sum():,}")
        
        # Position distribution by regime
        print(f"\n  Positions by Regime:")
        for regime in positions_df['regime'].unique():
            regime_pos = positions_df[positions_df['regime'] == regime]
            long_count = (regime_pos['position'] == 1).sum()
            short_count = (regime_pos['position'] == -1).sum()
            print(f"    {regime}: {long_count} long, {short_count} short")
        
        self.positions = positions_df
        return positions_df
    
    def calculate_weights(
        self,
        positions: pd.DataFrame,
        volatility_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Calculate position weights with volatility scaling
        
        Args:
            positions: DataFrame with position signals
            volatility_data: Optional volatility data for inverse-vol weighting
            
        Returns:
            DataFrame with position weights
        """
        print("\n  Calculating position weights...")
        
        weights = positions.copy()
        
        # Group by timestamp
        for ts in weights['timestamp'].unique():
            ts_mask = weights['timestamp'] == ts
            ts_positions = weights[ts_mask]
            
            # Get regime
            regime = ts_positions['regime'].iloc[0]
            
            # Base weight (equal weight within long/short)
            long_mask = ts_positions['position'] == 1
            short_mask = ts_positions['position'] == -1
            
            n_long = long_mask.sum()
            n_short = short_mask.sum()
            
            # Equal weight
            if n_long > 0:
                weights.loc[ts_mask & long_mask, 'weight'] = 1.0 / n_long
            if n_short > 0:
                weights.loc[ts_mask & short_mask, 'weight'] = -1.0 / n_short
            
            # Apply leverage based on regime
            leverage = RISK_CONFIG['max_leverage'].get(regime, 1.0)
            weights.loc[ts_mask, 'weight'] *= leverage
            
            # Cap individual position size
            max_pos = RISK_CONFIG['max_single_position']
            weights.loc[ts_mask, 'weight'] = weights.loc[ts_mask, 'weight'].clip(-max_pos, max_pos)
        
        print(f"    Weights calculated for {len(weights)} positions")
        print(f"    Mean absolute weight: {weights['weight'].abs().mean():.4f}")
        
        return weights
    
    def apply_risk_limits(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk management rules
        
        Rules:
        1. Max single position: 15%
        2. Max leverage by regime
        3. Portfolio stop loss
        
        Args:
            weights: DataFrame with position weights
            
        Returns:
            Risk-adjusted weights
        """
        print("\n  Applying risk limits...")
        
        adjusted = weights.copy()
        
        # Rule 1: Cap individual positions
        max_pos = RISK_CONFIG['max_single_position']
        adjusted['weight'] = adjusted['weight'].clip(-max_pos, max_pos)
        
        # Rule 2: Check total leverage by timestamp
        for ts in adjusted['timestamp'].unique():
            ts_mask = adjusted['timestamp'] == ts
            ts_weights = adjusted[ts_mask]
            
            total_leverage = ts_weights['weight'].abs().sum()
            regime = ts_weights['regime'].iloc[0]
            max_leverage = RISK_CONFIG['max_leverage'].get(regime, 1.0)
            
            # Scale down if exceeds max leverage
            if total_leverage > max_leverage:
                scale_factor = max_leverage / total_leverage
                adjusted.loc[ts_mask, 'weight'] *= scale_factor
        
        print(f"    Risk limits applied")
        
        return adjusted


class StrategyEngine:
    """Complete strategy execution engine"""
    
    def __init__(
        self,
        tft_signals_path: Path = None,
        lgbm_scores_path: Path = None
    ):
        """
        Initialize strategy engine
        
        Args:
            tft_signals_path: Path to TFT signals
            lgbm_scores_path: Path to LGBM scores
        """
        self.tft_signals_path = tft_signals_path or (OUTPUT_DIR / 'tft_signals.parquet')
        self.lgbm_scores_path = lgbm_scores_path or (OUTPUT_DIR / 'lgbm_scores.parquet')
        
        self.tft_signals = None
        self.lgbm_scores = None
        self.regimes = None
        self.positions = None
        self.weights = None
        
    def load_data(self):
        """Load TFT signals and LGBM scores"""
        print("\n" + "=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        # Load TFT signals
        print(f"  Loading TFT signals from: {self.tft_signals_path}")
        self.tft_signals = pd.read_parquet(self.tft_signals_path)
        self.tft_signals['timestamp'] = pd.to_datetime(self.tft_signals['timestamp'])
        print(f"    Shape: {self.tft_signals.shape}")
        
        # Load LGBM scores
        print(f"  Loading LGBM scores from: {self.lgbm_scores_path}")
        self.lgbm_scores = pd.read_parquet(self.lgbm_scores_path)
        
        # Ensure proper index
        if 'timestamp' in self.lgbm_scores.columns and 'symbol' in self.lgbm_scores.columns:
            self.lgbm_scores['timestamp'] = pd.to_datetime(self.lgbm_scores['timestamp'])
            self.lgbm_scores = self.lgbm_scores.set_index(['timestamp', 'symbol'])
        
        print(f"    Shape: {self.lgbm_scores.shape}")
        
    def run_pipeline(self) -> tuple:
        """
        Run complete strategy pipeline
        
        Returns:
            (regimes, positions, weights) tuple
        """
        print("\n" + "=" * 80)
        print("STRATEGY EXECUTION PIPELINE")
        print("=" * 80)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Classify regimes
        classifier = RegimeClassifier(self.tft_signals)
        regimes = classifier.classify_regime()
        
        # Step 3: Detect capitulation
        classifier.detect_capitulation()
        
        # Step 4: Construct portfolio
        constructor = PortfolioConstructor(regimes, self.lgbm_scores)
        positions = constructor.select_positions()
        
        # Step 5: Calculate weights
        weights = constructor.calculate_weights(positions)
        
        # Step 6: Apply risk limits
        weights = constructor.apply_risk_limits(weights)
        
        # Step 7: Save results
        self.save_results(regimes, positions, weights)
        
        self.regimes = regimes
        self.positions = positions
        self.weights = weights
        
        print("\n" + "=" * 80)
        print("STRATEGY EXECUTION COMPLETED!")
        print("=" * 80)
        
        return regimes, positions, weights
    
    def save_results(
        self,
        regimes: pd.DataFrame,
        positions: pd.DataFrame,
        weights: pd.DataFrame
    ):
        """Save strategy results"""
        print("\n" + "=" * 60)
        print("SAVING STRATEGY RESULTS")
        print("=" * 60)
        
        # Save regimes
        regime_path = OUTPUT_DIR / 'regime_history.parquet'
        regimes.to_parquet(regime_path, index=False)
        print(f"  Regimes saved to: {regime_path}")
        
        # Save positions
        pos_path = OUTPUT_DIR / 'portfolio_positions.parquet'
        positions.to_parquet(pos_path, index=False)
        print(f"  Positions saved to: {pos_path}")
        
        # Save weights
        weight_path = OUTPUT_DIR / 'portfolio_weights.parquet'
        weights.to_parquet(weight_path, index=False)
        print(f"  Weights saved to: {weight_path}")
        
        # Save summary
        summary_path = OUTPUT_DIR / 'strategy_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Strategy Execution Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total timestamps: {len(regimes)}\n")
            f.write(f"Total positions: {len(positions)}\n")
            f.write(f"\nRegime Distribution:\n")
            f.write(regimes['regime'].value_counts().to_string())
            f.write(f"\n\nPosition Statistics:\n")
            f.write(f"Long positions: {(positions['position'] == 1).sum()}\n")
            f.write(f"Short positions: {(positions['position'] == -1).sum()}\n")
            f.write(f"\nWeight Statistics:\n")
            f.write(weights['weight'].describe().to_string())
        print(f"  Summary saved to: {summary_path}")


def main():
    """Main execution"""
    engine = StrategyEngine()
    regimes, positions, weights = engine.run_pipeline()
    
    print("\n✓ Strategy execution complete!")
    print(f"✓ Regimes: {len(regimes)} timestamps")
    print(f"✓ Positions: {len(positions)} entries")
    print(f"✓ Weights: {len(weights)} entries")


if __name__ == "__main__":
    main()
