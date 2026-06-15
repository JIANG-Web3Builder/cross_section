"""
Main Execution Script for ETH/BTC Statistical Arbitrage Strategy

This script orchestrates the entire workflow:
1. Load and prepare data
2. Test cointegration
3. Calculate dynamic hedge ratios
4. Generate trading signals
5. Run backtest
6. Analyze performance
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import config
from data_loader import DataLoader
from cointegration import CointegrationAnalyzer
from hedge_ratio import HedgeRatioCalculator, SpreadAnalyzer
from signal_generator import SignalGenerator, PositionSizer
from backtester import Backtester
from performance_analytics import PerformanceAnalyzer

def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print("ETH/BTC STATISTICAL ARBITRAGE STRATEGY")
    print("Institutional-Grade Mean Reversion Trading System")
    print("="*70)
    
    # ========================================================================
    # STAGE 1: DATA LOADING
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 1: DATA LOADING")
    print("="*70)
    
    loader = DataLoader(
        data_dir=config.DATA_DIR,
        asset_y=config.ASSET_Y,
        asset_x=config.ASSET_X,
        timeframe=config.TIMEFRAME
    )
    
    df = loader.prepare_full_dataset(
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )
    
    # ========================================================================
    # STAGE 2: COINTEGRATION TESTING
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 2: COINTEGRATION TESTING")
    print("="*70)
    
    coint_analyzer = CointegrationAnalyzer(
        significance_level=config.ADF_SIGNIFICANCE_LEVEL
    )
    
    # Test stationarity of individual series
    print("\nTesting stationarity of individual price series...")
    eth_stationarity = coint_analyzer.test_stationarity(df['log_eth'], "log(ETH)")
    btc_stationarity = coint_analyzer.test_stationarity(df['log_btc'], "log(BTC)")
    
    # Engle-Granger cointegration test
    coint_results = coint_analyzer.engle_granger_two_step(
        df['log_eth'], 
        df['log_btc']
    )
    
    if not coint_results['is_cointegrated']:
        print("\n" + "!"*70)
        print("WARNING: No cointegration detected!")
        print("Strategy may not be viable. Proceeding with caution...")
        print("!"*70)
        
        response = input("\nContinue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting...")
            return
    
    # Plot cointegration analysis
    output_path = os.path.join(config.OUTPUT_DIR, 'cointegration_analysis.png')
    coint_analyzer.plot_cointegration_analysis(
        df['log_eth'], 
        df['log_btc'], 
        coint_results,
        output_path=output_path
    )
    
    # ========================================================================
    # STAGE 3: DYNAMIC HEDGE RATIO CALCULATION
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 3: DYNAMIC HEDGE RATIO CALCULATION")
    print("="*70)
    
    method = 'kalman' if config.USE_KALMAN else 'rolling_ols'
    hedge_calculator = HedgeRatioCalculator(
        method=method,
        window=config.ROLLING_WINDOW
    )
    
    if config.USE_KALMAN:
        beta, alpha, spread = hedge_calculator.calculate(
            df['log_eth'], 
            df['log_btc'],
            transition_cov=config.KALMAN_TRANSITION_COV,
            observation_cov=config.KALMAN_OBSERVATION_COV
        )
    else:
        beta, alpha, spread = hedge_calculator.calculate(
            df['log_eth'], 
            df['log_btc']
        )
    
    # Calculate half-life
    print("\nCalculating mean reversion characteristics...")
    half_life = SpreadAnalyzer.calculate_half_life(spread)
    print(f"Overall Half-Life: {half_life:.2f} hours")
    
    if half_life < config.HALFLIFE_MIN:
        print(f"⚠ Half-life too short ({half_life:.2f}h < {config.HALFLIFE_MIN}h) - May have execution issues")
    elif half_life > config.HALFLIFE_MAX:
        print(f"⚠ Half-life too long ({half_life:.2f}h > {config.HALFLIFE_MAX}h) - Slow mean reversion")
    else:
        print(f"✓ Half-life in acceptable range")
    
    # Calculate rolling half-life for regime detection
    rolling_hl = SpreadAnalyzer.rolling_half_life(spread, window=config.HALFLIFE_WINDOW)
    
    # Calculate Hurst exponent
    hurst = SpreadAnalyzer.hurst_exponent(spread)
    print(f"Hurst Exponent: {hurst:.4f}")
    if hurst < 0.5:
        print("✓ Mean reverting behavior confirmed (H < 0.5)")
    elif hurst > 0.5:
        print("⚠ Trending behavior detected (H > 0.5)")
    else:
        print("⚠ Random walk behavior (H ≈ 0.5)")
    
    # ========================================================================
    # STAGE 4: SIGNAL GENERATION
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 4: SIGNAL GENERATION")
    print("="*70)
    
    signal_gen = SignalGenerator(
        z_entry=config.ZSCORE_ENTRY,
        z_exit=config.ZSCORE_EXIT,
        z_stop=config.ZSCORE_STOP,
        zscore_window=config.ZSCORE_WINDOW,
        halflife_min=config.HALFLIFE_MIN,
        halflife_max=config.HALFLIFE_MAX
    )
    
    # Calculate Z-score
    zscore = signal_gen.calculate_zscore(spread)
    
    # Generate signals
    signals = signal_gen.generate_signals(zscore, half_life=rolling_hl)
    
    # Calculate signal quality metrics
    signal_metrics = signal_gen.calculate_signal_quality_metrics(signals)
    print(f"\nSignal Quality Metrics:")
    print(f"  Total Signals: {signal_metrics['total_signals']}")
    print(f"  Entry Signals: {signal_metrics['entry_signals']}")
    print(f"  Avg Holding Period: {signal_metrics['avg_holding_period_hours']:.2f} hours")
    print(f"  Time in Market: {signal_metrics['time_in_market_pct']:.2f}%")
    print(f"  Tradeable Regime: {signal_metrics['tradeable_regime_pct']:.2f}%")
    
    # ========================================================================
    # STAGE 5: BACKTESTING
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 5: BACKTESTING")
    print("="*70)
    
    backtester = Backtester(
        initial_capital=config.INITIAL_CAPITAL,
        maker_fee=config.MAKER_FEE,
        taker_fee=config.TAKER_FEE,
        slippage_bps=config.SLIPPAGE_BPS,
        funding_interval_hours=config.FUNDING_INTERVAL_HOURS
    )
    
    # Align data for backtest
    common_index = df.index.intersection(signals.index).intersection(beta.index)
    df_backtest = df.loc[common_index]
    signals_backtest = signals.loc[common_index]
    beta_backtest = beta.loc[common_index]
    
    # Run backtest
    results = backtester.run_backtest(
        df=df_backtest,
        signals=signals_backtest,
        beta=beta_backtest,
        position_size=config.INITIAL_CAPITAL * config.MAX_POSITION_SIZE,
        use_funding=True
    )
    
    # ========================================================================
    # STAGE 6: PERFORMANCE ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 6: PERFORMANCE ANALYSIS")
    print("="*70)
    
    analyzer = PerformanceAnalyzer(
        portfolio_values=backtester.portfolio_values,
        trades=backtester.trades
    )
    
    # Print detailed report
    analyzer.print_performance_report()
    
    # Generate visualizations
    print("\nGenerating performance visualizations...")
    
    # Performance dashboard
    perf_plot_path = os.path.join(config.OUTPUT_DIR, 'performance_dashboard.png')
    analyzer.plot_performance(output_path=perf_plot_path)
    
    # Trade analysis
    if len(backtester.trades) > 0:
        trade_plot_path = os.path.join(config.OUTPUT_DIR, 'trade_analysis.png')
        analyzer.plot_trade_analysis(output_path=trade_plot_path)
    
    # Additional strategy-specific plots
    print("\nGenerating strategy analysis plots...")
    plot_strategy_analysis(df_backtest, spread, zscore, beta, signals_backtest, rolling_hl)
    
    # ========================================================================
    # STAGE 7: EXPORT RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 7: EXPORTING RESULTS")
    print("="*70)
    
    # Export trades
    if len(backtester.trades) > 0:
        trades_path = os.path.join(config.OUTPUT_DIR, 'trades.csv')
        backtester.trades.to_csv(trades_path, index=False)
        print(f"Trades exported to: {trades_path}")
    
    # Export portfolio values
    portfolio_path = os.path.join(config.OUTPUT_DIR, 'portfolio_values.csv')
    backtester.portfolio_values.to_csv(portfolio_path)
    print(f"Portfolio values exported to: {portfolio_path}")
    
    # Export signals
    signals_path = os.path.join(config.OUTPUT_DIR, 'signals.csv')
    signals_export = signals_backtest.copy()
    signals_export['zscore'] = zscore.loc[common_index]
    signals_export['spread'] = spread.loc[common_index]
    signals_export['beta'] = beta_backtest
    signals_export.to_csv(signals_path)
    print(f"Signals exported to: {signals_path}")
    
    # Export summary metrics
    metrics = analyzer.calculate_metrics()
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(config.OUTPUT_DIR, 'performance_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Performance metrics exported to: {metrics_path}")
    
    print("\n" + "="*70)
    print("STRATEGY EXECUTION COMPLETED")
    print("="*70)
    print(f"\nAll results saved to: {config.OUTPUT_DIR}")
    
    return {
        'df': df_backtest,
        'signals': signals_backtest,
        'beta': beta_backtest,
        'spread': spread,
        'zscore': zscore,
        'results': results,
        'trades': backtester.trades,
        'metrics': metrics
    }


def plot_strategy_analysis(df, spread, zscore, beta, signals, half_life):
    """
    Create strategy-specific analysis plots
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    fig.suptitle('Statistical Arbitrage Strategy Analysis', fontsize=16, fontweight='bold')
    
    # 1. Spread and Z-score
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(spread.index, spread, label='Spread', color='blue', alpha=0.7, linewidth=1)
    ax1_twin.plot(zscore.index, zscore, label='Z-Score', color='red', alpha=0.7, linewidth=1)
    ax1_twin.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Entry Threshold')
    ax1_twin.axhline(y=-2, color='green', linestyle='--', alpha=0.5)
    ax1_twin.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Stop Loss')
    ax1_twin.axhline(y=-4, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Spread', color='blue')
    ax1_twin.set_ylabel('Z-Score', color='red')
    ax1.set_title('Spread and Z-Score')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Beta evolution
    ax2 = axes[1]
    ax2.plot(beta.index, beta, color='purple', linewidth=2)
    ax2.set_ylabel('Beta (Hedge Ratio)')
    ax2.set_title('Dynamic Hedge Ratio (Beta)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Positions and Z-score
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    ax3.fill_between(signals.index, 0, signals['position'], 
                     where=(signals['position'] > 0), color='green', alpha=0.3, label='Long Spread')
    ax3.fill_between(signals.index, 0, signals['position'], 
                     where=(signals['position'] < 0), color='red', alpha=0.3, label='Short Spread')
    ax3_twin.plot(zscore.index, zscore, color='black', alpha=0.5, linewidth=1, label='Z-Score')
    ax3.set_ylabel('Position', color='black')
    ax3_twin.set_ylabel('Z-Score', color='gray')
    ax3.set_title('Trading Positions and Z-Score')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Half-life evolution
    ax4 = axes[3]
    if half_life is not None and len(half_life) > 0:
        ax4.plot(half_life.index, half_life, color='orange', linewidth=2)
        ax4.axhline(y=24, color='green', linestyle='--', alpha=0.5, label='1 day')
        ax4.axhline(y=168, color='blue', linestyle='--', alpha=0.5, label='1 week')
        ax4.set_ylabel('Half-Life (hours)')
        ax4.set_title('Mean Reversion Half-Life')
        ax4.set_ylim(0, min(half_life.quantile(0.95) * 1.2, 500))
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(config.OUTPUT_DIR, 'strategy_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Strategy analysis plot saved to {output_path}")
    
    plt.close()


if __name__ == "__main__":
    try:
        results = main()
        print("\n✓ Strategy execution completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
