"""
Module 5: Backtest Engine
Complete backtesting framework with performance analysis

Key Features:
- Event-driven backtesting with realistic execution
- Transaction cost modeling (fees + slippage)
- Risk management (stop loss, position limits)
- Comprehensive performance metrics
- Regime-based analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config_tft import (
    OUTPUT_DIR, BACKTEST_CONFIG, RISK_CONFIG,
    LGBM_REFERENCE_DIR
)


class BacktestEngine:
    """Event-driven backtest engine"""
    
    def __init__(
        self,
        weights_path: Path = None,
        regimes_path: Path = None,
        price_data_path: Path = None
    ):
        """
        Initialize backtest engine
        
        Args:
            weights_path: Path to portfolio weights
            regimes_path: Path to regime history
            price_data_path: Path to price data (from LGBM system)
        """
        self.weights_path = weights_path or (OUTPUT_DIR / 'portfolio_weights.parquet')
        self.regimes_path = regimes_path or (OUTPUT_DIR / 'regime_history.parquet')
        self.price_data_path = price_data_path or (LGBM_REFERENCE_DIR.parent / 'crypto_截面' / 'data' / 'cross_section_close_cleaned.parquet')
        
        self.weights = None
        self.regimes = None
        self.prices = None
        self.returns = None
        
        self.equity_curve = None
        self.trades = []
        self.positions = {}
        self.cash = BACKTEST_CONFIG['initial_capital']
        
    def load_data(self):
        """Load backtest data"""
        print("\n" + "=" * 60)
        print("LOADING BACKTEST DATA")
        print("=" * 60)
        
        # Load weights
        print(f"  Loading weights from: {self.weights_path}")
        self.weights = pd.read_parquet(self.weights_path)
        self.weights['timestamp'] = pd.to_datetime(self.weights['timestamp'])
        print(f"    Shape: {self.weights.shape}")
        
        # Load regimes
        print(f"  Loading regimes from: {self.regimes_path}")
        self.regimes = pd.read_parquet(self.regimes_path)
        self.regimes['timestamp'] = pd.to_datetime(self.regimes['timestamp'])
        self.regimes = self.regimes.set_index('timestamp')
        print(f"    Shape: {self.regimes.shape}")
        
        # Load prices
        print(f"  Loading prices from: {self.price_data_path}")
        self.prices = pd.read_parquet(self.price_data_path)
        print(f"    Shape: {self.prices.shape}")
        
        # Calculate returns
        self.returns = self.prices.pct_change()
        print(f"    Returns calculated")
        
    def calculate_transaction_costs(
        self,
        trade_value: float,
        is_maker: bool = True
    ) -> float:
        """
        Calculate transaction costs
        
        Args:
            trade_value: Absolute value of trade
            is_maker: If True, use maker fee; else taker fee
            
        Returns:
            Total transaction cost
        """
        fee_rate = BACKTEST_CONFIG['maker_fee'] if is_maker else BACKTEST_CONFIG['taker_fee']
        slippage_rate = BACKTEST_CONFIG['slippage']
        
        fee = trade_value * fee_rate
        slippage = trade_value * slippage_rate
        
        return fee + slippage
    
    def execute_rebalance(
        self,
        timestamp: pd.Timestamp,
        target_weights: pd.DataFrame,
        current_prices: pd.Series
    ) -> dict:
        """
        Execute portfolio rebalance
        
        Args:
            timestamp: Current timestamp
            target_weights: Target portfolio weights
            current_prices: Current prices for all symbols
            
        Returns:
            Dictionary with rebalance results
        """
        # Calculate current portfolio value
        portfolio_value = self.cash
        for symbol, shares in self.positions.items():
            if symbol in current_prices.index:
                portfolio_value += shares * current_prices[symbol]
        
        # Target positions
        target_positions = {}
        total_cost = 0
        
        for _, row in target_weights.iterrows():
            symbol = row['symbol']
            weight = row['weight']
            
            if symbol not in current_prices.index:
                continue
            
            price = current_prices[symbol]
            target_value = portfolio_value * weight
            target_shares = target_value / price
            
            target_positions[symbol] = target_shares
            
            # Calculate trade
            current_shares = self.positions.get(symbol, 0)
            trade_shares = target_shares - current_shares
            trade_value = abs(trade_shares * price)
            
            if trade_value >= BACKTEST_CONFIG['min_trade_size']:
                # Execute trade
                cost = self.calculate_transaction_costs(trade_value)
                total_cost += cost
                
                self.positions[symbol] = target_shares
                self.cash -= trade_shares * price + cost
                
                # Record trade
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'shares': trade_shares,
                    'price': price,
                    'value': trade_shares * price,
                    'cost': cost
                })
        
        # Close positions not in target
        for symbol in list(self.positions.keys()):
            if symbol not in target_positions:
                if symbol in current_prices.index:
                    shares = self.positions[symbol]
                    price = current_prices[symbol]
                    trade_value = abs(shares * price)
                    cost = self.calculate_transaction_costs(trade_value)
                    
                    self.cash += shares * price - cost
                    total_cost += cost
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'shares': -shares,
                        'price': price,
                        'value': -shares * price,
                        'cost': cost
                    })
                    
                    del self.positions[symbol]
        
        return {
            'portfolio_value': portfolio_value,
            'total_cost': total_cost,
            'n_positions': len(self.positions)
        }
    
    def check_stop_loss(
        self,
        current_value: float,
        peak_value: float
    ) -> bool:
        """
        Check if stop loss is triggered
        
        Args:
            current_value: Current portfolio value
            peak_value: Peak portfolio value
            
        Returns:
            True if stop loss triggered
        """
        drawdown = (peak_value - current_value) / peak_value
        
        if drawdown >= RISK_CONFIG['portfolio_stop_loss']:
            print(f"    STOP LOSS TRIGGERED! Drawdown: {drawdown:.2%}")
            return True
        
        return False
    
    def run_backtest(self) -> pd.DataFrame:
        """
        Run complete backtest
        
        Returns:
            DataFrame with equity curve
        """
        print("\n" + "=" * 80)
        print("RUNNING BACKTEST")
        print("=" * 80)
        
        # Get unique timestamps from weights
        timestamps = sorted(self.weights['timestamp'].unique())
        
        print(f"  Backtest period: {timestamps[0]} to {timestamps[-1]}")
        print(f"  Total timestamps: {len(timestamps)}")
        print(f"  Initial capital: ${BACKTEST_CONFIG['initial_capital']:,.0f}")
        
        equity_curve = []
        peak_value = BACKTEST_CONFIG['initial_capital']
        stop_loss_cooldown = None
        
        for i, ts in enumerate(timestamps):
            # Get current prices
            if ts not in self.prices.index:
                continue
            
            current_prices = self.prices.loc[ts]
            
            # Calculate current portfolio value
            portfolio_value = self.cash
            for symbol, shares in self.positions.items():
                if symbol in current_prices.index:
                    portfolio_value += shares * current_prices[symbol]
            
            # Check stop loss
            if self.check_stop_loss(portfolio_value, peak_value):
                # Liquidate all positions
                for symbol in list(self.positions.keys()):
                    if symbol in current_prices.index:
                        shares = self.positions[symbol]
                        price = current_prices[symbol]
                        self.cash += shares * price
                        del self.positions[symbol]
                
                stop_loss_cooldown = ts + pd.Timedelta(hours=RISK_CONFIG.get('cooldown_hours', 24))
                print(f"    Stop loss at {ts}, cooldown until {stop_loss_cooldown}")
                continue
            
            # Check cooldown
            if stop_loss_cooldown and ts < stop_loss_cooldown:
                equity_curve.append({
                    'timestamp': ts,
                    'portfolio_value': self.cash,
                    'cash': self.cash,
                    'n_positions': 0,
                    'regime': self.regimes.loc[ts, 'regime'] if ts in self.regimes.index else 'UNKNOWN'
                })
                continue
            
            # Get target weights for this timestamp
            ts_weights = self.weights[self.weights['timestamp'] == ts]
            
            if len(ts_weights) == 0:
                # No rebalance, just record current value
                equity_curve.append({
                    'timestamp': ts,
                    'portfolio_value': portfolio_value,
                    'cash': self.cash,
                    'n_positions': len(self.positions),
                    'regime': self.regimes.loc[ts, 'regime'] if ts in self.regimes.index else 'UNKNOWN'
                })
                continue
            
            # Execute rebalance
            rebalance_result = self.execute_rebalance(ts, ts_weights, current_prices)
            
            # Update peak value
            if rebalance_result['portfolio_value'] > peak_value:
                peak_value = rebalance_result['portfolio_value']
            
            # Record equity curve
            equity_curve.append({
                'timestamp': ts,
                'portfolio_value': rebalance_result['portfolio_value'],
                'cash': self.cash,
                'n_positions': rebalance_result['n_positions'],
                'transaction_cost': rebalance_result['total_cost'],
                'regime': self.regimes.loc[ts, 'regime'] if ts in self.regimes.index else 'UNKNOWN'
            })
            
            # Progress update
            if (i + 1) % 100 == 0:
                pct_complete = (i + 1) / len(timestamps) * 100
                print(f"    Progress: {pct_complete:.1f}% | Value: ${rebalance_result['portfolio_value']:,.0f} | Positions: {rebalance_result['n_positions']}")
        
        self.equity_curve = pd.DataFrame(equity_curve)
        
        print(f"\n  Backtest completed!")
        print(f"  Final portfolio value: ${self.equity_curve['portfolio_value'].iloc[-1]:,.0f}")
        print(f"  Total return: {(self.equity_curve['portfolio_value'].iloc[-1] / BACKTEST_CONFIG['initial_capital'] - 1) * 100:.2f}%")
        
        return self.equity_curve
    
    def calculate_metrics(self) -> dict:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        print("\n" + "=" * 60)
        print("CALCULATING PERFORMANCE METRICS")
        print("=" * 60)
        
        if self.equity_curve is None:
            raise ValueError("Must run backtest first")
        
        # Portfolio returns
        pv = self.equity_curve['portfolio_value']
        returns = pv.pct_change().dropna()
        
        # Basic metrics
        total_return = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
        
        # Annualized metrics (assuming hourly data)
        hours_per_year = 365 * 24
        n_hours = len(returns)
        years = n_hours / hours_per_year
        
        annualized_return = ((pv.iloc[-1] / pv.iloc[0]) ** (1 / years) - 1) * 100
        annualized_vol = returns.std() * np.sqrt(hours_per_year) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Drawdown
        cummax = pv.cummax()
        drawdown = (pv - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) * 100
        
        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Transaction costs
        total_costs = self.equity_curve['transaction_cost'].sum() if 'transaction_cost' in self.equity_curve.columns else 0
        cost_pct = total_costs / BACKTEST_CONFIG['initial_capital'] * 100
        
        # Regime-based performance
        regime_returns = {}
        for regime in self.equity_curve['regime'].unique():
            regime_mask = self.equity_curve['regime'] == regime
            regime_pv = self.equity_curve.loc[regime_mask, 'portfolio_value']
            if len(regime_pv) > 1:
                regime_ret = (regime_pv.iloc[-1] / regime_pv.iloc[0] - 1) * 100
                regime_returns[regime] = regime_ret
        
        metrics = {
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'annualized_volatility_pct': annualized_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate_pct': win_rate,
            'total_trades': len(self.trades),
            'total_transaction_costs': total_costs,
            'transaction_cost_pct': cost_pct,
            'regime_returns': regime_returns
        }
        
        print(f"\n  Performance Summary:")
        print(f"    Total Return: {total_return:.2f}%")
        print(f"    Annualized Return: {annualized_return:.2f}%")
        print(f"    Annualized Volatility: {annualized_vol:.2f}%")
        print(f"    Sharpe Ratio: {sharpe:.2f}")
        print(f"    Max Drawdown: {max_drawdown:.2f}%")
        print(f"    Calmar Ratio: {calmar:.2f}")
        print(f"    Win Rate: {win_rate:.2f}%")
        print(f"    Total Trades: {len(self.trades):,}")
        print(f"    Transaction Costs: ${total_costs:,.0f} ({cost_pct:.2f}%)")
        
        print(f"\n  Returns by Regime:")
        for regime, ret in regime_returns.items():
            print(f"    {regime}: {ret:.2f}%")
        
        return metrics
    
    def save_results(self, metrics: dict):
        """Save backtest results"""
        print("\n" + "=" * 60)
        print("SAVING BACKTEST RESULTS")
        print("=" * 60)
        
        # Save equity curve
        equity_path = OUTPUT_DIR / 'equity_curve.parquet'
        self.equity_curve.to_parquet(equity_path, index=False)
        print(f"  Equity curve saved to: {equity_path}")
        
        # Save equity curve CSV for easy viewing
        equity_csv_path = OUTPUT_DIR / 'equity_curve.csv'
        self.equity_curve.to_csv(equity_csv_path, index=False)
        print(f"  Equity curve CSV saved to: {equity_csv_path}")
        
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_path = OUTPUT_DIR / 'trade_log.parquet'
            trades_df.to_parquet(trades_path, index=False)
            print(f"  Trade log saved to: {trades_path}")
        
        # Save metrics
        metrics_path = OUTPUT_DIR / 'backtest_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("TFT-LGBM Backtest Results\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Performance Metrics:\n")
            f.write("-" * 60 + "\n")
            for key, value in metrics.items():
                if key != 'regime_returns':
                    f.write(f"{key}: {value}\n")
            
            f.write("\nReturns by Regime:\n")
            f.write("-" * 60 + "\n")
            for regime, ret in metrics['regime_returns'].items():
                f.write(f"{regime}: {ret:.2f}%\n")
            
            f.write("\nBacktest Configuration:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Initial Capital: ${BACKTEST_CONFIG['initial_capital']:,.0f}\n")
            f.write(f"Maker Fee: {BACKTEST_CONFIG['maker_fee']:.4f}\n")
            f.write(f"Taker Fee: {BACKTEST_CONFIG['taker_fee']:.4f}\n")
            f.write(f"Slippage: {BACKTEST_CONFIG['slippage']:.4f}\n")
        
        print(f"  Metrics saved to: {metrics_path}")
    
    def run_pipeline(self) -> tuple:
        """
        Run complete backtest pipeline
        
        Returns:
            (equity_curve, metrics) tuple
        """
        print("\n" + "=" * 80)
        print("BACKTEST PIPELINE")
        print("=" * 80)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Run backtest
        equity_curve = self.run_backtest()
        
        # Step 3: Calculate metrics
        metrics = self.calculate_metrics()
        
        # Step 4: Save results
        self.save_results(metrics)
        
        print("\n" + "=" * 80)
        print("BACKTEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return equity_curve, metrics


def main():
    """Main execution"""
    engine = BacktestEngine()
    equity_curve, metrics = engine.run_pipeline()
    
    print("\n✓ Backtest complete!")
    print(f"✓ Final Return: {metrics['total_return_pct']:.2f}%")
    print(f"✓ Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"✓ Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")


if __name__ == "__main__":
    main()
