"""
Professional Backtesting Engine for Statistical Arbitrage
Includes realistic transaction costs, funding rates, and slippage
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class Backtester:
    """
    Institutional-grade backtesting engine for pairs trading
    
    Key Features:
    - Realistic transaction costs (maker/taker fees)
    - Funding rate simulation
    - Slippage modeling
    - Position tracking with beta-hedged portfolios
    - Detailed trade logging
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 maker_fee: float = 0.0002,
                 taker_fee: float = 0.0004,
                 slippage_bps: float = 5.0,
                 funding_interval_hours: int = 8):
        """
        Args:
            initial_capital: Starting capital in USD
            maker_fee: Maker fee rate (0.0002 = 0.02%)
            taker_fee: Taker fee rate (0.0004 = 0.04%)
            slippage_bps: Slippage in basis points
            funding_interval_hours: Funding payment interval
        """
        self.initial_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps / 10000  # Convert to decimal
        self.funding_interval_hours = funding_interval_hours
        
        # Results storage
        self.trades = []
        self.portfolio_values = []
        self.positions = []
        
    def simulate_funding_rate(self, timestamp: pd.Timestamp, 
                              base_rate: float = 0.0001,
                              volatility: float = 0.0002) -> float:
        """
        Simulate realistic funding rates
        In reality, you would use historical funding rate data
        """
        # Simple random walk around base rate
        noise = np.random.normal(0, volatility)
        funding_rate = base_rate + noise
        
        # Clamp to realistic range [-0.05%, +0.05%]
        return np.clip(funding_rate, -0.0005, 0.0005)
    
    def calculate_transaction_cost(self, notional: float, is_maker: bool = False) -> float:
        """Calculate transaction cost including fees and slippage"""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        total_cost = notional * (fee_rate + self.slippage_bps)
        return total_cost
    
    def execute_trade(self, timestamp: pd.Timestamp,
                     signal: int,
                     eth_price: float,
                     btc_price: float,
                     beta: float,
                     position_size: float) -> Dict:
        """
        Execute a pairs trade
        
        Signal = 1: Long Spread (Long ETH, Short BTC)
        Signal = -1: Short Spread (Short ETH, Long BTC)
        Signal = 0: Close position
        
        Args:
            position_size: Total USD value to trade
        """
        if signal == 0:
            return None
        
        # Calculate quantities
        # For beta-hedged portfolio: Long $X of ETH, Short $beta*X of BTC
        eth_notional = position_size * signal
        btc_notional = -position_size * beta * signal
        
        eth_quantity = eth_notional / eth_price
        btc_quantity = btc_notional / btc_price
        
        # Calculate costs (assume taker for simplicity, can be improved)
        eth_cost = self.calculate_transaction_cost(abs(eth_notional), is_maker=False)
        btc_cost = self.calculate_transaction_cost(abs(btc_notional), is_maker=False)
        total_cost = eth_cost + btc_cost
        
        trade = {
            'timestamp': timestamp,
            'signal': signal,
            'eth_price': eth_price,
            'btc_price': btc_price,
            'beta': beta,
            'eth_quantity': eth_quantity,
            'btc_quantity': btc_quantity,
            'eth_notional': eth_notional,
            'btc_notional': btc_notional,
            'transaction_cost': total_cost
        }
        
        return trade
    
    def calculate_pnl(self, 
                     entry_trade: Dict,
                     exit_trade: Dict,
                     funding_costs: float = 0) -> Dict:
        """
        Calculate P&L for a round-trip trade
        """
        # ETH P&L
        eth_pnl = (exit_trade['eth_price'] - entry_trade['eth_price']) * entry_trade['eth_quantity']
        
        # BTC P&L (note: BTC position is opposite)
        btc_pnl = (exit_trade['btc_price'] - entry_trade['btc_price']) * entry_trade['btc_quantity']
        
        # Total P&L
        gross_pnl = eth_pnl + btc_pnl
        net_pnl = gross_pnl - entry_trade['transaction_cost'] - exit_trade['transaction_cost'] - funding_costs
        
        return {
            'entry_time': entry_trade['timestamp'],
            'exit_time': exit_trade['timestamp'],
            'holding_period': (exit_trade['timestamp'] - entry_trade['timestamp']).total_seconds() / 3600,
            'eth_pnl': eth_pnl,
            'btc_pnl': btc_pnl,
            'gross_pnl': gross_pnl,
            'transaction_costs': entry_trade['transaction_cost'] + exit_trade['transaction_cost'],
            'funding_costs': funding_costs,
            'net_pnl': net_pnl,
            'return_pct': net_pnl / abs(entry_trade['eth_notional']) * 100
        }
    
    def run_backtest(self, 
                    df: pd.DataFrame,
                    signals: pd.DataFrame,
                    beta: pd.Series,
                    position_size: float = None,
                    use_funding: bool = True) -> pd.DataFrame:
        """
        Run full backtest simulation
        
        Args:
            df: DataFrame with eth_close, btc_close
            signals: DataFrame with signal column
            beta: Series with hedge ratios
            position_size: Fixed position size (if None, use full capital)
            use_funding: Whether to simulate funding costs
        """
        print("\n" + "="*60)
        print("RUNNING BACKTEST")
        print("="*60)
        
        # Align all data
        common_index = df.index.intersection(signals.index).intersection(beta.index)
        df = df.loc[common_index]
        signals = signals.loc[common_index]
        beta = beta.loc[common_index]
        
        # Initialize
        capital = self.initial_capital
        if position_size is None:
            position_size = capital
        
        portfolio_values = [capital]
        timestamps = [df.index[0]]
        
        current_position = 0
        entry_trade = None
        completed_trades = []
        
        # Track positions for funding calculation
        eth_position = 0
        btc_position = 0
        hours_since_funding = 0
        
        for i, idx in enumerate(df.index):
            signal = signals.loc[idx, 'signal']
            eth_price = df.loc[idx, 'eth_close']
            btc_price = df.loc[idx, 'btc_close']
            current_beta = beta.loc[idx]
            
            # Execute trade if signal changes
            if signal != 0 and current_position == 0:
                # Entry
                entry_trade = self.execute_trade(
                    idx, signal, eth_price, btc_price, current_beta, position_size
                )
                current_position = signal
                eth_position = entry_trade['eth_quantity']
                btc_position = entry_trade['btc_quantity']
                capital -= entry_trade['transaction_cost']
                
            elif signal != 0 and current_position != 0 and signal != current_position:
                # Exit current and enter new (rare but possible)
                exit_trade = self.execute_trade(
                    idx, -current_position, eth_price, btc_price, current_beta, position_size
                )
                
                # Calculate funding costs
                funding_cost = 0
                if use_funding and entry_trade is not None:
                    hours_held = (idx - entry_trade['timestamp']).total_seconds() / 3600
                    n_funding_payments = int(hours_held / self.funding_interval_hours)
                    
                    for _ in range(n_funding_payments):
                        eth_funding_rate = self.simulate_funding_rate(idx)
                        btc_funding_rate = self.simulate_funding_rate(idx)
                        
                        # Funding cost = position_value * funding_rate
                        # Long pays funding if rate > 0, short receives
                        eth_funding = -eth_position * eth_price * eth_funding_rate
                        btc_funding = -btc_position * btc_price * btc_funding_rate
                        funding_cost += eth_funding + btc_funding
                
                # Record completed trade
                pnl_result = self.calculate_pnl(entry_trade, exit_trade, funding_cost)
                completed_trades.append(pnl_result)
                capital += pnl_result['net_pnl']
                
                # Enter new position
                entry_trade = self.execute_trade(
                    idx, signal, eth_price, btc_price, current_beta, position_size
                )
                current_position = signal
                eth_position = entry_trade['eth_quantity']
                btc_position = entry_trade['btc_quantity']
                capital -= entry_trade['transaction_cost']
                
            elif signal == 0 and current_position != 0:
                # Exit
                exit_trade = self.execute_trade(
                    idx, -current_position, eth_price, btc_price, current_beta, position_size
                )
                
                # Calculate funding costs
                funding_cost = 0
                if use_funding and entry_trade is not None:
                    hours_held = (idx - entry_trade['timestamp']).total_seconds() / 3600
                    n_funding_payments = int(hours_held / self.funding_interval_hours)
                    
                    for _ in range(n_funding_payments):
                        eth_funding_rate = self.simulate_funding_rate(idx)
                        btc_funding_rate = self.simulate_funding_rate(idx)
                        eth_funding = -eth_position * eth_price * eth_funding_rate
                        btc_funding = -btc_position * btc_price * btc_funding_rate
                        funding_cost += eth_funding + btc_funding
                
                # Record completed trade
                pnl_result = self.calculate_pnl(entry_trade, exit_trade, funding_cost)
                completed_trades.append(pnl_result)
                capital += pnl_result['net_pnl']
                
                current_position = 0
                eth_position = 0
                btc_position = 0
                entry_trade = None
            
            # Calculate mark-to-market portfolio value
            if current_position != 0 and entry_trade is not None:
                # Unrealized P&L
                eth_unrealized = (eth_price - entry_trade['eth_price']) * eth_position
                btc_unrealized = (btc_price - entry_trade['btc_price']) * btc_position
                portfolio_value = capital + eth_unrealized + btc_unrealized
            else:
                portfolio_value = capital
            
            portfolio_values.append(portfolio_value)
            timestamps.append(idx)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'returns': pd.Series(portfolio_values).pct_change(),
            'drawdown': 0.0
        }, index=timestamps)
        
        # Calculate drawdown
        cummax = results_df['portfolio_value'].cummax()
        results_df['drawdown'] = (results_df['portfolio_value'] - cummax) / cummax
        
        # Store results
        self.portfolio_values = results_df
        self.trades = pd.DataFrame(completed_trades) if completed_trades else pd.DataFrame()
        
        # Print summary
        self._print_backtest_summary()
        
        return results_df
    
    def _print_backtest_summary(self):
        """Print backtest performance summary"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        if len(self.portfolio_values) == 0:
            print("No trades executed")
            return
        
        final_value = self.portfolio_values['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        print(f"\nPortfolio Performance:")
        print(f"  Initial Capital:    ${self.initial_capital:,.2f}")
        print(f"  Final Value:        ${final_value:,.2f}")
        print(f"  Total Return:       {total_return:.2f}%")
        print(f"  Max Drawdown:       {self.portfolio_values['drawdown'].min()*100:.2f}%")
        
        if len(self.trades) > 0:
            print(f"\nTrade Statistics:")
            print(f"  Total Trades:       {len(self.trades)}")
            print(f"  Winning Trades:     {(self.trades['net_pnl'] > 0).sum()}")
            print(f"  Losing Trades:      {(self.trades['net_pnl'] < 0).sum()}")
            print(f"  Win Rate:           {(self.trades['net_pnl'] > 0).mean()*100:.2f}%")
            print(f"  Avg Win:            ${self.trades[self.trades['net_pnl'] > 0]['net_pnl'].mean():.2f}")
            print(f"  Avg Loss:           ${self.trades[self.trades['net_pnl'] < 0]['net_pnl'].mean():.2f}")
            print(f"  Avg Holding:        {self.trades['holding_period'].mean():.2f} hours")
            print(f"  Total Txn Costs:    ${self.trades['transaction_costs'].sum():.2f}")
            print(f"  Total Funding:      ${self.trades['funding_costs'].sum():.2f}")
        
        # Calculate Sharpe ratio (annualized)
        returns = self.portfolio_values['returns'].dropna()
        if len(returns) > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24)  # Hourly to annual
            print(f"\nRisk Metrics:")
            print(f"  Sharpe Ratio:       {sharpe:.2f}")
            print(f"  Volatility (ann):   {returns.std() * np.sqrt(365*24) * 100:.2f}%")
