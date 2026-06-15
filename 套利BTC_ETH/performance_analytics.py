"""
Performance Analytics and Visualization Module
Comprehensive analysis tools for strategy evaluation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    """
    Institutional-grade performance analytics
    """
    
    def __init__(self, portfolio_values: pd.DataFrame, trades: pd.DataFrame):
        self.portfolio_values = portfolio_values
        self.trades = trades
        
    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        pv = self.portfolio_values['portfolio_value']
        returns = self.portfolio_values['returns'].dropna()
        
        # Basic metrics
        initial_value = pv.iloc[0]
        final_value = pv.iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Time-based metrics
        start_date = pv.index[0]
        end_date = pv.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        
        # Annualized return
        cagr = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else 0
        
        # Risk metrics
        volatility_hourly = returns.std()
        volatility_annual = volatility_hourly * np.sqrt(365 * 24)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365 * 24) if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = returns.mean() / downside_std * np.sqrt(365 * 24) if downside_std > 0 else 0
        
        # Drawdown metrics
        cummax = pv.cummax()
        drawdown = (pv - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        if len(self.trades) > 0:
            n_trades = len(self.trades)
            winning_trades = (self.trades['net_pnl'] > 0).sum()
            losing_trades = (self.trades['net_pnl'] < 0).sum()
            win_rate = winning_trades / n_trades if n_trades > 0 else 0
            
            avg_win = self.trades[self.trades['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
            avg_loss = self.trades[self.trades['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0
            
            profit_factor = abs(self.trades[self.trades['net_pnl'] > 0]['net_pnl'].sum() / 
                               self.trades[self.trades['net_pnl'] < 0]['net_pnl'].sum()) if losing_trades > 0 else np.inf
            
            avg_holding_period = self.trades['holding_period'].mean()
            
            # Expectancy
            expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        else:
            n_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_holding_period = 0
            expectancy = 0
        
        metrics = {
            'total_return': total_return * 100,
            'cagr': cagr * 100,
            'volatility_annual': volatility_annual * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'n_trades': n_trades,
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding_period,
            'expectancy': expectancy,
            'start_date': start_date,
            'end_date': end_date,
            'days': days
        }
        
        return metrics
    
    def print_performance_report(self):
        """Print detailed performance report"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*70)
        print("PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n{'Period:':<30} {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')} ({metrics['days']} days)")
        
        print(f"\n{'RETURNS':-^70}")
        print(f"{'Total Return:':<30} {metrics['total_return']:>10.2f}%")
        print(f"{'CAGR:':<30} {metrics['cagr']:>10.2f}%")
        
        print(f"\n{'RISK METRICS':-^70}")
        print(f"{'Annual Volatility:':<30} {metrics['volatility_annual']:>10.2f}%")
        print(f"{'Sharpe Ratio:':<30} {metrics['sharpe_ratio']:>10.2f}")
        print(f"{'Sortino Ratio:':<30} {metrics['sortino_ratio']:>10.2f}")
        print(f"{'Max Drawdown:':<30} {metrics['max_drawdown']:>10.2f}%")
        print(f"{'Calmar Ratio:':<30} {metrics['calmar_ratio']:>10.2f}")
        
        print(f"\n{'TRADE STATISTICS':-^70}")
        print(f"{'Total Trades:':<30} {metrics['n_trades']:>10}")
        print(f"{'Win Rate:':<30} {metrics['win_rate']:>10.2f}%")
        print(f"{'Average Win:':<30} ${metrics['avg_win']:>10.2f}")
        print(f"{'Average Loss:':<30} ${metrics['avg_loss']:>10.2f}")
        print(f"{'Profit Factor:':<30} {metrics['profit_factor']:>10.2f}")
        print(f"{'Expectancy:':<30} ${metrics['expectancy']:>10.2f}")
        print(f"{'Avg Holding Period:':<30} {metrics['avg_holding_period']:>10.2f} hours")
        
        print("="*70 + "\n")
    
    def plot_performance(self, output_path: Optional[str] = None):
        """
        Create comprehensive performance visualization
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Portfolio Value
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.portfolio_values.index, self.portfolio_values['portfolio_value'], 
                linewidth=2, color='#2E86AB')
        ax1.fill_between(self.portfolio_values.index, 
                         self.portfolio_values['portfolio_value'].iloc[0],
                         self.portfolio_values['portfolio_value'],
                         alpha=0.3, color='#2E86AB')
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (USD)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        ax2.fill_between(self.portfolio_values.index, 
                         0, self.portfolio_values['drawdown'] * 100,
                         color='#A23B72', alpha=0.6)
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Returns Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        returns = self.portfolio_values['returns'].dropna() * 100
        ax3.hist(returns, bins=100, alpha=0.7, color='#18A558', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Returns
        ax4 = fig.add_subplot(gs[2, 1])
        cumulative_returns = (1 + self.portfolio_values['returns']).cumprod() - 1
        ax4.plot(cumulative_returns.index, cumulative_returns * 100, 
                linewidth=2, color='#F18F01')
        ax4.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Cumulative Return (%)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe Ratio
        ax5 = fig.add_subplot(gs[2, 2])
        rolling_sharpe = (self.portfolio_values['returns'].rolling(window=720).mean() / 
                         self.portfolio_values['returns'].rolling(window=720).std() * 
                         np.sqrt(365 * 24))
        ax5.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='#C73E1D')
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_title('Rolling Sharpe Ratio (30d)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.grid(True, alpha=0.3)
        
        if len(self.trades) > 0:
            # 6. Trade P&L Distribution
            ax6 = fig.add_subplot(gs[3, 0])
            ax6.hist(self.trades['net_pnl'], bins=50, alpha=0.7, 
                    color='#6A4C93', edgecolor='black')
            ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax6.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
            ax6.set_xlabel('P&L (USD)')
            ax6.set_ylabel('Frequency')
            ax6.grid(True, alpha=0.3)
            
            # 7. Cumulative Trade P&L
            ax7 = fig.add_subplot(gs[3, 1])
            cumulative_pnl = self.trades['net_pnl'].cumsum()
            ax7.plot(range(len(cumulative_pnl)), cumulative_pnl, 
                    linewidth=2, marker='o', markersize=3, color='#1982C4')
            ax7.set_title('Cumulative Trade P&L', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Trade Number')
            ax7.set_ylabel('Cumulative P&L (USD)')
            ax7.grid(True, alpha=0.3)
            
            # 8. Win/Loss Analysis
            ax8 = fig.add_subplot(gs[3, 2])
            wins = self.trades[self.trades['net_pnl'] > 0]['net_pnl']
            losses = self.trades[self.trades['net_pnl'] < 0]['net_pnl']
            
            ax8.bar(['Wins', 'Losses'], [len(wins), len(losses)], 
                   color=['#18A558', '#A23B72'], alpha=0.7, edgecolor='black')
            ax8.set_title('Win/Loss Count', fontsize=12, fontweight='bold')
            ax8.set_ylabel('Number of Trades')
            ax8.grid(True, alpha=0.3, axis='y')
            
            # Add win rate text
            win_rate = len(wins) / len(self.trades) * 100
            ax8.text(0.5, 0.95, f'Win Rate: {win_rate:.1f}%', 
                    transform=ax8.transAxes, ha='center', va='top',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Statistical Arbitrage Strategy - Performance Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved to {output_path}")
        
        return fig
    
    def plot_trade_analysis(self, output_path: Optional[str] = None):
        """
        Detailed trade analysis visualization
        """
        if len(self.trades) == 0:
            print("No trades to analyze")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Trade Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Holding Period Distribution
        ax1 = axes[0, 0]
        ax1.hist(self.trades['holding_period'], bins=30, alpha=0.7, 
                color='#2E86AB', edgecolor='black')
        ax1.set_title('Holding Period Distribution')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. P&L vs Holding Period
        ax2 = axes[0, 1]
        colors = ['green' if x > 0 else 'red' for x in self.trades['net_pnl']]
        ax2.scatter(self.trades['holding_period'], self.trades['net_pnl'], 
                   c=colors, alpha=0.6, s=50)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('P&L vs Holding Period')
        ax2.set_xlabel('Holding Period (hours)')
        ax2.set_ylabel('Net P&L (USD)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Transaction Costs Analysis
        ax3 = axes[0, 2]
        cost_breakdown = pd.DataFrame({
            'Transaction Costs': [self.trades['transaction_costs'].sum()],
            'Funding Costs': [self.trades['funding_costs'].sum()],
            'Gross P&L': [self.trades['gross_pnl'].sum()]
        })
        cost_breakdown.T.plot(kind='bar', ax=ax3, legend=False, color=['#A23B72', '#F18F01', '#18A558'])
        ax3.set_title('Cost Breakdown')
        ax3.set_ylabel('USD')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Monthly Returns
        ax4 = axes[1, 0]
        self.trades['month'] = pd.to_datetime(self.trades['exit_time']).dt.to_period('M')
        monthly_pnl = self.trades.groupby('month')['net_pnl'].sum()
        colors_monthly = ['green' if x > 0 else 'red' for x in monthly_pnl]
        monthly_pnl.plot(kind='bar', ax=ax4, color=colors_monthly, alpha=0.7)
        ax4.set_title('Monthly P&L')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Net P&L (USD)')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Return Distribution by Trade
        ax5 = axes[1, 1]
        ax5.hist(self.trades['return_pct'], bins=30, alpha=0.7, 
                color='#6A4C93', edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax5.set_title('Return % Distribution')
        ax5.set_xlabel('Return (%)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # 6. ETH vs BTC P&L Contribution
        ax6 = axes[1, 2]
        pnl_contribution = pd.DataFrame({
            'ETH P&L': [self.trades['eth_pnl'].sum()],
            'BTC P&L': [self.trades['btc_pnl'].sum()]
        })
        pnl_contribution.T.plot(kind='bar', ax=ax6, legend=False, 
                               color=['#1982C4', '#F18F01'])
        ax6.set_title('P&L Contribution by Asset')
        ax6.set_ylabel('Total P&L (USD)')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Trade analysis plot saved to {output_path}")
        
        return fig
