"""
Visualization 可视化模块
- 净值曲线
- 回撤分析
- 因子分析图
- 持仓分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_DIR

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class StrategyVisualizer:
    """策略可视化"""
    
    def __init__(self, backtest_result=None):
        self.result = backtest_result
        
    def load_results(self):
        """从文件加载结果"""
        self.equity_curve = pd.read_csv(
            OUTPUT_DIR / 'equity_curve.csv', 
            index_col=0, parse_dates=True
        ).squeeze()
        
        self.metrics = pd.read_csv(OUTPUT_DIR / 'metrics.csv').iloc[0].to_dict()
        
        self.market_index = pd.read_parquet(OUTPUT_DIR / 'market_index.parquet')
        
        self.positions = pd.read_parquet(OUTPUT_DIR / 'positions.parquet')
        
        self.trades = pd.read_csv(OUTPUT_DIR / 'trades.csv')
        if 'timestamp' in self.trades.columns:
            self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'])
        
        self.feature_importance = pd.read_csv(OUTPUT_DIR / 'feature_importance.csv')
        
    def plot_equity_curve(self, save: bool = True):
        """绘制净值曲线"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), 
                                  gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 净值曲线
        ax1 = axes[0]
        
        # 策略净值
        equity = self.equity_curve if hasattr(self, 'equity_curve') else self.result.equity_curve
        ax1.plot(equity.index, equity.values, label='Strategy', linewidth=1.5, color='blue')
        
        # 大盘指数 (标准化到相同起点)
        market = self.market_index if hasattr(self, 'market_index') else None
        if market is not None:
            market_norm = market['price'] / market['price'].iloc[0]
            common_idx = equity.index.intersection(market_norm.index)
            ax1.plot(common_idx, market_norm.loc[common_idx].values, 
                    label='Market Index', linewidth=1, color='gray', alpha=0.7)
        
        ax1.set_title('Strategy Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        # 回撤曲线
        ax2 = axes[1]
        rolling_max = equity.cummax()
        drawdown = (equity / rolling_max - 1) * 100
        ax2.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=0.5)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_ylim([drawdown.min() * 1.1, 5])
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 滚动夏普
        ax3 = axes[2]
        returns = equity.pct_change().dropna()
        rolling_sharpe = returns.rolling(168).mean() / returns.rolling(168).std() * np.sqrt(24*365)
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values, color='green', linewidth=0.8)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax3.set_ylabel('Rolling Sharpe (7d)')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(OUTPUT_DIR / 'equity_curve.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: equity_curve.png")
        
        plt.close()
        
    def plot_monthly_returns(self, save: bool = True):
        """绘制月度收益热力图"""
        equity = self.equity_curve if hasattr(self, 'equity_curve') else self.result.equity_curve
        
        # 计算月度收益
        monthly = equity.resample('M').last().pct_change().dropna()
        
        # 创建年月矩阵
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values * 100
        })
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 热力图
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', 
                       vmin=-20, vmax=20)
        
        # 标签
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        # 添加数值标注
        for i in range(len(pivot.index)):
            for j in range(12):
                if j+1 in pivot.columns:
                    val = pivot.iloc[i, pivot.columns.get_loc(j+1)]
                    if not np.isnan(val):
                        color = 'white' if abs(val) > 10 else 'black'
                        ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                               color=color, fontsize=9)
        
        plt.colorbar(im, ax=ax, label='Monthly Return (%)')
        ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(OUTPUT_DIR / 'monthly_returns.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: monthly_returns.png")
        
        plt.close()
        
    def plot_feature_importance(self, top_n: int = 25, save: bool = True):
        """绘制特征重要性"""
        importance = self.feature_importance if hasattr(self, 'feature_importance') else None
        if importance is None:
            return
            
        top_features = importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))[::-1]
        
        bars = ax.barh(range(len(top_features)), top_features['importance_pct'].values,
                       color=colors)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (%)')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标注
        for i, (bar, val) in enumerate(zip(bars, top_features['importance_pct'].values)):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: feature_importance.png")
        
        plt.close()
        
    def plot_position_analysis(self, save: bool = True):
        """绘制持仓分析"""
        positions = self.positions if hasattr(self, 'positions') else self.result.positions
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 持仓数量随时间变化
        ax1 = axes[0, 0]
        n_positions = (positions > 0).sum(axis=1)
        ax1.plot(n_positions.index, n_positions.values, linewidth=0.8)
        ax1.set_title('Number of Positions Over Time')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 2. 持仓集中度
        ax2 = axes[0, 1]
        position_sum = positions.sum(axis=1)
        ax2.plot(position_sum.index, position_sum.values, linewidth=0.8, color='orange')
        ax2.set_title('Total Position Weight Over Time')
        ax2.set_ylabel('Total Weight')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 3. 最常持仓的币种
        ax3 = axes[1, 0]
        hold_counts = (positions > 0).sum().sort_values(ascending=True)
        top_20 = hold_counts.tail(20)
        ax3.barh(range(len(top_20)), top_20.values, color='steelblue')
        ax3.set_yticks(range(len(top_20)))
        ax3.set_yticklabels(top_20.index)
        ax3.set_title('Top 20 Most Held Symbols')
        ax3.set_xlabel('Hours Held')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. 换手率分布
        ax4 = axes[1, 1]
        turnover = positions.diff().abs().sum(axis=1)
        turnover_clean = turnover[turnover > 0]
        ax4.hist(turnover_clean.values * 100, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax4.set_title('Turnover Distribution')
        ax4.set_xlabel('Turnover (%)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(OUTPUT_DIR / 'position_analysis.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: position_analysis.png")
        
        plt.close()
    
    def plot_return_distribution(self, save: bool = True):
        """绘制收益分布"""
        equity = self.equity_curve if hasattr(self, 'equity_curve') else self.result.equity_curve
        returns = equity.pct_change().dropna() * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 收益分布直方图
        ax1 = axes[0]
        ax1.hist(returns.values, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax1.axvline(x=returns.mean(), color='green', linestyle='-', linewidth=1, label=f'Mean: {returns.mean():.3f}%')
        ax1.set_title('Hourly Returns Distribution')
        ax1.set_xlabel('Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # QQ图
        ax2 = axes[1]
        from scipy import stats
        stats.probplot(returns.values, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(OUTPUT_DIR / 'return_distribution.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: return_distribution.png")
        
        plt.close()
        
    def generate_report(self):
        """生成完整报告"""
        print("=" * 60)
        print("Generating visualization report...")
        
        self.plot_equity_curve()
        self.plot_monthly_returns()
        self.plot_feature_importance()
        self.plot_position_analysis()
        self.plot_return_distribution()
        
        print("  All visualizations saved!")
        
    def print_summary(self):
        """打印摘要报告"""
        metrics = self.metrics if hasattr(self, 'metrics') else self.result.metrics
        
        print("\n" + "=" * 60)
        print(" STRATEGY PERFORMANCE SUMMARY")
        print("=" * 60)
        
        summary = f"""
┌─────────────────────────────────────────────────────────┐
│                    PERFORMANCE METRICS                   │
├─────────────────────────────────────────────────────────┤
│  Total Return:           {metrics['total_return']*100:>8.2f}%                     │
│  Annualized Return:      {metrics['ann_return']*100:>8.2f}%                     │
│  Annualized Volatility:  {metrics['ann_volatility']*100:>8.2f}%                     │
│  Max Drawdown:           {metrics['max_drawdown']*100:>8.2f}%                     │
├─────────────────────────────────────────────────────────┤
│  Sharpe Ratio:           {metrics['sharpe_ratio']:>8.3f}                       │
│  Calmar Ratio:           {metrics['calmar_ratio']:>8.3f}                       │
│  Information Ratio:      {metrics['info_ratio']:>8.3f}                       │
├─────────────────────────────────────────────────────────┤
│  Win Rate:               {metrics['win_rate']*100:>8.2f}%                     │
│  Profit/Loss Ratio:      {metrics['profit_loss_ratio']:>8.3f}                       │
│  Avg Turnover:           {metrics['avg_turnover']*100:>8.2f}%                     │
├─────────────────────────────────────────────────────────┤
│  Benchmark Return:       {metrics['index_return']*100:>8.2f}%                     │
│  Excess Return:          {metrics['excess_return']*100:>8.2f}%                     │
└─────────────────────────────────────────────────────────┘
"""
        print(summary)


if __name__ == "__main__":
    viz = StrategyVisualizer()
    viz.load_results()
    viz.generate_report()
    viz.print_summary()
