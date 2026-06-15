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

from config import OUTPUT_DIR, DATA_FILES

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
        self.btc_close = pd.read_parquet(DATA_FILES['close'])['BTCUSDT']
        
        self.positions = pd.read_parquet(OUTPUT_DIR / 'positions.parquet')
        
        self.trades = pd.read_csv(OUTPUT_DIR / 'trades.csv')
        if 'timestamp' in self.trades.columns:
            self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'])
        
        self.feature_importance = pd.read_csv(OUTPUT_DIR / 'feature_importance.csv')
        
    def plot_equity_curve(self, save: bool = True):
        """绘制净值曲线"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), 
                                  gridspec_kw={'height_ratios': [3, 1, 1]})
        
        equity = self.equity_curve if hasattr(self, 'equity_curve') else self.result.equity_curve
        equity_norm = equity / equity.iloc[0]
        ax1 = axes[0]
        ax1.plot(equity_norm.index, equity_norm.values, label='Strategy', linewidth=2, color='blue')

        btc_close = self.btc_close if hasattr(self, 'btc_close') else None
        if btc_close is not None:
            btc_norm = btc_close / btc_close.iloc[0]
            common_btc_idx = equity_norm.index.intersection(btc_norm.index)
            ax1.plot(common_btc_idx, btc_norm.loc[common_btc_idx].values,
                    label='BTC', linewidth=1.2, color='orange', alpha=0.8)

        final_ret = (equity_norm.iloc[-1] - 1) * 100
        ax1.annotate(f'+{final_ret:.0f}%', xy=(equity_norm.index[-1], equity_norm.iloc[-1]),
                    fontsize=10, fontweight='bold', color='blue')
        ax1.set_ylabel('Normalized NAV')
        ax1.set_title('Strategy vs BTC', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
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
        rolling_sharpe = returns.rolling(720).mean() / returns.rolling(720).std() * np.sqrt(24*365)
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values, color='green', linewidth=0.8)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax3.set_ylabel('Rolling Sharpe (30d)')
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
        """绘制持仓分析 (修复版)"""
        positions = self.positions if hasattr(self, 'positions') else self.result.positions
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 持仓数量随时间变化 (修复: 计算long+short)
        ax1 = axes[0, 0]
        n_long = (positions > 0).sum(axis=1)
        n_short = (positions < 0).sum(axis=1)
        n_total = (positions != 0).sum(axis=1)
        ax1.plot(n_total.index, n_total.values, linewidth=0.8, label='Total', color='blue')
        ax1.plot(n_long.index, n_long.values, linewidth=0.6, label='Long', color='green', alpha=0.7)
        ax1.plot(n_short.index, n_short.values, linewidth=0.6, label='Short', color='red', alpha=0.7)
        ax1.set_title('Number of Positions Over Time')
        ax1.set_ylabel('Count')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 2. 多空权重
        ax2 = axes[0, 1]
        long_weight = positions.clip(lower=0).sum(axis=1)
        short_weight = positions.clip(upper=0).sum(axis=1)
        ax2.plot(long_weight.index, long_weight.values, linewidth=0.8, label='Long Weight', color='green')
        ax2.plot(short_weight.index, short_weight.values, linewidth=0.8, label='Short Weight', color='red')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax2.set_title('Long/Short Weight Over Time')
        ax2.set_ylabel('Total Weight')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 3. 最常持仓的币种 (long + short分开)
        ax3 = axes[1, 0]
        long_counts = (positions > 0).sum().sort_values(ascending=True).tail(15)
        short_counts = (positions < 0).sum().sort_values(ascending=True).tail(15)
        all_syms = list(set(long_counts.index) | set(short_counts.index))
        all_syms.sort(key=lambda s: (positions > 0).sum().get(s, 0) + (positions < 0).sum().get(s, 0))
        all_syms = all_syms[-20:]
        y_pos = range(len(all_syms))
        ax3.barh(y_pos, [(positions > 0)[s].sum() for s in all_syms], color='green', alpha=0.7, label='Long')
        ax3.barh(y_pos, [-(positions < 0)[s].sum() for s in all_syms], color='red', alpha=0.7, label='Short')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(all_syms, fontsize=7)
        ax3.set_title('Top 20 Most Held Symbols')
        ax3.set_xlabel('Hours Held')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. 换手率分布 (修复: 只在rebalance bar计算)
        ax4 = axes[1, 1]
        # 找到真正的rebalance bar: 持仓发生实际变化的bar
        pos_change = positions.diff()
        rebalance_mask = pos_change.abs().sum(axis=1) > 0.01  # 排除浮点噪音
        turnover_at_rebalance = pos_change.loc[rebalance_mask].abs().sum(axis=1)
        if len(turnover_at_rebalance) > 0:
            ax4.hist(turnover_at_rebalance.values * 100, bins=30, color='green', 
                    alpha=0.7, edgecolor='black')
            ax4.axvline(x=turnover_at_rebalance.mean()*100, color='red', linestyle='--',
                       label=f'Mean: {turnover_at_rebalance.mean()*100:.1f}%')
            ax4.legend(fontsize=8)
        ax4.set_title(f'Turnover Distribution (n={len(turnover_at_rebalance)} rebalances)')
        ax4.set_xlabel('Turnover per Rebalance (%)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(OUTPUT_DIR / 'position_analysis.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: position_analysis.png")
        
        plt.close()
    
    def plot_ic_decay(self, save: bool = True):
        """绘制IC Decay图"""
        ic_decay_path = OUTPUT_DIR / 'factor_research' / 'ic_decay_analysis.csv'
        if not ic_decay_path.exists():
            print("  IC decay file not found, skipping")
            return
        
        decay = pd.read_csv(ic_decay_path)
        lag_cols = [c for c in decay.columns if c.startswith('ric_')]
        lags = [int(c.replace('ric_', '').replace('h', '')) for c in lag_cols]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 取research_score最高的Top 10因子
        report_path = OUTPUT_DIR / 'factor_research' / 'factor_research_report.csv'
        if report_path.exists():
            report = pd.read_csv(report_path)
            top_factors = report.nlargest(10, 'research_score')['factor'].tolist()
        else:
            top_factors = decay['factor'].head(10).tolist()
        
        for _, row in decay[decay['factor'].isin(top_factors)].iterrows():
            vals = [row[c] for c in lag_cols]
            ax.plot(lags, vals, 'o-', label=row['factor'], linewidth=1.2, markersize=4)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Holding Period (hours)')
        ax.set_ylabel('Rank IC')
        ax.set_title('IC Decay Analysis (Top 10 Factors)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=7, ncol=2, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(OUTPUT_DIR / 'ic_decay.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: ic_decay.png")
        plt.close()
    
    def plot_attribution(self, save: bool = True):
        """绘制收益归因分析图"""
        positions = self.positions if hasattr(self, 'positions') else self.result.positions
        equity = self.equity_curve if hasattr(self, 'equity_curve') else self.result.equity_curve
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 累计收益分解: Gross vs Cost vs Net
        ax1 = axes[0, 0]
        returns_series = equity.pct_change().dropna()
        cumulative = (1 + returns_series).cumprod()
        ax1.plot(cumulative.index, cumulative.values, label='Net Return', color='blue', linewidth=1.5)
        ax1.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
        ax1.set_title('Cumulative Net Return')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 2. 滚动30天收益
        ax2 = axes[0, 1]
        rolling_30d = returns_series.rolling(720).sum() * 100  # 30天=720小时
        ax2.plot(rolling_30d.index, rolling_30d.values, color='purple', linewidth=0.8)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        ax2.set_title('Rolling 30-Day Return (%)')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 3. 多头 vs 空头累计收益
        ax3 = axes[1, 0]
        try:
            long_pos_df = pd.read_parquet(OUTPUT_DIR / 'long_positions.parquet')
            short_pos_df = pd.read_parquet(OUTPUT_DIR / 'short_positions.parquet')
            from config import DATA_FILES
            raw_returns = pd.read_parquet(DATA_FILES['close']).pct_change()
            
            common_idx = raw_returns.index.intersection(long_pos_df.index)
            common_cols = raw_returns.columns.intersection(long_pos_df.columns)
            
            long_ret = (long_pos_df.shift(1).loc[common_idx, common_cols] * 
                       raw_returns.loc[common_idx, common_cols]).sum(axis=1)
            short_ret = (short_pos_df.shift(1).loc[common_idx, common_cols] * 
                        raw_returns.loc[common_idx, common_cols]).sum(axis=1)
            
            cum_long = (1 + long_ret).cumprod()
            cum_short = (1 + short_ret).cumprod()
            
            ax3.plot(cum_long.index, cum_long.values, label='Long Leg', color='green', linewidth=1)
            ax3.plot(cum_short.index, cum_short.values, label='Short Leg', color='red', linewidth=1)
            ax3.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
            ax3.legend(fontsize=8)
        except Exception as e:
            ax3.text(0.5, 0.5, f'Error: {e}', transform=ax3.transAxes, ha='center')
        ax3.set_title('Long vs Short Leg Cumulative Return')
        ax3.set_ylabel('Cumulative Return')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 4. 每次rebalance的收益
        ax4 = axes[1, 1]
        # 按rebalance周期汇总收益
        daily_ret = returns_series.resample('D').sum() * 100
        colors = ['green' if r > 0 else 'red' for r in daily_ret.values]
        ax4.bar(daily_ret.index, daily_ret.values, color=colors, alpha=0.6, width=1)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_title('Daily Returns')
        ax4.set_ylabel('Return (%)')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        if save:
            plt.savefig(OUTPUT_DIR / 'attribution_analysis.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: attribution_analysis.png")
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
        self.plot_ic_decay()
        self.plot_attribution()
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
