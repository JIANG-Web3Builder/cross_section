"""
Phase 4: Strategy Assembly & Backtest 策略组装与回测模块
- 择时模块 (Timing Filter)
- 持仓逻辑
- 回测引擎
- 绩效评估
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import (
    TIMING_MA_PERIOD, TIMING_VOL_PERIOD, TIMING_VOL_QUANTILE,
    TOP_N_HOLD, REBALANCE_HOURS, MAKER_FEE, TAKER_FEE, SLIPPAGE,
    FORWARD_HOURS, OUTPUT_DIR
)


@dataclass
class BacktestResult:
    """回测结果"""
    equity_curve: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]


class TimingModule:
    """择时模块"""
    
    def __init__(self, market_index: pd.DataFrame):
        self.market_index = market_index
        self.signals: pd.Series = None
        
    def generate_signals(self) -> pd.Series:
        """
        生成择时信号
        - 指数 < MA168 -> 空仓
        - 波动率 > 历史90%分位数 -> 空仓
        """
        price = self.market_index['price']
        returns = self.market_index['returns']
        
        # MA择时
        ma = price.rolling(TIMING_MA_PERIOD).mean()
        ma_signal = price > ma
        
        # 波动率择时
        vol = returns.rolling(TIMING_VOL_PERIOD).std()
        vol_threshold = vol.expanding().quantile(TIMING_VOL_QUANTILE)
        vol_signal = vol < vol_threshold
        
        # 综合信号 (两个条件都满足才持仓)
        self.signals = (ma_signal & vol_signal).astype(int)
        
        # 统计
        total_bars = len(self.signals.dropna())
        hold_bars = self.signals.sum()
        hold_ratio = hold_bars / total_bars * 100
        
        print(f"  Timing signals generated:")
        print(f"    Total bars: {total_bars}")
        print(f"    Hold bars: {int(hold_bars)} ({hold_ratio:.1f}%)")
        print(f"    Empty bars: {total_bars - int(hold_bars)} ({100-hold_ratio:.1f}%)")
        
        return self.signals


class PositionManager:
    """持仓管理"""
    
    def __init__(self, universe: pd.DataFrame, predictions: pd.DataFrame,
                 timing_signals: pd.Series):
        self.universe = universe
        self.predictions = predictions
        self.timing_signals = timing_signals
        
    def generate_positions(self) -> pd.DataFrame:
        """
        生成持仓权重
        - 选取预测Rank最高的前N名
        - 等权或因子值加权
        """
        positions = pd.DataFrame(0.0, index=self.predictions.index, 
                                 columns=self.predictions.columns)
        
        timestamps = self.predictions.index
        
        for i, ts in enumerate(timestamps):
            # 检查择时信号
            if ts not in self.timing_signals.index or self.timing_signals[ts] == 0:
                continue
            
            # 只在换仓时间点换仓
            if i % REBALANCE_HOURS != 0:
                if i > 0:
                    positions.loc[ts] = positions.iloc[i-1]
                continue
            
            # 获取当前票池
            universe_mask = self.universe.loc[ts]
            
            # 获取预测值
            pred = self.predictions.loc[ts]
            pred = pred[universe_mask].dropna()
            
            if len(pred) < TOP_N_HOLD:
                continue
            
            # 选择Top N
            top_symbols = pred.nlargest(TOP_N_HOLD).index
            
            # 等权分配
            weight = 1.0 / TOP_N_HOLD
            positions.loc[ts, top_symbols] = weight
        
        # 前向填充(持仓不变时)
        positions = positions.ffill()
        
        return positions


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, returns: pd.DataFrame, positions: pd.DataFrame,
                 market_index: pd.DataFrame):
        self.returns = returns
        self.positions = positions
        self.market_index = market_index
        
    def run_backtest(self) -> BacktestResult:
        """运行回测"""
        print("=" * 60)
        print("Running backtest...")
        
        # 对齐数据
        common_idx = self.returns.index.intersection(self.positions.index)
        returns = self.returns.loc[common_idx]
        positions = self.positions.loc[common_idx]
        
        # 计算持仓变化(用于计算交易成本)
        position_changes = positions.diff().abs()
        
        # 单边交易成本
        trade_cost_rate = TAKER_FEE + SLIPPAGE
        
        # 计算策略收益
        # 持仓收益
        portfolio_returns = (positions.shift(1) * returns).sum(axis=1)
        
        # 扣除交易成本
        turnover = position_changes.sum(axis=1)
        trade_costs = turnover * trade_cost_rate
        
        net_returns = portfolio_returns - trade_costs
        
        # 累计收益
        equity_curve = (1 + net_returns).cumprod()
        
        # 计算交易记录
        trades = self._generate_trades(positions)
        
        # 计算绩效指标
        metrics = self._calc_metrics(net_returns, equity_curve, turnover)
        
        return BacktestResult(
            equity_curve=equity_curve,
            positions=positions,
            trades=trades,
            metrics=metrics
        )
    
    def _generate_trades(self, positions: pd.DataFrame) -> pd.DataFrame:
        """生成交易记录"""
        trades = []
        position_changes = positions.diff()
        
        for ts in position_changes.index[1:]:
            changes = position_changes.loc[ts]
            for symbol in changes[changes != 0].index:
                change = changes[symbol]
                trades.append({
                    'timestamp': ts,
                    'symbol': symbol,
                    'direction': 'BUY' if change > 0 else 'SELL',
                    'weight_change': abs(change)
                })
        
        return pd.DataFrame(trades)
    
    def _calc_metrics(self, returns: pd.Series, equity: pd.Series, 
                      turnover: pd.Series) -> Dict[str, float]:
        """计算绩效指标"""
        
        # 基本收益指标
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        
        # 年化收益 (假设1小时数据)
        n_hours = len(returns)
        n_years = n_hours / (24 * 365)
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # 波动率
        ann_vol = returns.std() * np.sqrt(24 * 365)
        
        # 夏普比率 (假设无风险利率为0)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # 最大回撤
        rolling_max = equity.cummax()
        drawdown = equity / rolling_max - 1
        max_drawdown = drawdown.min()
        
        # 卡尔玛比率
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # 盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 平均换手率
        avg_turnover = turnover.mean()
        
        # 与大盘对比
        index_returns = self.market_index['returns'].loc[returns.index]
        index_equity = (1 + index_returns).cumprod()
        index_total_return = index_equity.iloc[-1] / index_equity.iloc[0] - 1
        excess_return = total_return - index_total_return
        
        # 信息比率
        excess_returns = returns - index_returns
        tracking_error = excess_returns.std() * np.sqrt(24 * 365)
        info_ratio = (ann_return - (1 + index_total_return) ** (1/n_years) + 1) / tracking_error if tracking_error > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'avg_turnover': avg_turnover,
            'index_return': index_total_return,
            'excess_return': excess_return,
            'info_ratio': info_ratio,
            'n_hours': n_hours,
            'n_years': n_years,
        }
        
        return metrics


class StrategyAssembler:
    """策略组装器"""
    
    def __init__(self, panel_data: Dict[str, pd.DataFrame],
                 factors_normalized: Dict[str, pd.DataFrame],
                 factor_matrix: pd.DataFrame,
                 model, selected_features: List[str]):
        self.panel_data = panel_data
        self.factors_normalized = factors_normalized
        self.factor_matrix = factor_matrix
        self.model = model
        self.selected_features = selected_features
        
        self.timing_module: TimingModule = None
        self.predictions: pd.DataFrame = None
        self.backtest_result: BacktestResult = None
        
    def generate_predictions(self) -> pd.DataFrame:
        """生成模型预测"""
        print("=" * 60)
        print("Generating predictions...")
        
        # 获取选择的特征
        X = self.factor_matrix[self.selected_features]
        
        # 预测
        pred_values = self.model.predict(X)
        predictions_stacked = pd.Series(pred_values, index=X.index, name='prediction')
        
        # Unstack为宽表
        predictions = predictions_stacked.unstack(level='symbol')
        
        self.predictions = predictions
        
        print(f"  Predictions shape: {predictions.shape}")
        
        return predictions
    
    def run_strategy(self) -> BacktestResult:
        """运行完整策略"""
        print("\n" + "=" * 60)
        print("PHASE 4: STRATEGY ASSEMBLY & BACKTEST")
        print("=" * 60)
        
        # 择时模块
        print("Setting up timing module...")
        self.timing_module = TimingModule(self.panel_data['market_index'])
        timing_signals = self.timing_module.generate_signals()
        
        # 生成预测
        self.generate_predictions()
        
        # 生成持仓
        print("=" * 60)
        print("Generating positions...")
        pos_manager = PositionManager(
            self.panel_data['universe'],
            self.predictions,
            timing_signals
        )
        positions = pos_manager.generate_positions()
        
        # 运行回测
        backtest_engine = BacktestEngine(
            self.panel_data['returns'],
            positions,
            self.panel_data['market_index']
        )
        self.backtest_result = backtest_engine.run_backtest()
        
        # 打印结果
        self._print_results()
        
        # 保存结果
        self._save_results()
        
        print("\n" + "=" * 60)
        print("Phase 4 completed!")
        print("=" * 60)
        
        return self.backtest_result
    
    def _print_results(self):
        """打印回测结果"""
        print("=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        m = self.backtest_result.metrics
        
        print(f"\n{'='*40}")
        print("收益指标")
        print(f"{'='*40}")
        print(f"  总收益率:        {m['total_return']*100:>10.2f}%")
        print(f"  年化收益率:      {m['ann_return']*100:>10.2f}%")
        print(f"  大盘收益率:      {m['index_return']*100:>10.2f}%")
        print(f"  超额收益:        {m['excess_return']*100:>10.2f}%")
        
        print(f"\n{'='*40}")
        print("风险指标")
        print(f"{'='*40}")
        print(f"  年化波动率:      {m['ann_volatility']*100:>10.2f}%")
        print(f"  最大回撤:        {m['max_drawdown']*100:>10.2f}%")
        
        print(f"\n{'='*40}")
        print("风险调整收益")
        print(f"{'='*40}")
        print(f"  夏普比率:        {m['sharpe_ratio']:>10.3f}")
        print(f"  卡尔玛比率:      {m['calmar_ratio']:>10.3f}")
        print(f"  信息比率:        {m['info_ratio']:>10.3f}")
        
        print(f"\n{'='*40}")
        print("交易统计")
        print(f"{'='*40}")
        print(f"  胜率:            {m['win_rate']*100:>10.2f}%")
        print(f"  盈亏比:          {m['profit_loss_ratio']:>10.3f}")
        print(f"  平均换手率:      {m['avg_turnover']*100:>10.2f}%")
        
        print(f"\n{'='*40}")
        print("回测周期")
        print(f"{'='*40}")
        print(f"  总小时数:        {int(m['n_hours']):>10}")
        print(f"  总年数:          {m['n_years']:>10.2f}")
        
    def _save_results(self):
        """保存回测结果"""
        # 保存净值曲线
        self.backtest_result.equity_curve.to_csv(OUTPUT_DIR / 'equity_curve.csv')
        
        # 保存持仓
        self.backtest_result.positions.to_parquet(OUTPUT_DIR / 'positions.parquet')
        
        # 保存交易记录
        self.backtest_result.trades.to_csv(OUTPUT_DIR / 'trades.csv', index=False)
        
        # 保存指标
        metrics_df = pd.DataFrame([self.backtest_result.metrics])
        metrics_df.to_csv(OUTPUT_DIR / 'metrics.csv', index=False)
        
        # 保存预测
        self.predictions.to_parquet(OUTPUT_DIR / 'predictions.parquet')
        
        print(f"\n  Results saved to {OUTPUT_DIR}")


def run_full_pipeline():
    """运行完整的策略流水线"""
    from data_engineering import DataEngine
    from factor_engineering import FactorEngine
    from model_training import ModelTrainer
    
    print("\n" + "=" * 70)
    print(" CROSS-SECTIONAL MULTI-FACTOR STRATEGY")
    print(" 截面多因子量化策略")
    print("=" * 70)
    
    # Phase 1: 数据工程
    data_engine = DataEngine()
    panel_data = data_engine.run_pipeline()
    
    # Phase 2: 因子工程
    factor_engine = FactorEngine(panel_data)
    factors_normalized, factor_matrix = factor_engine.run_pipeline()
    
    # Phase 3: 模型训练
    trainer = ModelTrainer(panel_data, factors_normalized, factor_matrix)
    model, selected_features = trainer.run_pipeline()
    
    # Phase 4: 策略回测
    assembler = StrategyAssembler(
        panel_data, factors_normalized, factor_matrix,
        model, selected_features
    )
    result = assembler.run_strategy()
    
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = run_full_pipeline()
