"""
Phase 4: Strategy Assembly & Backtest - V3 REGENESIS
策略组装与回测模块 - 市场中性对冲版

特点:
- 市场中性: Long Top 10 Alts + Short BTC
- Beta中性权重配置
- 波动率倒数加权 + 换仓缓冲
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import (
    TIMING_RSI_PERIOD, TIMING_RSI_LONG, TIMING_RSI_SHORT, TIMING_RSI_RESAMPLE,
    LONG_N, TOTAL_HOLD, REBALANCE_HOURS, 
    MAKER_FEE, TAKER_FEE, SLIPPAGE, OUTPUT_DIR,
    BETA_LOOKBACK, STOP_LOSS_PCT, MAX_PORTFOLIO_LEVERAGE
)

# 初始资金
INITIAL_CAPITAL = 100000


@dataclass
class BacktestResult:
    """回测结果"""
    equity_curve: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    long_positions: pd.DataFrame = None
    short_positions: pd.DataFrame = None


class TimingModule:
    """择时模块 - 基于4小时BTC RSI"""
    
    def __init__(self, market_index: pd.DataFrame, btc_close: pd.Series = None):
        self.market_index = market_index
        self.btc_close = btc_close  # BTC收盘价
        self.signals: pd.Series = None
        self.position_multiplier: pd.Series = None  # 仓位乘数
        
    def calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI计算"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
        
    def generate_signals(self) -> pd.Series:
        """
        生成4小时BTC RSI择时信号
        - RSI < 35: 做多仓位增加 (position_multiplier > 1)
        - RSI > 65: 做空仓位增加 (position_multiplier 调整多空比例)
        - 中间区间: 正常仓位
        """
        if self.btc_close is None:
            print("  Warning: BTC close prices not provided, using market index")
            price = self.market_index['price']
        else:
            price = self.btc_close
        
        # 重采样到4小时级别
        price_4h = price.resample(TIMING_RSI_RESAMPLE).last().dropna()
        
        # 计算RSI
        rsi_4h = self.calc_rsi(price_4h, TIMING_RSI_PERIOD)
        
        # 将RSI信号向前填充到原始时间索引
        rsi = rsi_4h.reindex(price.index, method='ffill')
        
        # 生成仓位乘数
        # RSI < 35: 看多，多头仓位增加 (1.0 -> 1.5)
        # RSI > 65: 看空，空头仓位增加 (1.0 -> 1.5)
        # 中间区间: 正常仓位 (1.0)
        long_multiplier = pd.Series(1.0, index=price.index)
        short_multiplier = pd.Series(1.0, index=price.index)
        
        # RSI低于35: 做多仓位增加
        oversold = rsi < TIMING_RSI_LONG
        long_multiplier[oversold] = 1.5
        
        # RSI高于65: 做空仓位增加
        overbought = rsi > TIMING_RSI_SHORT
        short_multiplier[overbought] = 1.5
        
        # 存储乘数
        self.position_multiplier = pd.DataFrame({
            'long_mult': long_multiplier,
            'short_mult': short_multiplier,
            'rsi': rsi
        })
        
        # 择时信号: 始终为1 (不再空仓)
        self.signals = pd.Series(1, index=price.index)
        
        # 统计
        total_bars = len(rsi.dropna())
        oversold_bars = oversold.sum()
        overbought_bars = overbought.sum()
        
        print(f"  RSI Timing signals generated (4h BTC RSI):")
        print(f"    RSI Period: {TIMING_RSI_PERIOD}")
        print(f"    Long threshold (RSI<{TIMING_RSI_LONG}): {oversold_bars} bars ({oversold_bars/total_bars*100:.1f}%)")
        print(f"    Short threshold (RSI>{TIMING_RSI_SHORT}): {overbought_bars} bars ({overbought_bars/total_bars*100:.1f}%)")
        print(f"    Current RSI: {rsi.iloc[-1]:.1f}")
        
        return self.signals


class MarketNeutralPositionManager:
    """
    市场中性持仓管理 - 纯截面多空策略 (Cross-Sectional Long-Short)
    
    核心策略 (修复版):
    - 剔除BTC: BTC不参与排序，也不作为对冲端
    - 做多: 预测分最高的Top N个币 (等权)
    - 做空: 预测分最低的Bottom N个币 (等权)
    - 权重: 多头总仓位+1.0，空头总仓位-1.0
    - 目的: 完全对冲市场Beta风险，只赚取Alpha收益
    """
    
    def __init__(self, universe: pd.DataFrame, predictions: pd.DataFrame,
                 timing_signals: pd.Series, returns: pd.DataFrame,
                 market_returns: pd.Series, volume: pd.DataFrame,
                 btc_returns: pd.Series = None):
        self.universe = universe
        self.predictions = predictions
        self.timing_signals = timing_signals
        self.returns = returns
        self.market_returns = market_returns
        self.volume = volume
        
        # BTC收益率 (用于对冲)
        self.btc_returns = btc_returns if btc_returns is not None else market_returns
        
        # 当前持仓 (用于缓冲机制)
        self.current_long_symbols: List[str] = []
        
    def calc_rolling_beta(self, symbol: str, timestamp) -> float:
        """计算个币对BTC的Beta"""
        loc_idx = self.returns.index.get_loc(timestamp)
        if loc_idx < BETA_LOOKBACK:
            return 1.0
        
        hist_start = loc_idx - BETA_LOOKBACK
        
        coin_returns = self.returns[symbol].iloc[hist_start:loc_idx].dropna()
        btc_returns = self.btc_returns.iloc[hist_start:loc_idx]
        
        # 对齐
        common_idx = coin_returns.index.intersection(btc_returns.index)
        if len(common_idx) < 20:
            return 1.0
        
        coin_ret = coin_returns.loc[common_idx]
        btc_ret = btc_returns.loc[common_idx]
        
        # Beta = Cov(r_i, r_btc) / Var(r_btc)
        cov = coin_ret.cov(btc_ret)
        var = btc_ret.var()
        
        if var == 0:
            return 1.0
        
        return cov / var
    
    def calc_rolling_volatility(self, symbol: str, timestamp, lookback: int = 168) -> float:
        """计算个币的滚动波动率"""
        loc_idx = self.returns.index.get_loc(timestamp)
        if loc_idx < lookback:
            return 1.0
        
        hist_start = loc_idx - lookback
        coin_returns = self.returns[symbol].iloc[hist_start:loc_idx].dropna()
        
        if len(coin_returns) < 20:
            return 1.0
        
        return coin_returns.std()
    
    def calc_inverse_vol_weights(self, symbols: List[str], timestamp) -> Dict[str, float]:
        """
        波动率倒数加权
        逻辑: 给高波动的币种更低的权重，防止被妖币反噬
        """
        vols = {}
        for symbol in symbols:
            vol = self.calc_rolling_volatility(symbol, timestamp)
            vols[symbol] = max(vol, 1e-6)  # 避免除零
        
        # 倒数加权
        inv_vols = {s: 1.0 / v for s, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        
        weights = {s: iv / total_inv_vol for s, iv in inv_vols.items()}
        return weights
    
    def calc_beta_neutral_hedge_ratio(self, long_symbols: List[str], 
                                       long_weights: Dict[str, float],
                                       timestamp) -> float:
        """
        计算Beta中性的BTC对冲比例
        
        目标: sum(w_i * beta_i) = hedge_ratio * 1.0 (BTC beta = 1)
        => hedge_ratio = sum(w_i * beta_i)
        """
        total_beta_exposure = 0.0
        
        for symbol in long_symbols:
            weight = long_weights.get(symbol, 0)
            beta = self.calc_rolling_beta(symbol, timestamp)
            total_beta_exposure += weight * beta
        
        # 对冲比例 = 多头Beta暴露
        hedge_ratio = total_beta_exposure
        
        # 限制对冲比例范围 (0.3 - 1.5)
        hedge_ratio = max(0.3, min(1.5, hedge_ratio))
        
        return hedge_ratio
    
    def apply_buffer_mechanism(self, new_candidates: List[str], 
                                pred: pd.Series) -> List[str]:
        """
        换仓缓冲机制
        只有当新币种的预测分显著高于当前持仓币种时才换仓
        
        逻辑: 
        - 如果当前持仓币种仍在Top N*1.5，则保留
        - 只有新币Rank提升超过阈值才替换
        """
        if not self.current_long_symbols:
            return new_candidates[:LONG_N]
        
        # 计算所有币种的当前Rank
        all_ranks = pred.rank(ascending=False, pct=True)
        
        # 评估当前持仓
        keep_symbols = []
        for symbol in self.current_long_symbols:
            if symbol in all_ranks.index:
                current_rank = all_ranks[symbol]
                # 如果仍在前15% (约Top 10-15)，保留
                if current_rank <= 0.15:
                    keep_symbols.append(symbol)
        
        # 需要补充的数量
        need_count = LONG_N - len(keep_symbols)
        
        if need_count <= 0:
            return keep_symbols[:LONG_N]
        
        # 从新候选中选择 (排除已保留的)
        new_additions = []
        for symbol in new_candidates:
            if symbol not in keep_symbols:
                # 检查是否比当前持仓显著更好
                new_rank = all_ranks.get(symbol, 1.0)
                
                # 如果新币在Top 10%，加入
                if new_rank <= 0.10:
                    new_additions.append(symbol)
                    if len(new_additions) >= need_count:
                        break
        
        final_selection = keep_symbols + new_additions
        return final_selection[:LONG_N]
        
    def generate_positions(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        生成市场中性持仓权重 - 纯截面多空策略 (修复版)
        
        策略:
        - 剔除BTC: BTC不参与排序，也不作为对冲端
        - 多头: 预测分最高的Top N个币 (等权分配)
        - 空头: 预测分最低的Bottom N个币 (等权分配)
        - 权重: 多头总仓位+1.0，空头总仓位-1.0
        - 目的: 完全对冲市场Beta，只赚取选币能力的Alpha
        
        Returns:
            positions: 合并后的净持仓
            long_positions: 多头持仓
            short_positions: 空头持仓
        """
        positions = pd.DataFrame(0.0, index=self.predictions.index, 
                                 columns=self.predictions.columns)
        long_positions = pd.DataFrame(0.0, index=self.predictions.index,
                                      columns=self.predictions.columns)
        short_positions = pd.DataFrame(0.0, index=self.predictions.index,
                                       columns=self.predictions.columns)
        
        timestamps = self.predictions.index
        last_rebalance_idx = -REBALANCE_HOURS  # 强制第一次换仓
        
        for i, ts in enumerate(timestamps):
            # 检查择时信号
            if ts not in self.timing_signals.index or self.timing_signals[ts] == 0:
                continue
            
            # 换仓频率控制 (每REBALANCE_HOURS小时换一次)
            if i - last_rebalance_idx < REBALANCE_HOURS:
                if i > 0:
                    positions.loc[ts] = positions.iloc[i-1]
                    long_positions.loc[ts] = long_positions.iloc[i-1]
                    short_positions.loc[ts] = short_positions.iloc[i-1]
                continue
            
            # 获取当前票池
            if ts not in self.universe.index:
                continue
            universe_mask = self.universe.loc[ts]
            
            # *** 关键修改: 获取预测值，剔除BTC ***
            pred = self.predictions.loc[ts]
            pred = pred[universe_mask].dropna()
            pred = pred.drop('BTCUSDT', errors='ignore')  # 剔除BTC，不参与排序
            
            if len(pred) < LONG_N * 2:  # 至少需要2N个币才能做多空
                continue
            
            # *** 纯截面多空策略 ***
            # 1. 做多: 预测分最高的Top N个币
            long_symbols = pred.nlargest(LONG_N).index.tolist()
            
            # 2. 做空: 预测分最低的Bottom N个币
            short_symbols = pred.nsmallest(LONG_N).index.tolist()
            
            if len(long_symbols) == 0 or len(short_symbols) == 0:
                continue
            
            # *** 等权分配 ***
            # 多头: 每个币权重 = 1.0 / N (总权重 = 1.0)
            long_weight = 1.0 / len(long_symbols)
            for symbol in long_symbols:
                if symbol in positions.columns:
                    positions.loc[ts, symbol] = long_weight
                if symbol in long_positions.columns:
                    long_positions.loc[ts, symbol] = long_weight
            
            # 空头: 每个币权重 = -1.0 / N (总权重 = -1.0)
            short_weight = -1.0 / len(short_symbols)
            for symbol in short_symbols:
                if symbol in positions.columns:
                    positions.loc[ts, symbol] = short_weight
                if symbol in short_positions.columns:
                    # 修复：存储负权重，与positions保持一致
                    short_positions.loc[ts, symbol] = short_weight
            
            # 更新当前持仓
            self.current_long_symbols = long_symbols
            last_rebalance_idx = i
        
        # 前向填充
        positions = positions.ffill().fillna(0)
        long_positions = long_positions.ffill().fillna(0)
        short_positions = short_positions.ffill().fillna(0)
        
        return positions, long_positions, short_positions


class BacktestEngine:
    """回测引擎 - 多空版本"""
    
    def __init__(self, returns: pd.DataFrame, positions: pd.DataFrame,
                 long_positions: pd.DataFrame, short_positions: pd.DataFrame,
                 market_index: pd.DataFrame):
        self.returns = returns
        self.positions = positions
        self.long_positions = long_positions
        self.short_positions = short_positions
        self.market_index = market_index
        
    def run_backtest(self) -> BacktestResult:
        """运行回测 (无止损)"""
        print("=" * 60)
        print("Running backtest (Long-Short Strategy, No Stop-Loss)...")
        print(f"  Max portfolio leverage: {MAX_PORTFOLIO_LEVERAGE}x")
        
        # 对齐数据
        common_idx = self.returns.index.intersection(self.positions.index)
        returns = self.returns.loc[common_idx]
        positions = self.positions.loc[common_idx].copy()
        long_pos = self.long_positions.loc[common_idx].copy()
        short_pos = self.short_positions.loc[common_idx].copy()
        
        # 计算持仓变化(用于计算交易成本)
        position_changes = positions.diff().abs()
        
        # 单边交易成本
        trade_cost_rate = TAKER_FEE + SLIPPAGE
        
        # 计算策略收益
        # 多头收益: 正权重 * 正收益
        # 空头收益: 负权重 * 正收益 = 负收益; 即做空赚反向收益
        portfolio_returns = (positions.shift(1) * returns).sum(axis=1)
        
        # 扣除交易成本
        turnover = position_changes.sum(axis=1)
        trade_costs = turnover * trade_cost_rate
        
        net_returns = portfolio_returns - trade_costs
        
        # 累计收益 (从初始资金开始)
        equity_curve = INITIAL_CAPITAL * (1 + net_returns).cumprod()
        print(f"  Initial capital: ${INITIAL_CAPITAL:,}")
        
        # 计算交易记录
        trades = self._generate_trades(positions)
        
        # 计算绩效指标
        metrics = self._calc_metrics(net_returns, equity_curve, turnover,
                                      long_pos, short_pos, returns)
        
        return BacktestResult(
            equity_curve=equity_curve,
            positions=positions,
            trades=trades,
            metrics=metrics,
            long_positions=long_pos,
            short_positions=short_pos
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
                    'direction': 'LONG' if change > 0 else 'SHORT' if change < 0 else 'CLOSE',
                    'weight_change': change
                })
        
        return pd.DataFrame(trades)
    
    def _calc_metrics(self, returns: pd.Series, equity: pd.Series, 
                      turnover: pd.Series, long_pos: pd.DataFrame,
                      short_pos: pd.DataFrame, asset_returns: pd.DataFrame) -> Dict[str, float]:
        """计算绩效指标"""
        
        # 基本收益指标
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        
        # 年化收益 (假设1小时数据)
        n_hours = len(returns)
        n_years = n_hours / (24 * 365)
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # 波动率
        ann_vol = returns.std() * np.sqrt(24 * 365)
        
        # 夏普比率
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
        ann_index_return = (1 + index_total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        info_ratio = (ann_return - ann_index_return) / tracking_error if tracking_error > 0 else 0
        
        # 多空分解
        long_returns = (long_pos.shift(1) * asset_returns).sum(axis=1).loc[returns.index]
        # 修复：short_pos已经是负权重，不需要再加负号
        short_returns = (short_pos.shift(1) * asset_returns).sum(axis=1).loc[returns.index]
        
        long_total = (1 + long_returns).prod() - 1
        short_total = (1 + short_returns).prod() - 1
        
        # 组合Beta (相对大盘)
        portfolio_beta = returns.cov(index_returns) / index_returns.var() if index_returns.var() > 0 else 0
        
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
            'long_return': long_total,
            'short_return': short_total,
            'portfolio_beta': portfolio_beta,
            'n_hours': n_hours,
            'n_years': n_years,
        }
        
        return metrics


class StrategyAssembler:
    """策略组装器 - 多空对冲版"""
    
    def __init__(self, panel_data: Dict[str, pd.DataFrame],
                 factors_normalized: Dict[str, pd.DataFrame],
                 factor_matrix: pd.DataFrame,
                 model, selected_features: List[str],
                 test_times: pd.DatetimeIndex = None):
        self.panel_data = panel_data
        self.factors_normalized = factors_normalized
        self.factor_matrix = factor_matrix
        self.model = model
        self.selected_features = selected_features
        self.test_times = test_times  # 只在测试集上回测
        
        self.timing_module: TimingModule = None
        self.predictions: pd.DataFrame = None
        self.backtest_result: BacktestResult = None
        
    def generate_predictions(self) -> pd.DataFrame:
        """生成模型预测 - 只在测试集上预测，避免未来函数"""
        print("=" * 60)
        print("Generating predictions...")
        
        # 获取选择的特征
        X = self.factor_matrix[self.selected_features]
        
        # *** 关键修改: 只在测试集时间段预测 ***
        if self.test_times is not None:
            test_mask = X.index.get_level_values('timestamp').isin(self.test_times)
            X_test = X.loc[test_mask]
            print(f"  *** 只在测试集(后30%)上回测，避免未来函数 ***")
            print(f"  Test period: {self.test_times[0]} to {self.test_times[-1]}")
        else:
            X_test = X
            print(f"  WARNING: 未指定测试集，使用全部数据(可能有未来函数)")
        
        # 预测 (支持 EnsembleTrainer 返回的 pd.Series)
        pred_values = self.model.predict(X_test)
        if isinstance(pred_values, pd.Series):
            predictions_stacked = pred_values
        else:
            predictions_stacked = pd.Series(pred_values, index=X_test.index, name='prediction')
        
        # Unstack为宽表
        predictions = predictions_stacked.unstack(level='symbol')
        
        self.predictions = predictions
        
        print(f"  Predictions shape: {predictions.shape}")
        
        return predictions
    
    def run_strategy(self) -> BacktestResult:
        """运行完整策略 - 纯截面多空策略 (修复版)"""
        print("\n" + "=" * 60)
        print("PHASE 4: STRATEGY ASSEMBLY & BACKTEST")
        print("       === 纯截面多空策略 (Cross-Sectional Long-Short) ===")
        print("=" * 60)
        
        # 择时模块 - 使用BTC价格计算4小时RSI
        print("Setting up RSI timing module (4h BTC)...")
        btc_close = self.panel_data['close'].get('BTCUSDT', None)
        self.timing_module = TimingModule(self.panel_data['market_index'], btc_close=btc_close)
        timing_signals = self.timing_module.generate_signals()
        
        # 生成预测
        self.generate_predictions()
        
        # 生成持仓 (纯截面多空: Long Top N / Short Bottom N)
        print("=" * 60)
        print("Generating Cross-Sectional Long-Short positions...")
        print(f"  *** BTC剔除: 不参与排序，不作为对冲端 ***")
        print(f"  Long: Top {LONG_N} Altcoins (Equal Weight)")
        print(f"  Short: Bottom {LONG_N} Altcoins (Equal Weight)")
        print(f"  Rebalance: Every {REBALANCE_HOURS} hours")
        
        # 修正: 使用原始收益率(returns_raw)进行回测
        raw_returns = self.panel_data['returns']
        
        # *** 纯截面多空策略: 不需要BTC收益率 ***
        pos_manager = MarketNeutralPositionManager(
            self.panel_data['universe'],
            self.predictions,
            timing_signals,
            raw_returns,
            self.panel_data['market_index']['returns'],
            self.panel_data['quote_volume'],
            btc_returns=None  # 不使用BTC对冲
        )
        positions, long_positions, short_positions = pos_manager.generate_positions()
        
        # 持仓统计 (注意: short_positions存储的是负权重)
        avg_long = (long_positions > 0).sum(axis=1).mean()
        avg_short = (short_positions < 0).sum(axis=1).mean()  # 修复: 空头权重是负数
        print(f"  Avg long positions: {avg_long:.1f}")
        print(f"  Avg short positions: {avg_short:.1f}")
        
        # 运行回测 - 必须使用原始收益率
        print(f"  *** 使用原始收益率(RAW)进行回测结算 ***")
        backtest_engine = BacktestEngine(
            raw_returns,
            positions,
            long_positions,
            short_positions,
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
        print("BACKTEST RESULTS - Cross-Sectional Long-Short (Top N / Bottom N)")
        print("=" * 60)
        
        m = self.backtest_result.metrics
        equity = self.backtest_result.equity_curve
        
        print(f"\n{'='*40}")
        print("资金变化")
        print(f"{'='*40}")
        print(f"  初始资金:        ${INITIAL_CAPITAL:>10,}")
        print(f"  最终资金:        ${equity.iloc[-1]:>10,.2f}")
        
        print(f"\n{'='*40}")
        print("收益指标")
        print(f"{'='*40}")
        print(f"  总收益率:        {m['total_return']*100:>10.2f}%")
        print(f"  年化收益率:      {m['ann_return']*100:>10.2f}%")
        print(f"  大盘收益率:      {m['index_return']*100:>10.2f}%")
        print(f"  超额收益(Alpha): {m['excess_return']*100:>10.2f}%")
        
        print(f"\n{'='*40}")
        print("多空分解 (Long Top N / Short Bottom N)")
        print(f"{'='*40}")
        print(f"  多头收益:        {m['long_return']*100:>10.2f}%")
        print(f"  空头收益:        {m['short_return']*100:>10.2f}%")
        print(f"  组合Beta:        {m['portfolio_beta']:>10.3f}")
        
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
        self.backtest_result.long_positions.to_parquet(OUTPUT_DIR / 'long_positions.parquet')
        self.backtest_result.short_positions.to_parquet(OUTPUT_DIR / 'short_positions.parquet')
        
        # 保存交易记录
        self.backtest_result.trades.to_csv(OUTPUT_DIR / 'trades.csv', index=False)
        
        # 保存指标
        metrics_df = pd.DataFrame([self.backtest_result.metrics])
        metrics_df.to_csv(OUTPUT_DIR / 'metrics.csv', index=False)
        
        # 保存预测
        self.predictions.to_parquet(OUTPUT_DIR / 'predictions.parquet')
        
        print(f"\n  Results saved to {OUTPUT_DIR}")


def run_full_pipeline():
    """运行完整的策略流水线 (辅助入口, 与 main.py 逻辑一致)"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from config import OUTPUT_DIR
    from data_auditor import DataAuditor, load_raw_data
    from data_engineering import DataEngine
    from factor_engineering import FactorEngine
    from factor_research import FactorResearcher
    from factor_validation import FactorValidator
    from model_training import ModelTrainer
    
    print("\n" + "=" * 70)
    print(" CROSS-SECTIONAL MULTI-FACTOR STRATEGY - V3 REGENESIS")
    print(" 截面多因子量化策略 - 市场中性版")
    print("=" * 70)
    
    # Phase 0: 数据审计 (闭环)
    auditor = DataAuditor(load_raw_data())
    audit_report = auditor.run_full_audit()
    (OUTPUT_DIR / 'data_audit_report.txt').write_text('\n'.join(audit_report), encoding='utf-8')
    dead_coins = getattr(auditor, 'dead_coins', [])
    
    # Phase 1: 数据工程
    data_engine = DataEngine(extra_exclude=dead_coins)
    panel_data = data_engine.run_pipeline()
    
    # Phase 2: 因子工程
    factor_engine = FactorEngine(panel_data)
    factors_normalized, factor_matrix = factor_engine.run_pipeline()
    
    # Phase 2.5a: 因子研究
    researcher = FactorResearcher(panel_data, factors_normalized, factor_matrix)
    research_report = researcher.run_research()
    
    # Phase 2.5b: 因子筛选
    validator = FactorValidator(research_report, factor_matrix)
    _, selected_features = validator.run_pipeline()
    
    # Phase 3: 模型训练
    trainer = ModelTrainer(panel_data, factors_normalized, factor_matrix, selected_features=selected_features)
    rolling_predictor, selected_features, test_times = trainer.run_pipeline()
    
    # Phase 4: 策略回测
    assembler = StrategyAssembler(
        panel_data, factors_normalized, factor_matrix,
        rolling_predictor, selected_features, test_times
    )
    result = assembler.run_strategy()
    
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    from pathlib import Path
    result = run_full_pipeline()
