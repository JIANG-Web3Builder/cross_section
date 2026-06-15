"""
Phase 4: Strategy Assembly & Backtest - V3 REGENESIS
策略组装与回测模块 
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import (
    TIMING_RSI_PERIOD, TIMING_RSI_BULL, TIMING_RSI_BEAR,
    TIMING_BULL_LONG_RATIO, TIMING_BEAR_LONG_RATIO,
    LONG_N, TOTAL_HOLD, REBALANCE_HOURS,
    MAKER_FEE, TAKER_FEE, SLIPPAGE, OUTPUT_DIR,
    BETA_LOOKBACK, STOP_LOSS_PCT, MAX_PORTFOLIO_LEVERAGE,
    INITIAL_CAPITAL,
    CRASH_PROTECT_BTC_WINDOW, CRASH_PROTECT_BTC_DROP, CRASH_PROTECT_SHORT_SCALE,
    HEDGE_SYMBOLS, HEDGE_BTC_WEIGHT, HEDGE_ETH_WEIGHT, HEDGE_RATIO,
)


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
    """
    择时模块 — BTC日线RSI overlay过滤器
    
    输出 long_ratio:
      RSI > 60 → 小幅做空BTC/ETH overlay
      RSI < 40 → 小幅做多BTC/ETH overlay
      RSI 40~60 → 不做overlay
    注意: 核心山寨 L/S 仓位保持对称, long_ratio 仅用于表达 overlay 方向
    """
    
    def __init__(self, market_index: pd.DataFrame, btc_close: pd.Series = None):
        self.market_index = market_index
        self.btc_close = btc_close
        self.signals: pd.Series = None
        self.long_ratio: pd.Series = None
        
    def calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
        
    def generate_signals(self) -> pd.Series:
        """
        生成择时信号:
        1. BTC日线RSI → 主信号 (牛/熊/中性)
        2. 中性区不做方向偏移
        
        Returns: signals (always 1, 用于兼容)
        Side effect: self.long_ratio 被设置
        """
        if self.btc_close is None:
            price = self.market_index['price']
        else:
            price = self.btc_close
        
        # === 主信号: BTC日线RSI ===
        price_daily = price.resample('1D').last().dropna()
        rsi_daily = self.calc_rsi(price_daily, TIMING_RSI_PERIOD)
        # 前向填充到小时级
        rsi = rsi_daily.reindex(price.index, method='ffill')
        
        # === 综合判断 ===
        long_ratio = pd.Series(0.5, index=price.index)  # 默认中性
        
        # RSI > 60 → 牛市
        bull_rsi = rsi > TIMING_RSI_BULL
        long_ratio[bull_rsi] = TIMING_BULL_LONG_RATIO
        
        # RSI < 40 → 熊市
        bear_rsi = rsi < TIMING_RSI_BEAR
        long_ratio[bear_rsi] = TIMING_BEAR_LONG_RATIO
        
        neutral_mask = (~bull_rsi) & (~bear_rsi)
        
        self.long_ratio = long_ratio
        self.signals = pd.Series(1, index=price.index)
        
        # 统计
        n_total = len(long_ratio.dropna())
        n_bull_rsi = bull_rsi.sum()
        n_bear_rsi = bear_rsi.sum()
        n_neutral = neutral_mask.sum()
        
        print(f"  Timing: BTC Daily RSI({TIMING_RSI_PERIOD})")
        print(f"    RSI>{TIMING_RSI_BULL} (bull): {n_bull_rsi} ({n_bull_rsi/n_total*100:.1f}%) → long_ratio={TIMING_BULL_LONG_RATIO}")
        print(f"    RSI<{TIMING_RSI_BEAR} (bear): {n_bear_rsi} ({n_bear_rsi/n_total*100:.1f}%) → long_ratio={TIMING_BEAR_LONG_RATIO}")
        print(f"    RSI neutral zone: {n_neutral} ({n_neutral/n_total*100:.1f}%) → long_ratio=0.50")
        print(f"    Current long_ratio: {long_ratio.iloc[-1]:.2f}")
        
        return self.signals


class DirectionalPositionManager:
    """
    方向性持仓管理 — 对称核心L/S + BTC/ETH反向overlay
    
    核心逻辑:
    核心山寨仓位始终保持对称:
      多头山寨: +1.0 / N per coin
      空头山寨: -1.0 / N per coin
    
    BTC/ETH overlay 由 RSI regime 控制:
      bull (long_ratio>0.5): 小仓位做空BTC/ETH
      bear (long_ratio<0.5): 小仓位做多BTC/ETH
      neutral (long_ratio=0.5): 无overlay
    
    只在96h rebalance时写入持仓, 中间ffill, 不做逐小时调整
    """
    
    def __init__(self, universe: pd.DataFrame, predictions: pd.DataFrame,
                 timing_signals: pd.Series, long_ratio: pd.Series,
                 returns: pd.DataFrame, market_returns: pd.Series,
                 volume: pd.DataFrame, btc_close: pd.Series = None):
        self.universe = universe
        self.predictions = predictions
        self.timing_signals = timing_signals
        self.long_ratio = long_ratio
        self.returns = returns
        self.market_returns = market_returns
        self.volume = volume
        self.btc_close = btc_close
    
    def _apply_overlay(self, positions, long_positions, short_positions,
                       ts, long_ratio_val: float):
        """
        写入BTC/ETH overlay — 方向与山寨net exposure相反
        牛市(long_ratio高): net多山寨 → 做空BTC/ETH对冲
        熊市(long_ratio低): net空山寨 → 做多BTC/ETH对冲
        """
        # overlay_direction: long_ratio=0.7→-0.4, long_ratio=0.3→+0.4, 0.5→0
        overlay_direction = -(long_ratio_val - 0.5) * 2.0
        overlay_direction = np.clip(overlay_direction, -1.0, 1.0)
        
        btc_w = overlay_direction * HEDGE_RATIO * HEDGE_BTC_WEIGHT
        eth_w = overlay_direction * HEDGE_RATIO * HEDGE_ETH_WEIGHT
        
        if 'BTCUSDT' in positions.columns:
            positions.loc[ts, 'BTCUSDT'] = btc_w
            long_positions.loc[ts, 'BTCUSDT'] = max(0.0, btc_w)
            short_positions.loc[ts, 'BTCUSDT'] = min(0.0, btc_w)
        if 'ETHUSDT' in positions.columns:
            positions.loc[ts, 'ETHUSDT'] = eth_w
            long_positions.loc[ts, 'ETHUSDT'] = max(0.0, eth_w)
            short_positions.loc[ts, 'ETHUSDT'] = min(0.0, eth_w)

    def _btc_drop_signal(self, ts) -> float:
        if self.btc_close is None or ts not in self.btc_close.index:
            return 0.0
        series = self.btc_close.reindex(self.btc_close.index).ffill()
        loc = series.index.get_loc(ts)
        if isinstance(loc, slice) or loc < CRASH_PROTECT_BTC_WINDOW:
            return 0.0
        prev_price = series.iloc[loc - CRASH_PROTECT_BTC_WINDOW]
        curr_price = series.iloc[loc]
        if prev_price == 0 or pd.isna(prev_price) or pd.isna(curr_price):
            return 0.0
        return curr_price / prev_price - 1.0

    def _apply_crash_protection(self, positions, long_positions, short_positions, ts):
        btc_drop = self._btc_drop_signal(ts)
        if btc_drop > CRASH_PROTECT_BTC_DROP:
            return False

        alt_short_cols = [
            c for c in positions.columns
            if c not in HEDGE_SYMBOLS and positions.loc[ts, c] < 0
        ]
        for c in alt_short_cols:
            new_w = positions.loc[ts, c] * CRASH_PROTECT_SHORT_SCALE
            positions.loc[ts, c] = new_w
            short_positions.loc[ts, c] = new_w

        # crash时强制做多BTC/ETH overlay
        self._apply_overlay(positions, long_positions, short_positions, ts, 0.0)
        return True
    
    def generate_positions(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        持仓生成 — 每96h调仓, 非对称L/S + BTC/ETH反向overlay, ffill中间
        """
        all_columns = list(self.predictions.columns)
        for hs in HEDGE_SYMBOLS:
            if hs not in all_columns:
                all_columns.append(hs)
        
        positions = pd.DataFrame(0.0, index=self.predictions.index, columns=all_columns)
        long_positions = pd.DataFrame(0.0, index=self.predictions.index, columns=all_columns)
        short_positions = pd.DataFrame(0.0, index=self.predictions.index, columns=all_columns)
        
        smoothed_preds = self.predictions.ewm(span=15, axis=0).mean()
        # 平滑long_ratio避免频繁跳变 (EMA平滑, span=48h ≈ 2天)
        smoothed_ratio = self.long_ratio.ewm(span=48, adjust=False).mean()
        
        timestamps = self.predictions.index
        last_rebalance_idx = -REBALANCE_HOURS
        rebalance_bars = []
        
        current_long: List[str] = []
        current_short: List[str] = []
        
        BUFFER_KEEP_PCT = 0.45
        
        for i, ts in enumerate(timestamps):
            if ts not in self.timing_signals.index or self.timing_signals[ts] == 0:
                continue
            
            # 非rebalance时刻: 维持上一bar持仓
            if i - last_rebalance_idx < REBALANCE_HOURS:
                if i > 0:
                    positions.loc[ts] = positions.iloc[i-1]
                    long_positions.loc[ts] = long_positions.iloc[i-1]
                    short_positions.loc[ts] = short_positions.iloc[i-1]
                    self._apply_crash_protection(positions, long_positions, short_positions, ts)
                continue
            
            if ts not in self.universe.index:
                continue
            universe_mask = self.universe.loc[ts]
            
            pred = smoothed_preds.loc[ts]
            pred = pred[universe_mask].dropna()
            for hs in HEDGE_SYMBOLS:
                pred = pred.drop(hs, errors='ignore')
            
            if len(pred) < LONG_N * 2:
                continue
            
            pred_rank = pred.rank(ascending=False, pct=True)
            
            # --- Buffer: 多头 ---
            keep_long = [s for s in current_long
                         if s in pred_rank.index and pred_rank[s] <= BUFFER_KEEP_PCT]
            need_long = LONG_N - len(keep_long)
            if need_long > 0:
                cands = pred.drop(keep_long, errors='ignore').nlargest(need_long + 3)
                long_symbols = keep_long + [s for s in cands.index if s not in keep_long][:need_long]
            else:
                long_symbols = keep_long[:LONG_N]
            
            # --- Buffer: 空头 ---
            keep_short = [s for s in current_short
                          if s in pred_rank.index and pred_rank[s] >= (1 - BUFFER_KEEP_PCT)]
            need_short = LONG_N - len(keep_short)
            if need_short > 0:
                cands = pred.drop(keep_short, errors='ignore').nsmallest(need_short + 3)
                short_symbols = keep_short + [s for s in cands.index if s not in keep_short][:need_short]
            else:
                short_symbols = keep_short[:LONG_N]
            
            if len(long_symbols) == 0 or len(short_symbols) == 0:
                continue
            
            # === 获取当前long_ratio ===
            lr = smoothed_ratio.loc[ts] if ts in smoothed_ratio.index else 0.5
            lr = np.clip(lr, 0.2, 0.8)
            
            # === 对称核心L/S权重 ===
            long_total = 1.0
            short_total = 1.0

            lw = long_total / len(long_symbols)
            sw = -short_total / len(short_symbols)
            
            for s in long_symbols:
                if s in positions.columns:
                    positions.loc[ts, s] = lw
                    long_positions.loc[ts, s] = lw
            for s in short_symbols:
                if s in positions.columns:
                    positions.loc[ts, s] = sw
                    short_positions.loc[ts, s] = sw
            
            # === BTC/ETH反向overlay ===
            self._apply_overlay(positions, long_positions, short_positions, ts, lr)
            self._apply_crash_protection(positions, long_positions, short_positions, ts)
            
            current_long = long_symbols
            current_short = short_symbols
            last_rebalance_idx = i
            rebalance_bars.append(ts)
        
        # 前向填充
        positions = positions.ffill().fillna(0)
        long_positions = long_positions.ffill().fillna(0)
        short_positions = short_positions.ffill().fillna(0)
        
        self.rebalance_bars = pd.DatetimeIndex(rebalance_bars)
        
        # 统计
        pos_diff = positions.diff().abs().sum(axis=1)
        rebal_to = pos_diff[pos_diff > 0.01]
        avg_to = rebal_to.mean() * 100 if len(rebal_to) > 0 else 0
        net_exp = positions.sum(axis=1)
        
        print(f"  Rebalance events: {len(rebalance_bars)}")
        print(f"  Avg turnover per rebalance: {avg_to:.1f}%")
        print(f"  Avg net exposure: {net_exp.mean():.3f}")
        print(f"  Net exposure range: [{net_exp.min():.3f}, {net_exp.max():.3f}]")
        
        return positions, long_positions, short_positions


class BacktestEngine:
    """回测引擎 - 多空版本"""
    
    def __init__(self, returns: pd.DataFrame, positions: pd.DataFrame,
                 long_positions: pd.DataFrame, short_positions: pd.DataFrame,
                 market_index: pd.DataFrame, rebalance_bars: pd.DatetimeIndex = None):
        self.returns = returns
        self.positions = positions
        self.long_positions = long_positions
        self.short_positions = short_positions
        self.market_index = market_index
        self.rebalance_bars = rebalance_bars
        
    def run_backtest(self) -> BacktestResult:
        """运行回测 (无止损)"""
        print("=" * 60)
        print("Running backtest (Long-Short Strategy, No Stop-Loss)...")
        print(f"  Max portfolio leverage: {MAX_PORTFOLIO_LEVERAGE}x")
        
        # 对齐数据 (时间 + 列)
        common_idx = self.returns.index.intersection(self.positions.index)
        common_cols = self.returns.columns.intersection(self.positions.columns)
        returns = self.returns.loc[common_idx, common_cols]
        positions = self.positions.loc[common_idx].reindex(columns=common_cols, fill_value=0.0)
        long_pos = self.long_positions.loc[common_idx].reindex(columns=common_cols, fill_value=0.0)
        short_pos = self.short_positions.loc[common_idx].reindex(columns=common_cols, fill_value=0.0)
        
        # 交易成本: 只对实际买卖(symbol进出)收费, 权重微调不收费
        # 逻辑: 如果一个coin从0→有持仓或有持仓→0, 才算交易
        prev_pos = positions.shift(1).fillna(0)
        # 新开仓: 之前为0, 现在非0
        new_entry = (prev_pos == 0) & (positions != 0)
        # 平仓: 之前非0, 现在为0
        new_exit = (prev_pos != 0) & (positions == 0)
        # 方向切换: 之前多现在空 或 之前空现在多
        direction_flip = (prev_pos * positions) < 0
        # 只对这些真正的交易收费
        real_trades = new_entry | new_exit | direction_flip
        position_changes = positions.diff().abs() * real_trades.astype(float)
        
        # 单边交易成本
        trade_cost_rate = TAKER_FEE + SLIPPAGE
        
        # 计算策略收益
        portfolio_returns = (positions.shift(1) * returns).sum(axis=1)
        
        # 扣除交易成本 (只在rebalance时)
        turnover = position_changes.sum(axis=1)
        trade_costs = turnover * trade_cost_rate
        
        net_returns = portfolio_returns - trade_costs
        
        # 累计收益 (从初始资金开始)
        equity_curve = INITIAL_CAPITAL * (1 + net_returns).cumprod()
        
        # 打印成本统计
        total_cost = trade_costs.sum()
        total_gross = portfolio_returns.sum()
        n_rebalances = (turnover > 0).sum()
        avg_turnover_at_rebalance = turnover[turnover > 0].mean() if n_rebalances > 0 else 0
        print(f"  Initial capital: ${INITIAL_CAPITAL:,}")
        print(f"  Rebalance events with cost: {n_rebalances}")
        print(f"  Avg turnover per rebalance: {avg_turnover_at_rebalance*100:.1f}%")
        print(f"  Total gross return: {total_gross*100:.2f}%")
        print(f"  Total trading cost: {total_cost*100:.2f}%")
        
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
        
        # 多空分解 (对齐列)
        common_asset_cols = asset_returns.columns.intersection(long_pos.columns)
        long_returns = (long_pos.shift(1)[common_asset_cols] * asset_returns[common_asset_cols]).sum(axis=1).loc[returns.index]
        short_returns = (short_pos.shift(1)[common_asset_cols] * asset_returns[common_asset_cols]).sum(axis=1).loc[returns.index]
        
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
        """运行方向性策略 — 对称核心L/S + RSI控制BTC/ETH overlay"""
        print("\n" + "=" * 60)
        print("PHASE 4: STRATEGY ASSEMBLY & BACKTEST")
        print("       === 方向性截面策略 (Symmetric Core + RSI Overlay) ===")
        print(f"       Timing: RSI>{TIMING_RSI_BULL}=Short BTC/ETH overlay")
        print(f"               RSI<{TIMING_RSI_BEAR}=Long BTC/ETH overlay")
        print("               Neutral: No overlay")
        print(f"       Overlay: {HEDGE_SYMBOLS}, Ratio={HEDGE_RATIO}")
        print("=" * 60)
        
        # 择时模块 - BTC日线RSI
        print("Setting up timing module (BTC Daily RSI)...")
        btc_close = self.panel_data['close'].get('BTCUSDT', None)
        self.timing_module = TimingModule(self.panel_data['market_index'], btc_close=btc_close)
        timing_signals = self.timing_module.generate_signals()
        long_ratio = self.timing_module.long_ratio
        
        # 生成预测
        self.generate_predictions()
        
        # 生成持仓
        print("=" * 60)
        print("Generating Directional positions...")
        print(f"  Long: Top {LONG_N} Altcoins (symmetric)")
        print(f"  Short: Bottom {LONG_N} Altcoins (symmetric)")
        print(f"  BTC/ETH overlay: RSI-driven only, ratio={HEDGE_RATIO}")
        print(f"  Rebalance: Every {REBALANCE_HOURS} hours")
        
        raw_returns = self.panel_data['returns']
        
        pos_manager = DirectionalPositionManager(
            self.panel_data['universe'],
            self.predictions,
            timing_signals,
            long_ratio,
            raw_returns,
            self.panel_data['market_index']['returns'],
            self.panel_data['quote_volume'],
            btc_close=btc_close,
        )
        positions, long_positions, short_positions = pos_manager.generate_positions()
        
        avg_long = (long_positions > 0).sum(axis=1).mean()
        avg_short = (short_positions < 0).sum(axis=1).mean()
        print(f"  Avg long positions: {avg_long:.1f}")
        print(f"  Avg short positions: {avg_short:.1f}")
        
        # 运行回测 - 必须使用原始收益率
        print(f"  *** 使用原始收益率(RAW)进行回测结算 ***")
        backtest_engine = BacktestEngine(
            raw_returns,
            positions,
            long_positions,
            short_positions,
            self.panel_data['market_index'],
            rebalance_bars=pos_manager.rebalance_bars
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
        print("BACKTEST RESULTS - Directional Cross-Sectional (BTC/ETH Hedge + Timing)")
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
    from data_engine import DataEngine
    from factor_engine import FactorEngine
    from factor_research import FactorResearcher
    from factor_selector import FactorSelector
    from model_trainer import ModelTrainer
    
    print("\n" + "=" * 70)
    print(" CROSS-SECTIONAL MULTI-FACTOR STRATEGY - FINAL")
    print("=" * 70)
    
    # Phase 0+1: 数据引擎 (审计+清洗+宇宙+指数)
    engine = DataEngine()
    panel_data = engine.run_pipeline()
    
    # Phase 2: 因子工程
    factor_engine = FactorEngine(panel_data)
    factors_normalized, factor_matrix = factor_engine.run_pipeline()
    
    # Phase 2.5a: 因子研究
    researcher = FactorResearcher(panel_data, factors_normalized, factor_matrix)
    research_report = researcher.run_research()
    
    # Phase 2.5b: 因子筛选 (ICIR加权 + R>0.7去重)
    selector = FactorSelector(research_report, factor_matrix)
    _, selected_features = selector.run_pipeline()
    
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
