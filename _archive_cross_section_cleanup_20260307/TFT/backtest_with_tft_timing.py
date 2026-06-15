"""
TFT择时 + LGBM选股 回测系统

架构:
1. 使用 v2_多空山寨70_30 的完整LGBM流程进行选股
2. TFT信号只作为择时因子，不参与LGBM训练
3. 回测时根据TFT信号调整仓位

流程:
- Phase 1-3: 运行v2的完整流程（数据工程、因子工程、模型训练）
- Phase 4: 加载TFT择时信号
- Phase 5: 回测（LGBM选币 + TFT择时）
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config_tft import OUTPUT_DIR, PROJECT_ROOT

# 导入v2的模块
V2_DIR = PROJECT_ROOT.parent / 'v2_多空山寨70_30'
sys.path.insert(0, str(V2_DIR))

from data_engineering import DataEngine
from factor_engineering import FactorEngine
from model_training import ModelTrainer
from strategy_backtest import StrategyAssembler
from visualization import StrategyVisualizer


class TFTTimingIntegrator:
    """TFT择时信号整合器"""
    
    def __init__(self, tft_signals_path: Path = None):
        self.tft_signals_path = tft_signals_path or (OUTPUT_DIR / 'tft_signals.parquet')
        self.tft_signals = None
        
    def load_tft_signals(self) -> pd.DataFrame:
        """加载TFT择时信号"""
        print("\n" + "=" * 60)
        print("LOADING TFT TIMING SIGNALS")
        print("=" * 60)
        
        if not self.tft_signals_path.exists():
            print(f"  ⚠️ TFT信号文件不存在: {self.tft_signals_path}")
            print(f"  请先运行: python train_tft_lgbm_walkforward.py")
            return None
        
        signals = pd.read_parquet(self.tft_signals_path)
        signals['timestamp'] = pd.to_datetime(signals['timestamp'])
        
        print(f"  ✓ 加载成功: {len(signals)} 条信号")
        print(f"  时间范围: {signals['timestamp'].min()} -> {signals['timestamp'].max()}")
        print(f"  信号列: {[col for col in signals.columns if col.startswith('tft_')]}")
        
        self.tft_signals = signals
        return signals
    
    def classify_regime(self, tft_signals: pd.DataFrame) -> pd.DataFrame:
        """
        根据TFT信号分类市场状态
        
        状态定义:
        - BULL: 趋势向上 + 不确定性低 → 满仓做多
        - BEAR: 趋势向下 + 不确定性低 → 减仓或做空
        - CHOP: 不确定性中等 → 中性对冲
        - DANGER: 不确定性极高 → 空仓
        
        unc_low使用过去3个月的平均值（动态阈值）
        """
        signals = tft_signals.copy()
        signals = signals.sort_values('timestamp')
        
        # 计算滚动3个月的uncertainty平均值作为unc_low（动态阈值）
        lookback_hours = FINETUNE_CONFIG.get('unc_lookback_hours', 2160)  # 3个月
        signals['unc_low'] = signals['tft_uncertainty'].rolling(
            window=lookback_hours // 24,  # 转换为天数
            min_periods=30  # 至少需要30天数据
        ).mean()
        
        # 对于前期没有足够数据的，使用全局分位数
        global_unc_low = signals['tft_uncertainty'].quantile(0.3)
        signals['unc_low'] = signals['unc_low'].fillna(global_unc_low)
        
        # 计算其他阈值（使用全局分位数）
        unc_high = signals['tft_uncertainty'].quantile(0.7)
        unc_extreme = signals['tft_uncertainty'].quantile(0.85)
        
        # 分类逻辑（使用动态unc_low）
        def classify(row):
            unc = row['tft_uncertainty']
            unc_low_dynamic = row['unc_low']  # 使用过去3个月的平均值
            trend = row['tft_trend']
            
            if unc > unc_extreme:
                return 'DANGER'  # 极端波动，空仓
            elif unc < unc_low_dynamic and trend > 0:
                return 'BULL'    # 稳健上涨，满仓多头
            elif unc < unc_low_dynamic and trend < 0:
                return 'BEAR'    # 稳健下跌，减仓
            else:
                return 'CHOP'    # 震荡，中性对冲
        
        signals['regime'] = signals.apply(classify, axis=1)
        
        # 输出动态阈值的统计
        print(f"\n  动态unc_low统计:")
        print(f"    均值: {signals['unc_low'].mean():.4f}")
        print(f"    范围: {signals['unc_low'].min():.4f} ~ {signals['unc_low'].max():.4f}")
        
        # 统计
        regime_counts = signals['regime'].value_counts()
        print("\n  市场状态分布:")
        for regime, count in regime_counts.items():
            pct = count / len(signals) * 100
            print(f"    {regime}: {count} ({pct:.1f}%)")
        
        return signals
    
    def adjust_positions_by_regime(self, lgbm_scores: pd.DataFrame, 
                                   tft_signals: pd.DataFrame) -> pd.DataFrame:
        """
        根据TFT市场状态调整LGBM选出的持仓
        
        Args:
            lgbm_scores: LGBM预测分数 (timestamp, symbol, prediction)
            tft_signals: TFT信号 (timestamp, regime, tft_trend, etc.)
            
        Returns:
            调整后的持仓权重
        """
        print("\n" + "=" * 60)
        print("ADJUSTING POSITIONS BY TFT REGIME")
        print("=" * 60)
        
        # 合并LGBM分数和TFT信号
        tft_regime = tft_signals[['timestamp', 'regime', 'tft_trend', 'tft_uncertainty']].copy()
        
        positions = []
        
        for timestamp in lgbm_scores.index.get_level_values('timestamp').unique():
            # 获取当前时间的LGBM排名
            scores_t = lgbm_scores.loc[timestamp].sort_values('prediction', ascending=False)
            
            # 获取TFT状态
            tft_t = tft_regime[tft_regime['timestamp'] == timestamp]
            if len(tft_t) == 0:
                continue
            
            regime = tft_t.iloc[0]['regime']
            
            # 根据状态调整持仓
            if regime == 'BULL':
                # 牛市: Top 10做多，不做空
                long_symbols = scores_t.head(10).index.tolist()
                short_symbols = []
                leverage = 1.5
                
            elif regime == 'BEAR':
                # 熊市: 减少多头，增加空头
                long_symbols = scores_t.head(5).index.tolist()
                short_symbols = scores_t.tail(5).index.tolist()
                leverage = 0.5
                
            elif regime == 'CHOP':
                # 震荡: 多空对冲
                long_symbols = scores_t.head(7).index.tolist()
                short_symbols = scores_t.tail(7).index.tolist()
                leverage = 1.0
                
            else:  # DANGER
                # 危险: 空仓
                long_symbols = []
                short_symbols = []
                leverage = 0.0
            
            # 构建持仓
            for symbol in long_symbols:
                positions.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'weight': leverage / len(long_symbols) if long_symbols else 0,
                    'side': 'long',
                    'regime': regime
                })
            
            for symbol in short_symbols:
                positions.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'weight': -leverage / len(short_symbols) if short_symbols else 0,
                    'side': 'short',
                    'regime': regime
                })
        
        positions_df = pd.DataFrame(positions)
        
        if len(positions_df) > 0:
            print(f"  ✓ 生成持仓: {len(positions_df)} 条")
            print(f"\n  各状态下的平均持仓数:")
            regime_stats = positions_df.groupby('regime').size()
            for regime, count in regime_stats.items():
                avg_per_time = count / positions_df['timestamp'].nunique()
                print(f"    {regime}: {avg_per_time:.1f} 个币种")
        
        return positions_df


def main():
    """主流程"""
    print("\n" + "=" * 80)
    print("TFT择时 + LGBM选股 回测系统")
    print("=" * 80)
    print("\n架构:")
    print("  - LGBM: 独立训练，使用v2的完整因子（不含TFT）")
    print("  - TFT: 只作为择时信号，调整仓位")
    print("=" * 80)
    
    # ========== Phase 1-3: 运行v2的LGBM训练流程 ==========
    print("\n>>> Phase 1: Data Engineering (v2)")
    data_engine = DataEngine()
    panel_data = data_engine.run_pipeline()
    
    print("\n>>> Phase 2: Factor Engineering (v2)")
    factor_engine = FactorEngine(panel_data, use_cache=True)
    factors_normalized, factor_matrix = factor_engine.run_pipeline()
    
    print("\n>>> Phase 3: Model Training (v2 - Rolling LGBM)")
    trainer = ModelTrainer(panel_data, factors_normalized, factor_matrix)
    rolling_predictor, selected_features, test_times = trainer.run_pipeline()
    
    # 获取LGBM预测结果
    lgbm_predictions = rolling_predictor.predictions
    
    print(f"\n✓ LGBM训练完成")
    print(f"  预测样本: {len(lgbm_predictions):,}")
    print(f"  时间范围: {lgbm_predictions.index.get_level_values('timestamp').min()} -> "
          f"{lgbm_predictions.index.get_level_values('timestamp').max()}")
    
    # ========== Phase 4: 加载TFT择时信号 ==========
    print("\n>>> Phase 4: Loading TFT Timing Signals")
    integrator = TFTTimingIntegrator()
    tft_signals = integrator.load_tft_signals()
    
    if tft_signals is None:
        print("\n⚠️ 未找到TFT信号，使用纯LGBM策略")
        print("如需使用TFT择时，请先运行: python train_tft_lgbm_walkforward.py")
        use_tft = False
    else:
        use_tft = True
        tft_signals = integrator.classify_regime(tft_signals)
    
    # ========== Phase 5: 回测 ==========
    print("\n>>> Phase 5: Backtesting")
    
    if use_tft:
        # 使用TFT择时调整持仓
        positions = integrator.adjust_positions_by_regime(
            lgbm_predictions.to_frame('prediction'),
            tft_signals
        )
        
        # 转换为v2回测所需的格式
        # TODO: 这里需要根据v2的StrategyAssembler接口调整
        print("\n  使用TFT择时策略进行回测...")
        print("  (回测逻辑待实现)")
        
    else:
        # 纯LGBM策略
        print("\n  使用纯LGBM策略进行回测...")
        assembler = StrategyAssembler(
            panel_data=panel_data,
            rolling_predictor=rolling_predictor,
            test_times=test_times
        )
        backtest_results = assembler.run_backtest()
        
        # 可视化
        visualizer = StrategyVisualizer(backtest_results)
        visualizer.plot_all()
    
    print("\n" + "=" * 80)
    print("✓ 回测完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
