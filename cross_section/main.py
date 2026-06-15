"""
Cross-Sectional Multi-Factor Strategy - 最终合并版
截面多因子量化策略 - 市场中性 (Long Top N / Short Bottom N)

修复清单:
- 使用RAW数据，运行时清洗 (避免预处理泄露)
- 成交额加权市场指数 (替代中位数，修复IC虚高)
- Embargo = 2x FORWARD_HOURS = 24h (严格时间隔离)
- ICIR加权因子筛选 + R>0.7相关性去重
- 量化私募级数据清洗 (±15%/h cap, 流动性过滤, 存活偏差)
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_full_strategy(use_cache: bool = True):
    """运行完整策略流水线"""
    from config import OUTPUT_DIR
    from data_engine import DataEngine
    from factor_engine import FactorEngine
    from factor_research import FactorResearcher
    from factor_selector import FactorSelector
    from model_trainer import ModelTrainer
    from backtester import StrategyAssembler
    from visualizer import StrategyVisualizer

    print("\n" + "=" * 70)
    print(" CROSS-SECTIONAL MULTI-FACTOR STRATEGY - FINAL")
    print(" 截面多因子量化策略 - 市场中性版")
    print("=" * 70)
    print("\n修复项:")
    print("  - 使用RAW数据，运行时清洗 (消除预处理泄露)")
    print("  - 成交额加权市场指数 (修复IC虚高)")
    print("  - Embargo=24h=2x FORWARD_HOURS (严格时间隔离)")
    print("  - ICIR加权因子筛选 + R>0.7相关性去重")
    print("  - 收益率±15%/h硬上限 + 流动性过滤 + 存活偏差处理")
    print("=" * 70)

    # Phase 0+1: 数据引擎 (审计+清洗+宇宙+指数 一体化)
    print("\n>>> Phase 0+1: Data Engine")
    engine = DataEngine()
    panel_data = engine.run_pipeline()

    # Phase 2: 因子工程
    print("\n>>> Phase 2: Factor Engineering")
    factor_engine = FactorEngine(panel_data, use_cache=use_cache)
    factors_normalized, factor_matrix = factor_engine.run_pipeline()

    # Phase 2.5a: 因子研究 (私募级完整检验)
    print("\n>>> Phase 2.5a: Factor Research")
    researcher = FactorResearcher(panel_data, factors_normalized, factor_matrix)
    research_report = researcher.run_research()

    # Phase 2.5b: 因子筛选 (ICIR加权 + R>0.7去重)
    print("\n>>> Phase 2.5b: Factor Selection")
    selector = FactorSelector(research_report, factor_matrix)
    _, selected_features = selector.run_pipeline()

    # Phase 3: 模型训练 (滚动窗口, embargo=24h)
    print("\n>>> Phase 3: Rolling Model Training")
    trainer = ModelTrainer(panel_data, factors_normalized, factor_matrix,
                           selected_features=selected_features)
    rolling_predictor, selected_features, test_times = trainer.run_pipeline()

    # Phase 4: 策略回测
    print("\n>>> Phase 4: Strategy Assembly & Backtest")
    assembler = StrategyAssembler(
        panel_data, factors_normalized, factor_matrix,
        rolling_predictor, selected_features, test_times
    )
    result = assembler.run_strategy()

    # 可视化
    print("\n>>> Generating Visualizations")
    viz = StrategyVisualizer(result)
    viz.market_index = panel_data['market_index']
    viz.feature_importance = trainer.feature_importance
    viz.generate_report()
    viz.print_summary()

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return result


def run_visualization_only():
    from visualizer import StrategyVisualizer
    print("\n>>> Loading saved results and generating visualizations")
    viz = StrategyVisualizer()
    viz.load_results()
    viz.generate_report()
    viz.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Sectional Multi-Factor Strategy - Final'
    )
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Only generate visualization from saved results')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable factor caching, recalculate all factors')

    args = parser.parse_args()

    if args.visualize:
        run_visualization_only()
    else:
        run_full_strategy(use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
