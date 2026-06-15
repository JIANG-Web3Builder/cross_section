"""
Cross-Sectional Multi-Factor Strategy
截面多因子量化策略 - 主入口

Usage:
    python main.py                  # 运行完整流水线 (使用因子缓存)
    python main.py --no-cache       # 不使用缓存，重新计算所有因子
    python main.py --clear-cache    # 清除因子缓存
    python main.py --visualize      # 仅生成可视化报告
"""
import argparse
import sys
from pathlib import Path

# 确保项目路径在sys.path中
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_full_strategy(use_cache: bool = True):
    """运行完整策略流水线"""
    from data_engineering import DataEngine
    from factor_engineering import FactorEngine
    from model_training import ModelTrainer
    from strategy_backtest import StrategyAssembler
    from visualization import StrategyVisualizer
    
    print("\n" + "=" * 70)
    print(" CROSS-SECTIONAL MULTI-FACTOR STRATEGY")
    print(" 截面多因子量化策略")
    print("=" * 70)
    
    # Phase 1: 数据工程
    print("\n>>> Starting Phase 1: Data Engineering")
    data_engine = DataEngine()
    panel_data = data_engine.run_pipeline()
    
    # Phase 2: 因子工程 (支持缓存)
    print("\n>>> Starting Phase 2: Factor Engineering")
    factor_engine = FactorEngine(panel_data, use_cache=use_cache)
    factors_normalized, factor_matrix = factor_engine.run_pipeline()
    
    # Phase 3: 模型训练
    print("\n>>> Starting Phase 3: Model Training")
    trainer = ModelTrainer(panel_data, factors_normalized, factor_matrix)
    model, selected_features = trainer.run_pipeline()
    
    # Phase 4: 策略回测
    print("\n>>> Starting Phase 4: Strategy Assembly & Backtest")
    assembler = StrategyAssembler(
        panel_data, factors_normalized, factor_matrix,
        model, selected_features
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
    """仅运行可视化"""
    from visualization import StrategyVisualizer
    
    print("\n>>> Loading saved results and generating visualizations")
    viz = StrategyVisualizer()
    viz.load_results()
    viz.generate_report()
    viz.print_summary()


def clear_factor_cache():
    """清除因子缓存"""
    from factor_engineering import FACTOR_CACHE_DIR
    import shutil
    
    if FACTOR_CACHE_DIR.exists():
        count = len(list(FACTOR_CACHE_DIR.glob('*.parquet')))
        shutil.rmtree(FACTOR_CACHE_DIR)
        FACTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cleared {count} cached factors from {FACTOR_CACHE_DIR}")
    else:
        print("No cache to clear")


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Sectional Multi-Factor Strategy'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Only generate visualization from saved results'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable factor caching, recalculate all factors'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached factors and exit'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run only specific phase (1-4)'
    )
    
    args = parser.parse_args()
    
    if args.clear_cache:
        clear_factor_cache()
    elif args.visualize:
        run_visualization_only()
    else:
        run_full_strategy(use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
