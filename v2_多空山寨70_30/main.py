import argparse
import sys
from pathlib import Path

# 确保项目路径优先
PROJECT_ROOT = Path(__file__).parent

# 本目录优先
sys.path.insert(0, str(PROJECT_ROOT))

def run_full_strategy(use_cache: bool = True):
    """运行完整策略流水线"""
    from config import OUTPUT_DIR
    from data_auditor import DataAuditor, load_raw_data
    from data_engineering import DataEngine
    from factor_engineering import FactorEngine
    from factor_research import FactorResearcher
    from factor_validation import FactorValidator
    from model_training import ModelTrainer
    from strategy_backtest import StrategyAssembler
    from visualization import StrategyVisualizer
    
    print("\n" + "=" * 70)
    print(" CROSS-SECTIONAL MULTI-FACTOR STRATEGY - V3 REGENESIS")
    print(" 截面多因子量化策略 - 市场中性版 (Long Top N / Short Bottom N)")
    print("=" * 70)
    print("\n策略特点:")
    print("  - Mid-Cap Universe: 选取Rank 11-80腰部资产")
    print("  - 数据审计(闭环) + 因子研究 + 因子筛选 + 滚动训练 + OOS回测")
    print("  - 滚动训练: Walk-Forward Analysis (6个月训练/1个月测试)")
    print("  - 市场中性: Long Top 10 Alts + Short Bottom 10 Alts")
    print("  - 波动率倒数加权: 给高波动币种更低权重")
    print("  - 换仓缓冲: 降低换手率，每12小时换仓一次")
    print("=" * 70)
    
    # Phase 0: 数据审计 (闭环 — 发现的死币/问题币传递给下游)
    print("\n>>> Starting Phase 0: Data Audit (closed-loop)")
    auditor = DataAuditor(load_raw_data())
    audit_report = auditor.run_full_audit()
    (OUTPUT_DIR / 'data_audit_report.txt').write_text('\n'.join(audit_report), encoding='utf-8')
    dead_coins = getattr(auditor, 'dead_coins', [])
    if dead_coins:
        print(f"  Audit found {len(dead_coins)} dead/illiquid coins -> will be excluded downstream")
    
    # Phase 1: 数据工程 (接收审计结果)
    print("\n>>> Starting Phase 1: Data Engineering")
    data_engine = DataEngine(extra_exclude=dead_coins)
    panel_data = data_engine.run_pipeline()
    
    # Phase 2: 因子工程
    print("\n>>> Starting Phase 2: Factor Engineering")
    factor_engine = FactorEngine(panel_data, use_cache=use_cache)
    factors_normalized, factor_matrix = factor_engine.run_pipeline()
    
    # Phase 2.5a: 因子研究 (私募级完整检验)
    print("\n>>> Starting Phase 2.5a: Factor Research")
    researcher = FactorResearcher(panel_data, factors_normalized, factor_matrix)
    research_report = researcher.run_research()
    
    # Phase 2.5b: 因子筛选 (基于研究报告)
    print("\n>>> Starting Phase 2.5b: Factor Selection")
    validator = FactorValidator(research_report, factor_matrix)
    _, selected_features = validator.run_pipeline()
    
    # Phase 3: 模型训练 (滚动窗口)
    print("\n>>> Starting Phase 3: Rolling Model Training (Walk-Forward)")
    trainer = ModelTrainer(panel_data, factors_normalized, factor_matrix, selected_features=selected_features)
    rolling_predictor, selected_features, test_times = trainer.run_pipeline()
    
    # Phase 4: 策略回测
    print("\n>>> Starting Phase 4: Strategy Assembly & Backtest")
    print(f"*** 市场中性策略: 做多强势山寨 + 做空弱势山寨 ***")
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
    print(" V3 REGENESIS PIPELINE COMPLETED SUCCESSFULLY!")
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


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Sectional Multi-Factor Strategy - V2 Long-Short'
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
    
    args = parser.parse_args()
    
    if args.visualize:
        run_visualization_only()
    else:
        run_full_strategy(use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
