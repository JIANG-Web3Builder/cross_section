"""
快速重跑 Phase 4 回测 (使用已缓存的模型预测结果)
用于验证交易成本修复后的效果
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import json

from config import OUTPUT_DIR, DATA_FILES, FORWARD_HOURS
from data_engine import DataEngine
from model_trainer import RollingPredictor
from backtester import StrategyAssembler
from visualizer import StrategyVisualizer

print("=" * 70)
print(" QUICK RE-RUN: Phase 4 Only (Backtest + Visualization)")
print(" Using cached predictions from previous run")
print("=" * 70)

# 1. 重建 data engine (需要 panel_data)
print("\n>>> Loading data...")
engine = DataEngine()
panel_data = engine.run_pipeline()

# 2. 加载已缓存的预测结果
print("\n>>> Loading cached predictions and features...")
rolling_pred_df = pd.read_parquet(OUTPUT_DIR / 'rolling_predictions.parquet')
predictions = rolling_pred_df['prediction']

with open(OUTPUT_DIR / 'selected_features.json') as f:
    selected_features = json.load(f)

test_times = predictions.index.get_level_values('timestamp').unique().sort_values()

# 3. 创建 RollingPredictor
rolling_predictor = RollingPredictor(predictions, selected_features)

# 4. 加载 factor_matrix (需要给 StrategyAssembler)
print(">>> Loading factor matrix...")
factor_matrix = pd.read_parquet(OUTPUT_DIR / 'factor_matrix.parquet')

# 5. 运行 Phase 4
print("\n>>> Phase 4: Strategy Assembly & Backtest")
assembler = StrategyAssembler(
    panel_data, {}, factor_matrix,
    rolling_predictor, selected_features, test_times
)
result = assembler.run_strategy()

# 6. 可视化
print("\n>>> Generating Visualizations")
viz = StrategyVisualizer(result)
viz.market_index = panel_data['market_index']
try:
    viz.feature_importance = pd.read_csv(OUTPUT_DIR / 'feature_importance.csv')
except:
    pass
viz.generate_report()
viz.print_summary()

print("\n" + "=" * 70)
print(" RE-RUN COMPLETED!")
print("=" * 70)
