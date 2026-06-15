"""测试对称50/50 + 无overlay, 验证代码是否正确复现baseline"""
import sys
sys.path.insert(0, '.')

# Override config before importing anything else
import config
config.TIMING_BULL_LONG_RATIO = 0.50
config.TIMING_BEAR_LONG_RATIO = 0.50
config.HEDGE_RATIO = 0.0

import pandas as pd
import json
from data_engine import DataEngine
from model_trainer import RollingPredictor
from backtester import StrategyAssembler

engine = DataEngine()
panel_data = engine.run_pipeline()

rolling_pred_df = pd.read_parquet(config.OUTPUT_DIR / 'rolling_predictions.parquet')
predictions = rolling_pred_df['prediction']
with open(config.OUTPUT_DIR / 'selected_features.json') as f:
    selected_features = json.load(f)
test_times = predictions.index.get_level_values('timestamp').unique().sort_values()
rolling_predictor = RollingPredictor(predictions, selected_features)
factor_matrix = pd.read_parquet(config.OUTPUT_DIR / 'factor_matrix.parquet')

assembler = StrategyAssembler(
    panel_data, {}, factor_matrix,
    rolling_predictor, selected_features, test_times
)
result = assembler.run_strategy()

m = result.metrics
print(f"\n=== SYMMETRIC BASELINE CHECK ===")
print(f"Total Return: {m['total_return']*100:.2f}%")
print(f"Sharpe: {m['sharpe_ratio']:.3f}")
print(f"Max DD: {m['max_drawdown']*100:.2f}%")
print(f"Turnover: {m['avg_turnover']*100:.2f}%")
