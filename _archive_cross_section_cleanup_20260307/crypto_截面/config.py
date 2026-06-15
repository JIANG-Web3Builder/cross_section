"""
Cross-Sectional Strategy Configuration
截面量化策略配置文件
"""
import os
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 数据文件
DATA_FILES = {
    'open': DATA_DIR / 'cross_section_open.parquet',
    'high': DATA_DIR / 'cross_section_high.parquet',
    'low': DATA_DIR / 'cross_section_low.parquet',
    'close': DATA_DIR / 'cross_section_close.parquet',
    'volume': DATA_DIR / 'cross_section_volume.parquet',
    'quote_volume': DATA_DIR / 'cross_section_quote_volume.parquet',
    'returns': DATA_DIR / 'cross_section_returns.parquet',
}

# 动态票池参数
UNIVERSE_TOP_N = 50  # 每个时间点保留成交额Top N的币种
LOOKBACK_HOURS_VOLUME = 24  # 计算成交额的回看窗口(小时)
MIN_HISTORY_HOURS = 720  # 最少需要720小时历史数据

# 数据清洗参数
WINSORIZE_STD = 3.0  # 缩尾处理：超过N倍标准差的值拉回
MAX_RETURN_THRESHOLD = 0.5  # 单小时最大收益率阈值

# 排除的币种(稳定币、问题币)
EXCLUDE_SYMBOLS = ['USDCUSDT', 'USTCUSDT', 'LUNAUSDT', 'FTTUSDT']

# 因子计算参数
FACTOR_PARAMS = {
    # 动量类
    'rsi_short': 24,      # RSI短周期(1天)
    'rsi_long': 168,      # RSI长周期(1周)
    'slope_period': 24,   # 趋势斜率周期
    
    # 反转类
    'williams_period': 24,
    'bias_period': 24,
    
    # 波动率类
    'atr_period': 24,
    'std_period': 24,
    
    # 量价类
    'vwap_period': 24,
    'pvt_period': 24,
}

# 标签构建参数
FORWARD_HOURS = 24  # 预测未来24小时收益

# 模型参数
MODEL_PARAMS = {
    'objective': 'regression',
    'metric': 'l2',
    'max_depth': 5,
    'learning_rate': 0.02,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 500,
    'early_stopping_rounds': 50,
    'verbose': -1,
}

# 训练参数
TRAIN_RATIO = 0.7  # 70%训练，30%验证

# 择时参数
TIMING_MA_PERIOD = 168  # 大盘MA周期(1周)
TIMING_VOL_PERIOD = 24  # 波动率计算周期
TIMING_VOL_QUANTILE = 0.90  # 波动率危险阈值分位数

# 持仓参数
TOP_N_HOLD = 5  # 持仓Top N
REBALANCE_HOURS = 4  # 换仓频率(小时)

# 交易成本
MAKER_FEE = 0.0002  # 0.02% Maker费率
TAKER_FEE = 0.0004  # 0.04% Taker费率
SLIPPAGE = 0.0005   # 0.05% 滑点
