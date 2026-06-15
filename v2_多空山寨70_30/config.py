"""
Cross-Sectional Strategy Configuration - V3 REGENESIS
截面量化策略配置文件 - 市场中性对冲版

特点:
- Mid-Cap Universe: Rank 11-100 腰部资产
- 滚动训练: Walk-Forward Analysis
- 波动率倒数加权 + 换仓缓冲
"""
import os
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).parent
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
LOCAL_DATA_DIR.mkdir(exist_ok=True)

DATA_DIR = LOCAL_DATA_DIR
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FACTOR_CACHE_DIR = OUTPUT_DIR / 'factors_cache'
FACTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 数据文件 (使用清洗后的数据)
DATA_FILES = {
    'open': DATA_DIR / 'cross_section_open_cleaned.parquet',
    'high': DATA_DIR / 'cross_section_high_cleaned.parquet',
    'low': DATA_DIR / 'cross_section_low_cleaned.parquet',
    'close': DATA_DIR / 'cross_section_close_cleaned.parquet',
    'volume': DATA_DIR / 'cross_section_volume_cleaned.parquet',
    'quote_volume': DATA_DIR / 'cross_section_quote_volume_cleaned.parquet',
    'returns': DATA_DIR / 'cross_section_returns_cleaned.parquet',
}

# 原始数据文件 (用于对比或回测原始收益)
DATA_FILES_RAW = {
    'open': DATA_DIR / 'cross_section_open.parquet',
    'high': DATA_DIR / 'cross_section_high.parquet',
    'low': DATA_DIR / 'cross_section_low.parquet',
    'close': DATA_DIR / 'cross_section_close.parquet',
    'volume': DATA_DIR / 'cross_section_volume.parquet',
    'quote_volume': DATA_DIR / 'cross_section_quote_volume.parquet',
    'returns': DATA_DIR / 'cross_section_returns.parquet',
}

# 动态票池参数 (Mid-Cap Universe: Rank 11-100)
LOOKBACK_HOURS_VOLUME = 24  # 计算成交额的回看窗口(小时)
MIN_HISTORY_HOURS = 720  # 最少需要720小时历史数据

# 数据清洗参数
WINSORIZE_STD = 4.0  # 缩尾处理：超过N倍标准差的值拉回
MAX_RETURN_THRESHOLD = 0.5  # 单小时最大收益率阈值
ORTHOGONALIZATION_WINDOW = 360  # 残差正交化滚动窗口 (15天)
FACTOR_VALIDATION_QUANTILES = 5
FACTOR_VALIDATION_MIN_CROSS_SECTION = 20
FACTOR_VALIDATION_MIN_TIMESTAMPS = 120
FACTOR_SELECTION_MIN_ABS_IC = 0.005
FACTOR_SELECTION_MIN_POSITIVE_RATIO = 0.52
FACTOR_SELECTION_MAX_FEATURES = 24
FACTOR_SELECTION_MIN_FEATURES = 12
FACTOR_SELECTION_MAX_CORR = 0.85

# 排除的币种(稳定币、问题币)
EXCLUDE_SYMBOLS = ['USDCUSDT', 'USTCUSDT', 'LUNAUSDT', 'FTTUSDT']

# 标签构建参数
FORWARD_HOURS = 12  # 预测未来12小时收益 (与换仓周期一致)

# ============================================================================
# 滚动训练参数 (Walk-Forward Analysis)
# ============================================================================
ROLLING_TRAIN_SIZE = 4320   # 训练窗口: 6个月 (约180天 * 24小时)
ROLLING_TEST_SIZE = 720     # 测试窗口: 1个月 (约30天 * 24小时)
ROLLING_EMBARGO = 12        # 隔离期: 12小时 (防止Label泄露，与FORWARD_HOURS匹配)

# ============================================================================
# 双模融合参数 (Dual-Model Ensemble: LGBM + CatBoost)
# ============================================================================

# LightGBM 配置 (压制过拟合版)
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 5000,           # 给足够多的空间，完全依赖 early_stopping
    'learning_rate': 0.005,         # [修改] 降慢学习率，逼迫模型多迭代几次 (0.01 -> 0.005)
    
    # 树结构限制 (最关键的修改)
    'num_leaves': 63,               # [修改] 从 63 降到 31 (甚至可以用 15)。逼迫模型做粗粒度判断。
    'max_depth': 7,                 # [修改] 从 7 降到 5。浅树泛化能力更强。
    'min_child_samples': 50,        # [修改] 降至 30，相信 early_stopping 正则化
    
    # 正则化 (适度)
    'reg_alpha': 0.1,               # [修改] L1正则，降回 0.5
    'reg_lambda': 0.5,              # [修改] L2正则，降回 0.5
    'min_split_gain': 0.01,         # [修改] 降低分裂阈值
    
    # 采样
    'subsample': 0.8,               # 稍微降低采样率，增加随机性
    'subsample_freq': 1,
    'colsample_bytree': 0.8,        # [修改] 每次只看 60% 的特征，防止某个泄露特征主导模型
    
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42
}

# CatBoost 配置 (高噪声对抗版)
CATBOOST_PARAMS = {
    'iterations': 5000,
    'learning_rate': 0.005,         # 降低学习率
    'depth': 5,                     # [修改] 降到 5。
    'l2_leaf_reg': 5,               # [修改] L2正则，降回 5，相信 early_stopping
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    
    # 增加随机性
    'rsm': 0.6,                     # (Random Subspace Method) 类似 colsample_bytree，只用 60% 特征
    'subsample': 0.7,               # 样本采样
    
    'od_type': 'Iter',
    'od_wait': 200,                 # [修改] 增加耐心，别太早停，配合低学习率
    'task_type': 'CPU',
    'thread_count': -1,
    'verbose': 0,
    'allow_writing_files': False,
    'random_seed': 2025
}
# 融合权重 (可根据验证集ICIR动态调整)
ENSEMBLE_WEIGHTS = {
    'lgbm': 0.5,
    'catboost': 0.5
}

# 择时参数 - 基于4小时BTC RSI
TIMING_RSI_PERIOD = 14      # RSI计算周期 (14个4小时K线)
TIMING_RSI_LONG = 35        # RSI低于此值，做多仓位增加
TIMING_RSI_SHORT = 65       # RSI高于此值，做空仓位增加
TIMING_RSI_RESAMPLE = '4h'  # RSI使用4小时级别

# 持仓参数 - 纯截面多空策略 (Cross-Sectional Long-Short)
LONG_N = 10          # 做多/做空数量 (Top 10 / Bottom 10)
TOTAL_HOLD = 10      # 总持仓数量
REBALANCE_HOURS = 12 # 换仓频率(12小时，降低换手率和交易成本)

# Beta中性参数
BETA_LOOKBACK = 168  # Beta计算回看周期(1周)

# 交易成本
MAKER_FEE = 0.0002  # 0.02% Maker费率
TAKER_FEE = 0.0004  # 0.04% Taker费率
SLIPPAGE = 0.001    # 0.1% 滑点

# 风控参数
STOP_LOSS_PCT = None          # 移除单标的止损
MAX_PORTFOLIO_LEVERAGE = 2.0  # 组合最大杠杆限制
