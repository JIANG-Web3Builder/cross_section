"""
Cross-Sectional Multi-Factor Strategy Configuration
截面多因子量化策略 - 最终合并版

修复清单:
- 使用RAW数据，运行时清洗(避免预处理泄露)
- Embargo = 2x FORWARD_HOURS (严格时间隔离)
- 收紧收益率上限 ±15%/h (crypto合理范围)
- 成交额加权市场指数 (替代中位数)
- 相关性阈值 0.7 (从0.85收紧)
- ICIR作为因子选择权重
"""
from pathlib import Path

# ============================================================================
# 路径配置 (统一数据目录)
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
STRATEGY_ROOT = PROJECT_ROOT.parent

# 数据：统一存放在 d:\strategy\data\cross_section\
DATA_DIR = STRATEGY_ROOT / "data" / "cross_section"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FACTOR_CACHE_DIR = OUTPUT_DIR / "factors_cache"
FACTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 原始数据文件 (不使用预清洗版本，运行时清洗)
DATA_FILES = {
    'open':         DATA_DIR / 'cross_section_open.parquet',
    'high':         DATA_DIR / 'cross_section_high.parquet',
    'low':          DATA_DIR / 'cross_section_low.parquet',
    'close':        DATA_DIR / 'cross_section_close.parquet',
    'volume':       DATA_DIR / 'cross_section_volume.parquet',
    'quote_volume': DATA_DIR / 'cross_section_quote_volume.parquet',
    'returns':      DATA_DIR / 'cross_section_returns.parquet',
}

# ============================================================================
# 排除币种
# ============================================================================
EXCLUDE_SYMBOLS = [
    'USDCUSDT', 'USTCUSDT', 'LUNAUSDT', 'FTTUSDT',
    'SRUSDUSDT', 'DAIUSDT', 'TUSDUSDT', 'BUSDUSDT',  # 稳定币
]

# ============================================================================
# 数据清洗参数 (量化私募标准)
# ============================================================================
MAX_RETURN_THRESHOLD = 0.15       # 单小时最大收益率 ±15% (crypto合理范围)
WINSORIZE_STD = 3.5               # 截面缩尾: ±3.5倍标准差
MIN_HOURLY_QUOTE_VOLUME = 50000   # 最低小时成交额 $50K (排除僵尸币)
FFILL_LIMIT = 6                   # 前向填充最多6小时 (超过视为停牌)
MIN_DATA_COVERAGE = 0.85          # 最低数据覆盖率 85%
DEAD_COIN_HOURS = 48              # 连续48小时零成交/零变动视为死币
DELIST_BUFFER_HOURS = 72          # 退市前72小时踢出宇宙

# ============================================================================
# 动态票池参数
# ============================================================================
LOOKBACK_HOURS_VOLUME = 24        # 成交额回看窗口
MIN_HISTORY_HOURS = 720           # 最少720小时历史
UNIVERSE_TOP_EXCLUDE = 10         # 排除成交额Top N (大票)
UNIVERSE_BOTTOM_RANK = 100        # 选取Rank 11-100

# ============================================================================
# 标签与正交化
# ============================================================================
FORWARD_HOURS = 12                # 预测未来12小时收益
REBALANCE_HOURS = 96              # 换仓频率: 每96小时 (参数扫描最优)
ORTHOGONALIZATION_WINDOW = 360    # BTC正交化滚动窗口

# ============================================================================
# 因子研究与筛选参数
# ============================================================================
FACTOR_VALIDATION_QUANTILES = 5
FACTOR_VALIDATION_MIN_CROSS_SECTION = 20
FACTOR_VALIDATION_MIN_TIMESTAMPS = 120

# 因子筛选 (ICIR加权 + R>0.7去重)
FACTOR_SELECTION_MIN_ABS_IC = 0.005
FACTOR_SELECTION_MIN_ICIR = 0.10         # crypto弱信号环境下适当放宽
FACTOR_SELECTION_MIN_POSITIVE_RATIO = 0.52
FACTOR_SELECTION_MAX_FEATURES = 20
FACTOR_SELECTION_MIN_FEATURES = 8
FACTOR_SELECTION_MAX_CORR = 0.70        # 从0.85收紧到0.70

# ============================================================================
# 滚动训练参数
# ============================================================================
ROLLING_TRAIN_SIZE = 4320         # 训练窗口: 6个月
ROLLING_TEST_SIZE = 720           # 测试窗口: 1个月
ROLLING_EMBARGO = 24              # 隔离期: 2x FORWARD_HOURS = 24小时

# ============================================================================
# 模型参数
# ============================================================================
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 3000,
    'learning_rate': 0.005,
    'num_leaves': 31,
    'max_depth': 5,
    'min_child_samples': 100,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'min_split_gain': 0.01,
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree': 0.6,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42
}

CATBOOST_PARAMS = {
    'iterations': 3000,
    'learning_rate': 0.005,
    'depth': 5,
    'l2_leaf_reg': 10,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'rsm': 0.6,
    'subsample': 0.7,
    'od_type': 'Iter',
    'od_wait': 200,
    'task_type': 'CPU',
    'thread_count': -1,
    'verbose': 0,
    'allow_writing_files': False,
    'random_seed': 2025
}

ENSEMBLE_WEIGHTS = {
    'lgbm': 0.5,
    'catboost': 0.5
}

# ============================================================================
# 择时参数 (BTC日线RSI overlay过滤器)
# ============================================================================
# 主信号: BTC日线RSI
#   RSI > 60 → 保持对称 alt L/S 核心 + 小仓做空BTC/ETH
#   RSI < 40 → 保持对称 alt L/S 核心 + 小仓做多BTC/ETH
#   RSI 40~60 → 保持对称 alt L/S 核心 + 无overlay
TIMING_RSI_PERIOD = 14                       # RSI周期 (日线)
TIMING_RSI_BULL = 60                         # RSI牛市阈值
TIMING_RSI_BEAR = 40                         # RSI熊市阈值
TIMING_BULL_LONG_RATIO = 0.55                # 牛市: 多头占总权重55%, 空头占45% (net=+0.2)
TIMING_BEAR_LONG_RATIO = 0.45                # 熊市: 多头占总权重45%, 空头占55% (net=-0.2)

# ============================================================================
# 持仓与交易
# ============================================================================
LONG_N = 10
TOTAL_HOLD = 10
INITIAL_CAPITAL = 100000
BETA_LOOKBACK = 168

# 交易成本 (保守)
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
SLIPPAGE = 0.001

# 风控
STOP_LOSS_PCT = None
MAX_PORTFOLIO_LEVERAGE = 2.0

# ============================================================================
# Crash保护参数
# ============================================================================
CRASH_PROTECT_BTC_WINDOW = 12                # BTC回看窗口 (小时)
CRASH_PROTECT_BTC_DROP = -0.05               # BTC 12h 跌幅超过 -5% 时启动保护
CRASH_PROTECT_SHORT_SCALE = 0.5              # 启动保护时 short gross 缩到 50%

# ============================================================================
# BTC/ETH Overlay参数
# ============================================================================
HEDGE_SYMBOLS = ['BTCUSDT', 'ETHUSDT']       # overlay标的
HEDGE_BTC_WEIGHT = 0.6                       # BTC占overlay 60%
HEDGE_ETH_WEIGHT = 0.4                       # ETH占overlay 40%
HEDGE_RATIO = 0.2                            # overlay总仓位 (占总权重的比例, 小仓位)
