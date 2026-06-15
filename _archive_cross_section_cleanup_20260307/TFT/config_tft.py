"""
TFT-LGBM Hybrid System Configuration
Production-Ready Configuration for Crypto Quantitative Trading

Architecture:
- TFT: Macro regime prediction (Beta & Risk)
- LGBM: Cross-sectional alpha selection
- Training: 2021-2023 base + 2024-2026 rolling fine-tune
"""
from pathlib import Path
import json
import numpy as np

# ============================================================================
# Path Configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
FACTOR_DIR = PROJECT_ROOT / "factor"
FINAL_DIR = PROJECT_ROOT / "final_df"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

# Create directories
for dir_path in [FINAL_DIR, OUTPUT_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Data Configuration
# ============================================================================
# Time range
START_DATE = "2021-01-01"
BASE_TRAIN_END = "2024-01-01"  # Base training period: 2021-2023
ROLLING_START = "2024-01-01"   # Rolling fine-tune starts from 2024
END_DATE = "2026-01-01"

# Resampling frequency
RESAMPLE_FREQ = "1H"  # Force 1-hour frequency

# ============================================================================
# TFT Data Engineering Configuration
# ============================================================================
# Input files in factor directory
FACTOR_FILES = {
    'btc_factors': 'BTCUSDT_factors.parquet',
    'eth_factors': 'ETHUSDT_factors.parquet',
    'btc_dominance': 'btc_dominance_1h.parquet',
    'btc_spx_corr': 'btc_spx_correlation.parquet',
    'known_time': 'known_time.parquet',
}

# Column prefix mapping
COLUMN_PREFIX = {
    'btc_factors': 'btc_',
    'eth_factors': 'eth_',
    'btc_dominance': '',  # Keep original names
    'btc_spx_corr': '',
    'known_time': '',
}

# Target construction - Triple Barrier Classification
TARGET_FORWARD_HOURS = 24  # Predict 24-hour forward barrier hits
TARGET_NAME = 'target_24h'  # Will contain class labels: 0=Hold, 1=Long, 2=Short
FEATURE_SELECTION_FILE = OUTPUT_DIR / 'selected_factors.json'

# Missing value handling
FFILL_LIMIT = 3  # Forward fill limit for minor misalignments

CANDIDATE_FACTOR_FEATURES = [
    'log_return_1h', 'log_return_4h', 'log_return_24h', 'log_return_168h',
    'volatility_24h', 'volatility_168h', 'volatility_term_structure',
    'volume_change_ratio', 'momentum_gap_24h_168h', 'return_skew_24h',
    'amihud_illiquidity_24h', 'volume_volatility_24h', 'price_range_24h',
    'price_efficiency_24h'
]

DERIVATIVE_FEATURES = [
    'funding_rate',
    'open_interest_value',
    'top_trader_ls_ratio',
    'taker_buy_sell_ratio',
]

MACRO_FEATURES = [
    'btc_dominance_change',
    'spx_correlation_rolling_7d',
    'eth_btc_ratio_change',
]

FACTOR_RESEARCH_CONFIG = {
    'forward_return_hours': TARGET_FORWARD_HOURS,
    'quantiles': 3,
    'min_assets_per_timestamp': 3,
    'min_coverage': 0.60,
    'min_abs_rank_ic': 0.02,
    'min_abs_rank_ic_ir': 0.10,
    'min_sign_consistency': 0.50,
    'max_factor_correlation': 0.85,
    'top_n_factors': 8,
}

def load_selected_factor_features():
    default_features = CANDIDATE_FACTOR_FEATURES.copy()
    if not FEATURE_SELECTION_FILE.exists():
        return default_features

    try:
        payload = json.loads(FEATURE_SELECTION_FILE.read_text(encoding='utf-8'))
    except Exception:
        return default_features

    selected = payload.get('selected_factors', default_features)
    selected = [feature for feature in selected if feature in default_features]
    return selected or default_features

def load_selected_factor_directions():
    if not FEATURE_SELECTION_FILE.exists():
        return {feature: 1 for feature in CANDIDATE_FACTOR_FEATURES}

    try:
        payload = json.loads(FEATURE_SELECTION_FILE.read_text(encoding='utf-8'))
    except Exception:
        return {feature: 1 for feature in CANDIDATE_FACTOR_FEATURES}

    directions = payload.get('factor_directions', {})
    return {
        feature: int(directions.get(feature, 1))
        for feature in CANDIDATE_FACTOR_FEATURES
    }

def build_time_varying_unknown_reals():
    return load_selected_factor_features() + DERIVATIVE_FEATURES + MACRO_FEATURES

def refresh_tft_variables():
    TFT_VARIABLES['time_varying_unknown_reals'] = build_time_varying_unknown_reals()
    return TFT_VARIABLES

# ============================================================================
# TFT Model Configuration
# ============================================================================
TFT_CONFIG = {
    # Architecture
    'max_encoder_length': 336,       # Look back 14 days (336 hours) - 从168h降级避免信息稀释
    'max_prediction_length': 24,    # Predict 24 hours ahead
    'hidden_size': 96,               # Control complexity
    'lstm_layers': 4,
    'dropout': 0.3,                  # Strong regularization
    'attention_head_size': 4,
    'hidden_continuous_size': 16,
    
    # Training
    'learning_rate': 1e-3,
    'batch_size': 128,
    'max_epochs': 50,
    'gradient_clip_val': 0.1,
    'patience': 10,  # Early stopping patience
    
    # Loss function - CrossEntropy for Triple Barrier Classification
    'loss': 'CrossEntropy',
    'output_size': 3,  # 3 classes: 0=Hold, 1=Long, 2=Short
    
    # Optimizer
    'optimizer': 'ranger',  # Ranger optimizer (RAdam + Lookahead)
    'reduce_on_plateau_patience': 4,
}

# TFT Variable Groups (must match data columns)
TFT_VARIABLES = {
    'static_categoricals': ['symbol'],
    
    'time_varying_known_reals': [
        'hour_sin', 'hour_cos', 'day_of_week'
    ],
    
    'time_varying_unknown_reals': build_time_varying_unknown_reals(),
    
    'target': TARGET_NAME,
}

# Base training configuration
BASE_TRAIN_CONFIG = {
    'learning_rate': 1e-3,
    'max_epochs': 50,
    'batch_size': 64,
}

# Rolling fine-tune configuration (3个月训练，预测1周)
FINETUNE_CONFIG = {
    'learning_rate': 1e-4,  # 10x lower than base training
    'max_epochs': 3,         # Quick adaptation
    'batch_size': 64,
    'train_window_hours': 4320,    # 6 months fine-tuning window
    'test_window_hours': 24,       # Daily update
    'embargo_hours': 0,
    'unc_lookback_hours': 2160,   # 3个月回看期 (用于计算unc_low)
}

# ============================================================================
# TFT Signal Extraction Configuration
# ============================================================================
# Extract signals from TFT predictions
SIGNAL_CONFIG = {
    'trend': 'p50',           # Use median (P50) as trend signal
    'uncertainty': 'iqr',     # Inter-Quantile Range (P90 - P10) as uncertainty
    'upside_risk': 'p95',     # 95th percentile
    'downside_risk': 'p05',   # 5th percentile
}

# ============================================================================
# LGBM Integration Configuration
# ============================================================================
# Reference existing LGBM system
LGBM_REFERENCE_DIR = PROJECT_ROOT.parent / "v2_多空山寨70_30"

# TFT signal injection
TFT_SIGNAL_FEATURES = [
    'tft_trend',           # P50 prediction
    'tft_uncertainty',     # P90 - P10
    'tft_upside',          # P95
    'tft_downside',        # P05
]

# Feature interaction (critical for performance boost)
INTERACTION_FEATURES = [
    ('log_return_168h', 'tft_trend', 'inter_mom_trend'),           # Momentum(168h) * Trend
    ('volatility_168h', 'tft_uncertainty', 'inter_vol_risk'),      # Vol(168h) / Uncertainty
    ('log_return_24h', 'tft_trend', 'inter_ret_trend'),            # Return(24h) * Trend
    ('volume_change_ratio', 'tft_uncertainty', 'inter_vol_regime'), # Volume Change vs Risk
]

# LGBM Parameters (inherit from v2 but with adjustments)
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 5000,
    'learning_rate': 0.005,
    
    # Tree structure
    'num_leaves': 31,
    'max_depth': 5,
    'min_child_samples': 500,
    
    # Regularization
    'reg_alpha': 10,
    'reg_lambda': 10,
    'min_split_gain': 0.1,
    
    # Sampling
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree': 0.6,
    
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42
}

# Rolling training for LGBM (synchronized with TFT)
LGBM_ROLLING_CONFIG = {
    'train_window_hours': 4320,  # 6 months
    'test_window_hours': 720,    # 1 month
    'embargo_hours': 48,         # Prevent label leakage
}

# ============================================================================
# Strategy Execution Configuration
# ============================================================================
# Regime Classification Thresholds
REGIME_THRESHOLDS = {
    'uncertainty_low': 0.3,      # Below this = low uncertainty
    'uncertainty_high': 0.7,     # Above this = high uncertainty
    'trend_positive': 0.0,       # Above this = bullish
    'trend_negative': 0.0,       # Below this = bearish
    'uncertainty_extreme': 0.85, # Extreme volatility threshold
}

# Regime definitions
REGIMES = {
    'S1_BULL': {
        'name': 'Bull Market',
        'condition': 'uncertainty < low AND trend > 0',
        'action': 'Full long, top selection',
        'leverage': 2.0,
        'long_n': 10,
        'short_n': 0,
    },
    'S2_BEAR': {
        'name': 'Bear Market',
        'condition': 'uncertainty < low AND trend < 0',
        'action': 'Net short or cash',
        'leverage': 1.0,
        'long_n': 0,
        'short_n': 5,
    },
    'S3_CHOP': {
        'name': 'Choppy Market',
        'condition': 'low < uncertainty < high',
        'action': 'Neutral hedge (long strong, short weak)',
        'leverage': 1.0,
        'long_n': 5,
        'short_n': 5,
    },
    'S4_DANGER': {
        'name': 'Dangerous Market',
        'condition': 'uncertainty > high',
        'action': 'Cash (circuit breaker)',
        'leverage': 0.0,
        'long_n': 0,
        'short_n': 0,
    },
    'S5_PANIC': {
        'name': 'Panic Capitulation',
        'condition': 'uncertainty extreme AND capitulation signal',
        'action': 'Left-side buy BTC only',
        'leverage': 1.0,
        'long_n': 1,  # Only BTC
        'short_n': 0,
    },
}

# Smart capitulation detection
CAPITULATION_CONFIG = {
    'uncertainty_threshold': 0.85,
    'rsi_threshold': 25,
    'volume_spike_threshold': 2.0,  # 2x average volume
    'lookback_hours': 24,
}

# Portfolio construction
PORTFOLIO_CONFIG = {
    'default_long_n': 5,
    'default_short_n': 5,
    'max_position_pct': 0.15,  # Max 15% per coin
    'rebalance_hours': 12,      # Rebalance every 12 hours
}

# ============================================================================
# Risk Management Configuration
# ============================================================================
RISK_CONFIG = {
    # Position limits
    'max_single_position': 0.15,    # 15% NAV per coin
    'max_leverage': {
        'S1_BULL': 1.5,             # Reduced from 2.0
        'S2_BEAR': 1.0,
        'S3_CHOP': 1.0,
        'S4_DANGER': 0.0,
        'S5_PANIC': 1.0,
    },
    
    # Stop loss
    'portfolio_stop_loss': 0.15,    # Relaxed to 15% to avoid whipsaw
    'cooldown_hours': 24,           # Cooldown period after stop loss
    
    # Volatility scaling
    'vol_target': 0.50,             # Increased to 50% to match crypto reality
    'vol_lookback_hours': 168,      # 1 week lookback for vol calculation
}

# ============================================================================
# Backtest Configuration
# ============================================================================
BACKTEST_CONFIG = {
    'initial_capital': 1_000_000,   # $1M initial capital
    'maker_fee': 0.0002,            # 0.02% maker fee
    'taker_fee': 0.0004,            # 0.04% taker fee
    'slippage': 0.001,              # 0.1% slippage
    'min_trade_size': 100,          # Minimum $100 per trade
}

# ============================================================================
# Logging and Monitoring
# ============================================================================
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': OUTPUT_DIR / 'tft_lgbm.log',
}

# Save important metrics
SAVE_METRICS = [
    'tft_predictions',
    'tft_signals',
    'lgbm_scores',
    'regime_history',
    'portfolio_weights',
    'equity_curve',
    'trade_log',
    'feature_importance',
]

# ============================================================================
# Validation Configuration
# ============================================================================
VALIDATION_CONFIG = {
    'check_future_leakage': True,
    'check_data_alignment': True,
    'check_missing_values': True,
    'max_missing_pct': 0.05,  # Max 5% missing values allowed
}

# ============================================================================
# Helper Functions
# ============================================================================
def get_tft_dataset_path():
    """Get path to TFT training dataset"""
    return FINAL_DIR / 'tft_training_dataset.parquet'

def get_tft_signals_path():
    """Get path to TFT signals output"""
    return OUTPUT_DIR / 'tft_signals.parquet'

def get_lgbm_scores_path():
    """Get path to LGBM scores output"""
    return OUTPUT_DIR / 'lgbm_scores.parquet'

def get_model_path(model_type, version='base'):
    """Get model save path"""
    return MODEL_DIR / f'{model_type}_{version}.pth'

def validate_config():
    """Validate configuration consistency"""
    errors = []
    
    # Check if factor files exist
    for key, filename in FACTOR_FILES.items():
        path = FACTOR_DIR / filename
        if not path.exists():
            errors.append(f"Missing factor file: {path}")
    
    # Check TFT variable consistency
    all_vars = (
        TFT_VARIABLES['time_varying_known_reals'] + 
        TFT_VARIABLES['time_varying_unknown_reals']
    )
    if len(all_vars) != len(set(all_vars)):
        errors.append("Duplicate variables in TFT_VARIABLES")
    
    # Check regime thresholds
    if REGIME_THRESHOLDS['uncertainty_low'] >= REGIME_THRESHOLDS['uncertainty_high']:
        errors.append("Invalid uncertainty thresholds")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return True

if __name__ == "__main__":
    print("TFT-LGBM Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Factor Dir: {FACTOR_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Model Dir: {MODEL_DIR}")
    print(f"\nTime Range: {START_DATE} -> {END_DATE}")
    print(f"Base Training: {START_DATE} -> {BASE_TRAIN_END}")
    print(f"Rolling Fine-tune: {ROLLING_START} -> {END_DATE}")
    print("\nValidating configuration...")
    try:
        validate_config()
        print("✓ Configuration valid!")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
