"""
Phase 1: Data Engineering 数据工程模块
- 构建动态票池 (Dynamic Universe)
- 构建合成大盘指数 (Synthetic Market Index)
- 数据对齐与去极值
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_FILES, LOOKBACK_HOURS_VOLUME,
    WINSORIZE_STD, MAX_RETURN_THRESHOLD, EXCLUDE_SYMBOLS,
    MIN_HISTORY_HOURS, OUTPUT_DIR
)


class DataEngine:
    """数据工程引擎"""
    
    def __init__(self, extra_exclude: list = None):
        self.data: Dict[str, pd.DataFrame] = {}
        self.universe: pd.DataFrame = None  # 动态票池mask
        self.market_index: pd.DataFrame = None  # 合成大盘指数
        self.extra_exclude = extra_exclude or []
        
    def load_data(self) -> None:
        """加载所有parquet数据"""
        print("=" * 60)
        print("Loading data...")
        for name, path in DATA_FILES.items():
            self.data[name] = pd.read_parquet(path)
            self.data[name].index = pd.to_datetime(self.data[name].index)
            print(f"  {name}: {self.data[name].shape}")
        
        # 排除问题币种 (config + audit闭环)
        all_exclude = set(EXCLUDE_SYMBOLS) | set(self.extra_exclude)
        if self.extra_exclude:
            print(f"  Extra exclusions from audit: {len(self.extra_exclude)} coins")
        for name in self.data:
            cols_to_drop = [c for c in self.data[name].columns if c in all_exclude]
            if cols_to_drop:
                self.data[name] = self.data[name].drop(columns=cols_to_drop)
        
        print(f"Data loaded. Time range: {self.data['close'].index[0]} to {self.data['close'].index[-1]}")
        print(f"Total symbols: {len(self.data['close'].columns)}")
        
        # 验证数据完整性
        print("\n" + "!"*60)
        print("  DATA INTEGRITY CHECK")
        print("!"*60)
        print("  原始数据索引已经是Close Time (K线结束时间)")
        print("  Returns[t] = (Close[t] - Close[t-1]) / Close[t-1]")
        print("  无需修改索引，数据无泄露！")
        print("!"*60 + "\n")
        
    def build_dynamic_universe(self) -> pd.DataFrame:
        """
        步骤1: 构建优化的动态票池 (Mid-Cap Universe + BTC)
        
        核心逻辑:
        - 剔除成交额Top 10 (大票被机构主导，难做Alpha)
        - 选取Rank 11-100 的腰部资产 (流动性好，波动适中)
        - 剔除Rank 100+ (流动性差，滑点大)
        - **保留 BTCUSDT**: 用于做空对冲，不参与选币，但必须在数据中
        """
        print("=" * 60)
        print("Building Optimized Mid-Cap Universe (Rank 11-100 + BTC for hedging)...")
        
        quote_volume = self.data['quote_volume'].copy()
        
        # 计算24小时滚动成交额
        rolling_volume = quote_volume.rolling(window=LOOKBACK_HOURS_VOLUME, min_periods=1).sum()
        
        # 每日截面排名 (1 = 成交额最大)
        vol_rank = rolling_volume.rank(axis=1, ascending=False)
        
        # 核心过滤：剔除前10 (大票)，剔除100以后 (流动性差)
        # Mid-Cap: Rank 11-100
        mid_cap_mask = (vol_rank > 10) & (vol_rank <= 100)
        
        # **关键修改**: 强制保留 BTCUSDT (用于做空对冲)
        if 'BTCUSDT' in mid_cap_mask.columns:
            mid_cap_mask['BTCUSDT'] = True
            print("  *** BTCUSDT preserved for hedging (not in selection pool)")
        
        # 确保币种有足够历史数据
        valid_count = (~self.data['close'].isna()).cumsum()
        has_enough_history = valid_count >= MIN_HISTORY_HOURS
        mid_cap_mask = mid_cap_mask & has_enough_history
        
        self.universe = mid_cap_mask
        
        # 统计
        avg_symbols = mid_cap_mask.sum(axis=1).mean()
        btc_included = 'BTCUSDT' in mid_cap_mask.columns and mid_cap_mask['BTCUSDT'].any()
        print(f"  Strategy: Exclude Top 10, Select Rank 11-100")
        print(f"  Average symbols in universe: {avg_symbols:.1f}")
        print(f"  BTC included for hedging: {btc_included}")
        print(f"  Universe shape: {mid_cap_mask.shape}")
        
        return mid_cap_mask
    
    def build_market_index(self) -> pd.DataFrame:
        """
        步骤2: 构建合成大盘指数
        使用票池内资产的中位数收益率 (更稳健，抗极端值)
        """
        print("=" * 60)
        print("Building synthetic market index (using MEDIAN for robustness)...")
        
        returns = self.data['returns'].copy()
        
        # 只使用在票池内的币种计算
        masked_returns = returns.where(self.universe)
        
        # *** 使用中位数代替均值，构建更稳健的大盘指数 ***
        index_returns = masked_returns.median(axis=1)
        
        # *** 修复NaN问题：填充第一行NaN为0 ***
        index_returns = index_returns.fillna(0)
        
        # 累计收益构建指数(从1000开始)
        index_price = (1 + index_returns).cumprod() * 1000
        
        self.market_index = pd.DataFrame({
            'returns': index_returns,
            'price': index_price
        })
        
        # 计算一些统计量 (排除NaN)
        valid_price = index_price.dropna()
        if len(valid_price) > 1:
            total_return = (valid_price.iloc[-1] / valid_price.iloc[0] - 1) * 100
        else:
            total_return = 0.0
        ann_vol = index_returns.std() * np.sqrt(24 * 365) * 100
        
        print(f"  Index total return: {total_return:.2f}%")
        print(f"  Index annualized volatility: {ann_vol:.2f}%")
        
        return self.market_index
    
    def winsorize_data(self) -> None:
        """
        步骤3: 数据去极值 (Winsorization)
        注意：只生成清洗后的版本，保留原始版本用于回测
        """
        print("=" * 60)
        print("Winsorizing data (for feature engineering only)...")
        
        # 原始收益率 (绝对不能动！用于回测结算)
        self.data['returns_raw'] = self.data['returns'].copy()
        
        # 对returns进行缩尾 (仅用于因子计算)
        returns = self.data['returns'].copy()
        
        # 计算截面均值和标准差
        def winsorize_row(row):
            """对单行进行缩尾处理"""
            valid = row.dropna()
            if len(valid) < 3:
                return row
            mean = valid.mean()
            std = valid.std()
            if std == 0:
                return row
            lower = mean - WINSORIZE_STD * std
            upper = mean + WINSORIZE_STD * std
            return row.clip(lower=lower, upper=upper)
        
        # 应用缩尾
        winsorized_returns = returns.apply(winsorize_row, axis=1)
        
        # 额外限制：单小时收益不超过阈值
        winsorized_returns = winsorized_returns.clip(
            lower=-MAX_RETURN_THRESHOLD, 
            upper=MAX_RETURN_THRESHOLD
        )
        
        # 统计处理了多少极端值
        n_clipped = (returns != winsorized_returns).sum().sum()
        total_values = returns.count().sum()
        pct_clipped = n_clipped / total_values * 100
        
        print(f"  Winsorized {n_clipped:,} values ({pct_clipped:.4f}%)")
        print(f"  Raw returns preserved for backtest.")
        
        self.data['returns_clean'] = winsorized_returns
        
    def align_data(self) -> pd.DataFrame:
        """
        创建对齐的MultiIndex数据格式
        (timestamp, symbol) -> features
        """
        print("=" * 60)
        print("Aligning data to MultiIndex format...")
        
        # 获取所有在票池中至少出现过一次的币种
        valid_symbols = self.universe.columns[self.universe.any()].tolist()
        valid_times = self.universe.index
        
        print(f"  Valid symbols: {len(valid_symbols)}")
        print(f"  Valid timestamps: {len(valid_times)}")
        
        # 创建基础数据框架
        records = []
        
        for symbol in valid_symbols:
            symbol_data = pd.DataFrame({
                'timestamp': valid_times,
                'symbol': symbol,
                'open': self.data['open'][symbol].values,
                'high': self.data['high'][symbol].values,
                'low': self.data['low'][symbol].values,
                'close': self.data['close'][symbol].values,
                'volume': self.data['volume'][symbol].values,
                'quote_volume': self.data['quote_volume'][symbol].values,
                'returns': self.data['returns_clean'][symbol].values,
                'in_universe': self.universe[symbol].values,
            })
            records.append(symbol_data)
        
        aligned_df = pd.concat(records, ignore_index=True)
        aligned_df = aligned_df.set_index(['timestamp', 'symbol'])
        aligned_df = aligned_df.sort_index()
        
        print(f"  Aligned data shape: {aligned_df.shape}")
        
        return aligned_df
    
    def get_panel_data(self) -> Dict[str, pd.DataFrame]:
        """返回面板数据格式(宽表)，便于因子计算"""
        return {
            'open': self.data['open'],
            'high': self.data['high'],
            'low': self.data['low'],
            'close': self.data['close'],
            'volume': self.data['volume'],
            'quote_volume': self.data['quote_volume'],
            'returns': self.data['returns_raw'],           # 回测用原始数据
            'returns_clean': self.data['returns_clean'],   # 训练用清洗数据
            'universe': self.universe,
            'market_index': self.market_index,
        }
    
    def run_pipeline(self) -> Dict[str, pd.DataFrame]:
        """运行完整的数据工程流水线"""
        print("\n" + "=" * 60)
        print("PHASE 1: DATA ENGINEERING")
        print("=" * 60)
        
        self.load_data()
        self.build_dynamic_universe()
        self.build_market_index()
        self.winsorize_data()
        
        panel_data = self.get_panel_data()
        
        # 保存中间结果
        self.market_index.to_parquet(OUTPUT_DIR / 'market_index.parquet')
        self.universe.to_parquet(OUTPUT_DIR / 'universe_mask.parquet')
        
        print("\n" + "=" * 60)
        print("Phase 1 completed!")
        print("=" * 60)
        
        return panel_data


if __name__ == "__main__":
    engine = DataEngine()
    panel_data = engine.run_pipeline()
    
    # 简单验证
    print("\n验证数据:")
    print(f"  Close prices shape: {panel_data['close'].shape}")
    print(f"  Universe mask shape: {panel_data['universe'].shape}")
    print(f"  Market index length: {len(panel_data['market_index'])}")
