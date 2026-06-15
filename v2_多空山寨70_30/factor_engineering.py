"""
Phase 2: Factor Engineering 因子工程模块
基于 因子.md 完整因子库实现
- Alpha 101 精选因子 (15个)
- 机构实战统计因子 (35个)
- 截面标准化 (Cross-Sectional Z-Score)
- 因子缓存功能 (避免重复计算)
- Numba加速复杂滚动计算
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_DIR, FACTOR_CACHE_DIR, ORTHOGONALIZATION_WINDOW

# 因子缓存目录 - 使用本地输出配置
FACTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Numba 加速函数 (向量化滚动计算)
# ============================================================================
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    
    @njit(parallel=True, cache=True)
    def rolling_argmax_nb(arr, window):
        """Numba加速的滚动argmax"""
        n = len(arr)
        result = np.empty(n)
        result[:window-1] = np.nan
        for i in prange(window-1, n):
            max_idx = 0
            max_val = arr[i-window+1]
            for j in range(1, window):
                if arr[i-window+1+j] > max_val:
                    max_val = arr[i-window+1+j]
                    max_idx = j
            result[i] = window - 1 - max_idx  # 距离当前位置的时间
        return result
    
    @njit(parallel=True, cache=True)
    def rolling_argmin_nb(arr, window):
        """Numba加速的滚动argmin"""
        n = len(arr)
        result = np.empty(n)
        result[:window-1] = np.nan
        for i in prange(window-1, n):
            min_idx = 0
            min_val = arr[i-window+1]
            for j in range(1, window):
                if arr[i-window+1+j] < min_val:
                    min_val = arr[i-window+1+j]
                    min_idx = j
            result[i] = window - 1 - min_idx
        return result
    
    @njit(parallel=True, cache=True)
    def rolling_mad_nb(arr, window):
        """Numba加速的滚动Mean Absolute Deviation"""
        n = len(arr)
        result = np.empty(n)
        result[:window-1] = np.nan
        for i in prange(window-1, n):
            window_data = arr[i-window+1:i+1]
            mean_val = np.mean(window_data)
            result[i] = np.mean(np.abs(window_data - mean_val))
        return result

except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available, using slower pandas operations")


def apply_rolling_numba(df: pd.DataFrame, func, window: int) -> pd.DataFrame:
    """对DataFrame的每列应用numba加速的滚动函数"""
    if not NUMBA_AVAILABLE:
        return None
    
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        arr = df[col].values.astype(np.float64)
        result[col] = func(arr, window)
    return result

class FactorEngine:
    """因子工程引擎 - 完整因子库 (支持缓存)"""
    
    def __init__(self, panel_data: Dict[str, pd.DataFrame], use_cache: bool = True):
        self.open = panel_data['open']
        self.high = panel_data['high']
        self.low = panel_data['low']
        self.close = panel_data['close']
        self.volume = panel_data['volume']
        self.quote_volume = panel_data['quote_volume']
        # 修正：计算因子必须用去极值后的数据
        self.returns = panel_data['returns_clean']
        # 保留原始收益率引用备用
        self.raw_returns = panel_data['returns']
        self.universe = panel_data['universe']
        self.market_index = panel_data['market_index']
        
        self.use_cache = use_cache
        self.factors: Dict[str, pd.DataFrame] = {}
        self.factors_normalized: Dict[str, pd.DataFrame] = {}
    
    # ========================================================================
    # 因子缓存管理
    # ========================================================================
    
    def _get_factor_path(self, factor_name: str) -> Path:
        """获取因子文件路径"""
        return FACTOR_CACHE_DIR / f'{factor_name}.parquet'
    
    def _factor_exists(self, factor_name: str) -> bool:
        """检查因子是否已缓存"""
        return self._get_factor_path(factor_name).exists()
    
    def _save_factor(self, factor_name: str, factor_df: pd.DataFrame):
        """保存单个因子到缓存"""
        factor_df.to_parquet(self._get_factor_path(factor_name))
    
    def _load_factor(self, factor_name: str) -> pd.DataFrame:
        """从缓存加载单个因子"""
        return pd.read_parquet(self._get_factor_path(factor_name))
    
    def _calc_and_cache(self, factor_name: str, calc_func, *args, **kwargs) -> pd.DataFrame:
        """计算因子并缓存 (如果已缓存则直接加载)"""
        if self.use_cache and self._factor_exists(factor_name):
            return self._load_factor(factor_name)
        
        # 计算因子
        factor_df = calc_func(*args, **kwargs)
        
        # 保存到缓存
        if self.use_cache:
            self._save_factor(factor_name, factor_df)
        
        return factor_df
    
    def get_cached_factors(self) -> List[str]:
        """获取已缓存的因子列表"""
        return [f.stem for f in FACTOR_CACHE_DIR.glob('*.parquet')]
    
    def clear_cache(self, factor_names: Optional[List[str]] = None):
        """清除因子缓存"""
        if factor_names is None:
            # 清除所有缓存
            for f in FACTOR_CACHE_DIR.glob('*.parquet'):
                f.unlink()
            print("  Cleared all factor cache")
        else:
            # 清除指定因子缓存
            for name in factor_names:
                path = self._get_factor_path(name)
                if path.exists():
                    path.unlink()
            print(f"  Cleared cache for {len(factor_names)} factors")
    
    # ========================================================================
    # 第一部分：Alpha 101 精选因子 (Top 15)
    # ========================================================================
    
    def alpha001(self) -> pd.DataFrame:
        """
        Alpha001: 动量反转因子 (修复版)
        
        原始逻辑存在量纲混乱问题 (混合了close和std)。
        新逻辑:
        - 使用returns的SignedPower(returns^2, 保留符号)
        - 计算滚动窗口内的Ts_ArgMax (距离最大值的时间)
        - 截面Rank标准化
        
        逻辑: 捕捉短期动量的极值点，用于反转交易
        """
        # 使用returns而非混合close和std
        data = self.returns
        
        # SignedPower: 保留符号的平方 (放大极端收益)
        signed_power = np.sign(data) * (data.abs() ** 2)
        
        # Ts_ArgMax: 找到滚动5期内最大值的位置
        # 使用Numba加速版本 (如果可用)
        if NUMBA_AVAILABLE:
            ts_argmax = apply_rolling_numba(signed_power, rolling_argmax_nb, 5)
        else:
            # 回退方案: 使用滚动最大值的比率近似
            rolling_max = signed_power.rolling(5).max()
            ts_argmax = (signed_power / (rolling_max + 1e-10))
        
        # 截面Rank标准化
        return ts_argmax.rank(axis=1, pct=True)
    
    def alpha006(self) -> pd.DataFrame:
        """Alpha006: 量价相关性 - 识别诱多陷阱"""
        return -1 * self.open.rolling(10).corr(self.volume)
    
    def alpha009(self) -> pd.DataFrame:
        """Alpha009: 短期均线趋势 - 捕捉短期惯性"""
        delta = self.close.diff(5)
        return delta.where(delta.abs() > 0, -delta)
    
    def alpha012(self) -> pd.DataFrame:
        """Alpha012: 量价剧烈反转 - 识别恐慌盘"""
        sign_delta_vol = np.sign(self.volume.diff(1))
        delta_close = self.close.diff(1)
        return sign_delta_vol * (-1 * delta_close)
    
    def alpha013(self) -> pd.DataFrame:
        """Alpha013: 量加权协方差 - 确认资金推动的真实性"""
        cov = self.close.rolling(5).cov(self.volume)
        return cov.rank(axis=1, pct=True)
    
    def alpha023(self) -> pd.DataFrame:
        """Alpha023: 高波动均值回归 - 防止高位追涨"""
        high_20 = self.high.rolling(20).max()
        condition = self.close > high_20.shift(1)
        vol = self.returns.rolling(20).std()
        vol_high = vol > vol.rolling(100).quantile(0.8)
        signal = condition & vol_high
        return -signal.astype(float)
    
    def alpha026(self) -> pd.DataFrame:
        """Alpha026: 动量与量能排序 - 识别量价配合失效的顶部"""
        rank_vol = self.volume.rank(axis=1, pct=True)
        rank_high = self.high.rank(axis=1, pct=True)
        corr = rank_vol.rolling(5).corr(rank_high)
        return -corr
    
    def alpha028(self) -> pd.DataFrame:
        """Alpha028: 盘中均价偏离 - 捕捉日内反转"""
        vwap = (self.high + self.low + self.close) / 3
        deviation = self.close - vwap
        vol_decay = self.volume / self.volume.rolling(10).mean()
        return deviation * vol_decay
    
    def alpha040(self) -> pd.DataFrame:
        """Alpha040: 波动成本 - 识别筹码松动的时刻"""
        vol = self.returns.rolling(10).std()
        vol_rank = vol.rank(axis=1, pct=True)
        volume_rank = self.volume.rank(axis=1, pct=True)
        return -vol_rank * volume_rank
    
    def alpha049(self) -> pd.DataFrame:
        """Alpha049: 相对强弱位置 - 纯粹的趋势确认"""
        highest = self.high.rolling(24).max()
        lowest = self.low.rolling(24).min()
        return (self.close - lowest) / (highest - lowest + 1e-10)
    
    def alpha053(self) -> pd.DataFrame:
        """Alpha053: 影线博弈 - 识别多空力量转换"""
        close_low = self.close - self.low
        high_close = self.high - self.close
        return (close_low - high_close) / (close_low + 1e-10)
    
    def alpha054(self) -> pd.DataFrame:
        """Alpha054: 开盘跳空 - 利用市场的非理性情绪"""
        gap = self.open - self.close.shift(1)
        gap_pct = gap / (self.close.shift(1) + 1e-10)
        return -gap_pct  # 跳空后反向修复
    
    def alpha060(self) -> pd.DataFrame:
        """Alpha060: 夹板突破 - 识别突破形态的强度 (优化版)"""
        hl_range = self.high - self.low
        position = (self.close - self.low) / (hl_range + 1e-10)
        # 简化: 使用EMA代替加权均值
        return position.ewm(span=10, adjust=False).mean()
    
    def alpha098(self) -> pd.DataFrame:
        """Alpha098: VWAP回归 - 机构算法单造成的定价偏差"""
        vwap = (self.high + self.low + self.close) / 3
        rank_corr = vwap.rolling(10).corr(self.volume).rank(axis=1, pct=True)
        rank_delta_open = self.open.diff(5).rank(axis=1, pct=True)
        return rank_corr - rank_delta_open
    
    def alpha101(self) -> pd.DataFrame:
        """Alpha101: 实体力度 - 信号强度过滤器"""
        body = self.close - self.open
        total_range = self.high - self.low + 1e-10
        return body / total_range
    
    # ========================================================================
    # 第二部分：机构实战统计因子 (35个)
    # A. 动量类 (Momentum) - 11个
    # ========================================================================
    
    def calc_roc(self, period: int) -> pd.DataFrame:
        """ROC: 价格变化率"""
        return (self.close / self.close.shift(period) - 1) * 100
    
    def calc_bias(self, period: int) -> pd.DataFrame:
        """Bias: 乖离率"""
        ma = self.close.rolling(period).mean()
        return self.close / ma - 1
    
    def calc_slope(self, period: int) -> pd.DataFrame:
        """Slope: 线性回归斜率 (使用差分近似)"""
        # 简化版本: 使用(末尾-开头)/周期 近似斜率
        return (self.close - self.close.shift(period)) / period
    
    def calc_rsi(self, period: int) -> pd.DataFrame:
        """RSI: 相对强弱指数"""
        delta = self.close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def calc_macd_hist(self) -> pd.DataFrame:
        """MACD_Hist: MACD柱状图"""
        ema12 = self.close.ewm(span=12, adjust=False).mean()
        ema26 = self.close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return (macd - signal) / self.close  # 标准化
    
    def calc_trix(self, period: int = 15) -> pd.DataFrame:
        """TRIX: 三重平滑EMA的变化率"""
        ema1 = self.close.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return ema3.pct_change() * 100
    
    def calc_cci(self, period: int) -> pd.DataFrame:
        """CCI: 商品通道指数 (Numba加速)"""
        tp = (self.high + self.low + self.close) / 3
        ma = tp.rolling(period).mean()
        
        # 使用Numba加速计算MAD
        if NUMBA_AVAILABLE:
            md = apply_rolling_numba(tp, rolling_mad_nb, period)
        else:
            # 回退到std近似
            md = tp.rolling(period).std() * 0.8
        
        return (tp - ma) / (0.015 * md + 1e-10)
    
    def calc_apo(self, fast: int = 12, slow: int = 26) -> pd.DataFrame:
        """APO: 绝对价格震荡器"""
        ema_fast = self.close.ewm(span=fast, adjust=False).mean()
        ema_slow = self.close.ewm(span=slow, adjust=False).mean()
        return (ema_fast - ema_slow) / self.close
    
    def calc_cmo(self, period: int = 14) -> pd.DataFrame:
        """CMO: 钱德动量摆动指标"""
        delta = self.close.diff()
        sum_up = delta.where(delta > 0, 0).rolling(period).sum()
        sum_down = (-delta).where(delta < 0, 0).rolling(period).sum()
        return 100 * (sum_up - sum_down) / (sum_up + sum_down + 1e-10)
    
    # ========================================================================
    # B. 波动率类 (Volatility) - 6个
    # ========================================================================
    
    def calc_atr(self, period: int) -> pd.DataFrame:
        """ATR: 平均真实波幅"""
        tr1 = self.high - self.low
        tr2 = (self.high - self.close.shift(1)).abs()
        tr3 = (self.low - self.close.shift(1)).abs()
        tr = pd.DataFrame(np.maximum.reduce([tr1.values, tr2.values, tr3.values]),
                          index=tr1.index, columns=tr1.columns)
        return tr.rolling(period).mean()
    
    def calc_natr(self, period: int) -> pd.DataFrame:
        """NATR: 归一化ATR"""
        return self.calc_atr(period) / self.close
    
    def calc_stddev(self, period: int) -> pd.DataFrame:
        """StdDev: 收益率标准差"""
        return self.returns.rolling(period).std()
    
    def calc_ulcer_index(self, period: int = 14) -> pd.DataFrame:
        """UI: 溃疡指数 - 衡量下行风险"""
        rolling_max = self.close.rolling(period).max()
        pct_drawdown = (self.close - rolling_max) / rolling_max * 100
        return np.sqrt((pct_drawdown ** 2).rolling(period).mean())
    
    def calc_bb_width(self, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
        """BB_Width: 布林带宽度"""
        ma = self.close.rolling(period).mean()
        std = self.close.rolling(period).std()
        upper = ma + std_mult * std
        lower = ma - std_mult * std
        return (upper - lower) / ma
    
    def calc_parkinson_vol(self, period: int) -> pd.DataFrame:
        """Parkinson_Vol: Parkinson波动率"""
        hl_ratio = np.log(self.high / self.low)
        return np.sqrt(hl_ratio.pow(2).rolling(period).mean() / (4 * np.log(2)))
    
    # ========================================================================
    # C. 量价与流动性 (Volume & Liquidity) - 7个
    # ========================================================================
    
    def calc_vwap_dist(self, period: int = 24) -> pd.DataFrame:
        """VWAP_Dist: VWAP偏离度"""
        tp = (self.high + self.low + self.close) / 3
        vwap = (tp * self.volume).rolling(period).sum() / (self.volume.rolling(period).sum() + 1e-10)
        return self.close / vwap - 1
    
    def calc_obv(self) -> pd.DataFrame:
        """OBV: 能量潮"""
        direction = np.sign(self.returns)
        return (direction * self.volume).cumsum()
    
    def calc_obv_norm(self) -> pd.DataFrame:
        """OBV标准化"""
        obv = self.calc_obv()
        return obv / obv.rolling(168).std()
    
    def calc_mfi(self, period: int) -> pd.DataFrame:
        """MFI: 资金流量指标"""
        tp = (self.high + self.low + self.close) / 3
        raw_mf = tp * self.volume
        positive_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(period).sum()
        negative_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(period).sum()
        return 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
    
    def calc_vol_ratio(self, period: int) -> pd.DataFrame:
        """Vol_Ratio: 成交量比率"""
        return self.volume / (self.volume.rolling(period).mean() + 1e-10)
    
    def calc_pvt(self) -> pd.DataFrame:
        """PVT: 量价趋势指标"""
        return (self.returns * self.volume).cumsum()
    
    def calc_pvt_norm(self) -> pd.DataFrame:
        """PVT标准化"""
        pvt = self.calc_pvt()
        return pvt / (pvt.rolling(168).std() + 1e-10)
    
    def calc_liquidity_ratio(self) -> pd.DataFrame:
        """Liquidity_Ratio: 流动性比率 (Amihud)"""
        return self.returns.abs() / (self.volume + 1e-10)
    
    def calc_davids_ratio(self) -> pd.DataFrame:
        """Davids_Ratio: 买卖力道比率"""
        buy_power = self.close - self.low
        sell_power = self.high - self.close
        return buy_power / (sell_power + 1e-10)
    
    # ========================================================================
    # D. 统计分布特征 (Statistical Moments) - 6个
    # ========================================================================
    
    def calc_skewness(self, period: int) -> pd.DataFrame:
        """Skewness: 偏度"""
        return self.returns.rolling(period).skew()
    
    def calc_kurtosis(self, period: int) -> pd.DataFrame:
        """Kurtosis: 峰度"""
        return self.returns.rolling(period).kurt()
    
    def calc_zscore_close(self, period: int = 24) -> pd.DataFrame:
        """Z_Score_Close: 标准分"""
        ma = self.close.rolling(period).mean()
        std = self.close.rolling(period).std()
        return (self.close - ma) / (std + 1e-10)
    
    def calc_ts_argmax(self, period: int) -> pd.DataFrame:
        """Ts_ArgMax: 距离最高价的时间 (Numba加速)"""
        if NUMBA_AVAILABLE:
            return apply_rolling_numba(self.high, rolling_argmax_nb, period)
        else:
            # 回退到近似方法
            rolling_max = self.high.rolling(period).max()
            return (self.high / rolling_max)
    
    def calc_ts_argmin(self, period: int) -> pd.DataFrame:
        """Ts_ArgMin: 距离最低价的时间 (Numba加速)"""
        if NUMBA_AVAILABLE:
            return apply_rolling_numba(self.low, rolling_argmin_nb, period)
        else:
            # 回退到近似方法
            rolling_min = self.low.rolling(period).min()
            return (rolling_min / self.low)
    
    def calc_corr_pv(self, period: int) -> pd.DataFrame:
        """Correlation_PV: 价量相关性"""
        return self.close.rolling(period).corr(self.volume)
    
    # ========================================================================
    # E. K线形态特征 (Pattern) - 5个
    # ========================================================================
    
    def calc_shadow_upper(self) -> pd.DataFrame:
        """Shadow_Upper: 上影线长度"""
        body_top = pd.DataFrame(np.maximum(self.open.values, self.close.values),
                                index=self.open.index, columns=self.open.columns)
        return (self.high - body_top) / (self.close + 1e-10)
    
    def calc_shadow_lower(self) -> pd.DataFrame:
        """Shadow_Lower: 下影线长度"""
        body_bottom = pd.DataFrame(np.minimum(self.open.values, self.close.values),
                                   index=self.open.index, columns=self.open.columns)
        return (body_bottom - self.low) / (self.close + 1e-10)
    
    def calc_body_size(self) -> pd.DataFrame:
        """Body_Size: K线实体力度"""
        return (self.close - self.open).abs() / (self.close + 1e-10)
    
    def calc_gap(self) -> pd.DataFrame:
        """Gap: 跳空缺口幅度"""
        return (self.open - self.close.shift(1)) / (self.close.shift(1) + 1e-10)
    
    def calc_log_return(self) -> pd.DataFrame:
        """Log_Return: 对数收益率"""
        return np.log(self.close / self.close.shift(1))
    
    # ========================================================================
    # 额外扩展因子
    # ========================================================================
    
    def calc_momentum(self, period: int) -> pd.DataFrame:
        """Momentum: 动量"""
        return self.close.pct_change(period)
    
    def calc_williams_r(self, period: int) -> pd.DataFrame:
        """Williams %R"""
        highest = self.high.rolling(period).max()
        lowest = self.low.rolling(period).min()
        return (highest - self.close) / (highest - lowest + 1e-10) * -100
    
    def calc_stochastic_k(self, period: int) -> pd.DataFrame:
        """Stochastic %K"""
        lowest = self.low.rolling(period).min()
        highest = self.high.rolling(period).max()
        return (self.close - lowest) / (highest - lowest + 1e-10) * 100
    
    def calc_realized_vol(self, period: int) -> pd.DataFrame:
        """Realized Volatility"""
        return np.sqrt((self.returns ** 2).rolling(period).sum())
    
    def calc_hl_ratio(self, period: int) -> pd.DataFrame:
        """High-Low Ratio"""
        return (self.high.rolling(period).max() - self.low.rolling(period).min()) / self.close
    
    def calc_max_drawdown(self, period: int) -> pd.DataFrame:
        """Rolling Max Drawdown"""
        rolling_max = self.close.rolling(period).max()
        drawdown = self.close / rolling_max - 1
        return drawdown.rolling(period).min()
    
    def calc_up_down_ratio(self, period: int) -> pd.DataFrame:
        """Up/Down Ratio"""
        up = (self.returns > 0).rolling(period).sum()
        down = (self.returns < 0).rolling(period).sum()
        return up / (down + 1e-10)
    
    # ========================================================================
    # 因子生成主函数 (支持缓存)
    # ========================================================================
    
    def _get_factor_definitions(self) -> Dict[str, tuple]:
        """
        定义所有因子及其计算函数
        返回: {因子名: (计算函数, 参数...)}
        """
        return {
            # Alpha 101 精选因子 (15个)
            'alpha001': (self.alpha001,),
            'alpha006': (self.alpha006,),
            'alpha009': (self.alpha009,),
            'alpha012': (self.alpha012,),
            'alpha013': (self.alpha013,),
            'alpha023': (self.alpha023,),
            'alpha026': (self.alpha026,),
            'alpha028': (self.alpha028,),
            'alpha040': (self.alpha040,),
            'alpha049': (self.alpha049,),
            'alpha053': (self.alpha053,),
            'alpha054': (self.alpha054,),
            'alpha060': (self.alpha060,),
            'alpha098': (self.alpha098,),
            'alpha101': (self.alpha101,),
            # 动量类因子
            'roc_6': (self.calc_roc, 6),
            'roc_24': (self.calc_roc, 24),
            'roc_72': (self.calc_roc, 72),
            'bias_24': (self.calc_bias, 24),
            'bias_72': (self.calc_bias, 72),
            'slope_12': (self.calc_slope, 12),
            'slope_24': (self.calc_slope, 24),
            'rsi_14': (self.calc_rsi, 14),
            'rsi_24': (self.calc_rsi, 24),
            'macd_hist': (self.calc_macd_hist,),
            'trix': (self.calc_trix,),
            'cci_14': (self.calc_cci, 14),
            'cci_24': (self.calc_cci, 24),
            'apo': (self.calc_apo,),
            'cmo': (self.calc_cmo,),
            # 波动率类因子
            'atr_14': (self.calc_atr, 14),
            'natr_14': (self.calc_natr, 14),
            'natr_24': (self.calc_natr, 24),
            'stddev_24': (self.calc_stddev, 24),
            'stddev_72': (self.calc_stddev, 72),
            'ulcer_index': (self.calc_ulcer_index,),
            'bb_width': (self.calc_bb_width,),
            'parkinson_vol': (self.calc_parkinson_vol, 24),
            # 量价与流动性因子
            'vwap_dist_24': (self.calc_vwap_dist, 24),
            'vwap_dist_72': (self.calc_vwap_dist, 72),
            'obv_norm': (self.calc_obv_norm,),
            'mfi_14': (self.calc_mfi, 14),
            'mfi_24': (self.calc_mfi, 24),
            'vol_ratio_24': (self.calc_vol_ratio, 24),
            'vol_ratio_72': (self.calc_vol_ratio, 72),
            'pvt_norm': (self.calc_pvt_norm,),
            'liquidity_ratio': (self.calc_liquidity_ratio,),
            'davids_ratio': (self.calc_davids_ratio,),
            # 统计分布特征
            'skewness_24': (self.calc_skewness, 24),
            'skewness_72': (self.calc_skewness, 72),
            'kurtosis_24': (self.calc_kurtosis, 24),
            'kurtosis_72': (self.calc_kurtosis, 72),
            'zscore_close': (self.calc_zscore_close,),
            'ts_argmax_24': (self.calc_ts_argmax, 24),
            'ts_argmin_24': (self.calc_ts_argmin, 24),
            'corr_pv_12': (self.calc_corr_pv, 12),
            'corr_pv_24': (self.calc_corr_pv, 24),
            # K线形态特征
            'shadow_upper': (self.calc_shadow_upper,),
            'shadow_lower': (self.calc_shadow_lower,),
            'body_size': (self.calc_body_size,),
            'gap': (self.calc_gap,),
            'log_return': (self.calc_log_return,),
            # 扩展因子
            'momentum_12': (self.calc_momentum, 12),
            'momentum_24': (self.calc_momentum, 24),
            'momentum_72': (self.calc_momentum, 72),
            'momentum_168': (self.calc_momentum, 168),
            'williams_r_24': (self.calc_williams_r, 24),
            'williams_r_72': (self.calc_williams_r, 72),
            'stoch_k_24': (self.calc_stochastic_k, 24),
            'stoch_k_72': (self.calc_stochastic_k, 72),
            'realized_vol_24': (self.calc_realized_vol, 24),
            'realized_vol_72': (self.calc_realized_vol, 72),
            'hl_ratio_24': (self.calc_hl_ratio, 24),
            'hl_ratio_72': (self.calc_hl_ratio, 72),
            'max_dd_72': (self.calc_max_drawdown, 72),
            'max_dd_168': (self.calc_max_drawdown, 168),
            'up_down_ratio_24': (self.calc_up_down_ratio, 24),
            'up_down_ratio_72': (self.calc_up_down_ratio, 72),
        }
    
    def generate_all_factors(self, factor_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        生成所有因子 - 支持缓存
        
        Args:
            factor_names: 指定要计算的因子列表，None表示计算所有因子
        """
        print("\n" + "=" * 60)
        print("PHASE 2: FACTOR ENGINEERING (支持缓存)")
        print("=" * 60)
        
        factor_defs = self._get_factor_definitions()
        
        if factor_names is None:
            factor_names = list(factor_defs.keys())
        
        # 统计缓存情况
        cached = [n for n in factor_names if self._factor_exists(n)]
        to_calc = [n for n in factor_names if not self._factor_exists(n)]
        
        if self.use_cache:
            print(f"  Cache enabled: {FACTOR_CACHE_DIR}")
            print(f"  Cached factors: {len(cached)}, To calculate: {len(to_calc)}")
        
        factors = {}
        calc_count = 0
        load_count = 0
        
        for i, name in enumerate(factor_names):
            if name not in factor_defs:
                print(f"  Warning: Unknown factor '{name}', skipping")
                continue
            
            # 检查缓存
            if self.use_cache and self._factor_exists(name):
                factors[name] = self._load_factor(name)
                load_count += 1
            else:
                # 计算因子
                func_def = factor_defs[name]
                func = func_def[0]
                args = func_def[1:] if len(func_def) > 1 else ()
                
                factors[name] = func(*args)
                calc_count += 1
                
                # 保存到缓存
                if self.use_cache:
                    self._save_factor(name, factors[name])
            
            # 进度显示
            if (i + 1) % 20 == 0 or (i + 1) == len(factor_names):
                print(f"  Progress: {i+1}/{len(factor_names)} factors")
        
        self.factors = factors
        print(f"\n  Loaded from cache: {load_count}")
        print(f"  Newly calculated: {calc_count}")
        print(f"  Total factors: {len(factors)}")
        
        return factors
    
    def add_new_factor(self, factor_name: str, calc_func, *args) -> pd.DataFrame:
        """
        添加新因子 (计算并缓存)
        
        Args:
            factor_name: 因子名称
            calc_func: 计算函数
            *args: 计算函数的参数
        
        Returns:
            计算好的因子DataFrame
        """
        print(f"  Adding new factor: {factor_name}")
        
        if self._factor_exists(factor_name):
            print(f"    Factor '{factor_name}' already exists in cache, loading...")
            factor_df = self._load_factor(factor_name)
        else:
            print(f"    Calculating factor '{factor_name}'...")
            factor_df = calc_func(*args)
            if self.use_cache:
                self._save_factor(factor_name, factor_df)
                print(f"    Saved to cache")
        
        self.factors[factor_name] = factor_df
        return factor_df
    
    def orthogonalize_factors(self):
        """
        残差正交化: 剔除大盘(BTC)影响
        Feature_new = Feature_raw - (alpha + beta * Return_BTC)
        
        目的: 强迫模型学习剥离大盘后的纯粹个股规律
        """
        print("=" * 60)
        print(f"Residual Orthogonalization (Stripping BTC influence, Window={ORTHOGONALIZATION_WINDOW})...")
        
        # 获取BTC收益率 (使用清洗后的收益率)
        if 'BTCUSDT' not in self.returns.columns:
            print("  Warning: BTCUSDT not found in returns, skipping orthogonalization")
            return
            
        btc_ret = self.returns['BTCUSDT']
        window = ORTHOGONALIZATION_WINDOW
        
        # 预计算BTC的滚动统计量
        print("  Calculating rolling statistics for BTC...")
        btc_mean = btc_ret.rolling(window=window).mean()
        btc_var = btc_ret.rolling(window=window).var()
        
        ortho_factors = {}
        processed_count = 0
        
        for name, df in self.factors.items():
            # 滚动协方差 (Vectorized)
            # df.rolling().cov(series) 计算DataFrame每列与Series的滚动协方差
            cov = df.rolling(window=window).cov(btc_ret)
            
            # 计算Beta (Cov / Var)
            # 注意: 使用axis=0进行广播
            beta = cov.div(btc_var, axis=0)
            
            # 计算Alpha (Mean_Y - Beta * Mean_X)
            df_mean = df.rolling(window=window).mean()
            alpha = df_mean - beta.multiply(btc_mean, axis=0)
            
            # 计算预期值 (Systematic component = Alpha + Beta * X)
            systematic = alpha + beta.multiply(btc_ret, axis=0)
            
            # 计算残差 (Residual = Original - Systematic)
            residual = df - systematic
            
            # 替换原始因子
            ortho_factors[name] = residual
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count} factors...")
        
        self.factors = ortho_factors
        print(f"  Done. Orthogonalized {len(ortho_factors)} factors.")

    def cross_sectional_normalize(self) -> Dict[str, pd.DataFrame]:
        """
        截面标准化 (Cross-Sectional Z-Score)
        
        对每个时间切片，对票池内的因子值做Z-Score
        """
        print("=" * 60)
        print("Cross-sectional normalization...")
        
        normalized = {}
        
        for name, factor_df in self.factors.items():
            # 只对票池内的币种计算
            masked = factor_df.where(self.universe)
            
            # 截面均值和标准差
            cs_mean = masked.mean(axis=1)
            cs_std = masked.std(axis=1)
            
            # Z-Score标准化
            normalized_df = masked.sub(cs_mean, axis=0).div(cs_std.replace(0, np.nan), axis=0)
            
            # 缩尾处理(±3倍标准差)
            normalized_df = normalized_df.clip(-3, 3)
            
            normalized[name] = normalized_df
        
        self.factors_normalized = normalized
        
        # 验证
        sample_factor = list(normalized.keys())[0]
        sample_stats = normalized[sample_factor].stack().describe()
        print(f"  Sample factor '{sample_factor}' stats after normalization:")
        print(f"    Mean: {sample_stats['mean']:.4f}, Std: {sample_stats['std']:.4f}")
        print(f"    Min: {sample_stats['min']:.4f}, Max: {sample_stats['max']:.4f}")
        
        return normalized
    
    def create_factor_matrix(self) -> pd.DataFrame:
        """创建因子矩阵 (长表格式)"""
        print("=" * 60)
        print("Creating factor matrix...")
        
        factor_list = []
        for name, factor_df in self.factors_normalized.items():
            stacked = factor_df.stack()
            stacked.name = name
            factor_list.append(stacked)
        
        factor_matrix = pd.concat(factor_list, axis=1)
        factor_matrix.index.names = ['timestamp', 'symbol']
        
        # 只保留票池内的记录
        universe_stacked = self.universe.stack()
        factor_matrix = factor_matrix[universe_stacked]
        factor_matrix = factor_matrix.dropna(how='all')
        
        print(f"  Factor matrix shape: {factor_matrix.shape}")
        print(f"  Total factors: {len(factor_matrix.columns)}")
        
        return factor_matrix
    
    def calc_factor_ic(self, forward_returns: pd.DataFrame) -> pd.DataFrame:
        """计算因子IC值"""
        print("=" * 60)
        print("Calculating factor IC...")
        
        ic_results = {}
        for name, factor_df in self.factors_normalized.items():
            ic_series = factor_df.corrwith(forward_returns, axis=1)
            ic_results[name] = {
                'ic_mean': ic_series.mean(),
                'ic_std': ic_series.std(),
                'icir': ic_series.mean() / (ic_series.std() + 1e-10),
                'ic_positive_ratio': (ic_series > 0).mean()
            }
        
        ic_df = pd.DataFrame(ic_results).T
        ic_df = ic_df.sort_values('icir', ascending=False)
        
        print("  Top 15 factors by ICIR:")
        print(ic_df.head(15).to_string())
        
        return ic_df
    
    def run_pipeline(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """运行因子工程流水线"""
        self.generate_all_factors()
        self.orthogonalize_factors()  # 新增：残差正交化
        self.cross_sectional_normalize()
        factor_matrix = self.create_factor_matrix()
        
        # 保存
        factor_matrix.to_parquet(OUTPUT_DIR / 'factor_matrix.parquet')
        
        print("\n" + "=" * 60)
        print("Phase 2 completed!")
        print(f"Total factors: {len(self.factors)}")
        print("=" * 60)
        
        return self.factors_normalized, factor_matrix
    


if __name__ == "__main__":
    from data_engineering import DataEngine
    
    data_engine = DataEngine()
    panel_data = data_engine.run_pipeline()
    
    factor_engine = FactorEngine(panel_data)
    factors_normalized, factor_matrix = factor_engine.run_pipeline()
    
    print(f"\nFinal factor matrix: {factor_matrix.shape}")
