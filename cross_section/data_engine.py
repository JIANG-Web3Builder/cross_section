"""
Phase 0+1: Data Engine 数据引擎 (量化私募级)

整合数据审计 + 数据清洗 + 宇宙构建，形成闭环:
1. 加载RAW数据 (不使用预清洗版本)
2. 数据审计: 死币检测、异常值统计
3. 收益率清洗: ±15%/h硬上限 + 截面缩尾
4. 流动性过滤: 最低成交额门槛
5. 存活偏差处理: 退市前buffer踢出
6. 宇宙构建: Mid-Cap Rank 11-100
7. 市场指数: 成交额加权 (非中位数)
8. 因子用清洗数据，回测用原始数据
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_FILES, OUTPUT_DIR, EXCLUDE_SYMBOLS,
    MAX_RETURN_THRESHOLD, WINSORIZE_STD,
    MIN_HOURLY_QUOTE_VOLUME, FFILL_LIMIT,
    MIN_DATA_COVERAGE, DEAD_COIN_HOURS, DELIST_BUFFER_HOURS,
    LOOKBACK_HOURS_VOLUME, MIN_HISTORY_HOURS,
    UNIVERSE_TOP_EXCLUDE, UNIVERSE_BOTTOM_RANK,
)


class DataEngine:
    """数据引擎 — 量化私募级数据清洗"""

    def __init__(self, extra_exclude: list = None):
        self.data: Dict[str, pd.DataFrame] = {}
        self.universe: pd.DataFrame = None
        self.market_index: pd.DataFrame = None
        self.extra_exclude = extra_exclude or []
        self.audit_log: List[str] = []
        self.dead_coins: List[str] = []

    def _log(self, msg: str):
        print(f"  [DataEngine] {msg}")
        self.audit_log.append(msg)

    # ------------------------------------------------------------------
    # Phase 0: Load + Audit
    # ------------------------------------------------------------------
    def load_data(self) -> None:
        """加载RAW parquet数据"""
        print("=" * 60)
        print("Loading RAW data (no pre-cleaning)...")
        for name, path in DATA_FILES.items():
            if not path.exists():
                raise FileNotFoundError(f"Data file missing: {path}")
            self.data[name] = pd.read_parquet(path)
            self.data[name].index = pd.to_datetime(self.data[name].index)
            print(f"  {name}: {self.data[name].shape}")

        # 排除已知问题币种
        all_exclude = set(EXCLUDE_SYMBOLS) | set(self.extra_exclude)
        for name in self.data:
            cols_to_drop = [c for c in self.data[name].columns if c in all_exclude]
            if cols_to_drop:
                self.data[name] = self.data[name].drop(columns=cols_to_drop)

        print(f"  Time range: {self.data['close'].index[0]} to {self.data['close'].index[-1]}")
        print(f"  Symbols after exclusion: {len(self.data['close'].columns)}")

    def run_audit(self) -> List[str]:
        """数据质量审计"""
        print("=" * 60)
        print("DATA QUALITY AUDIT")
        print("=" * 60)

        c = self.data['close']
        r = self.data['returns']
        v = self.data.get('quote_volume', self.data.get('volume'))

        # 1. 缺失率
        total_cells = c.size
        missing = c.isna().sum().sum()
        self._log(f"Missing data: {missing/total_cells*100:.1f}%")

        # 2. 极端收益
        pumps = (r > 1.0).sum().sum()
        dumps = (r < -0.5).sum().sum()
        self._log(f"Pumps >100%/h: {pumps}, Dumps <-50%/h: {dumps}")
        self._log(f"Return range: [{r.min().min()*100:.1f}%, {r.max().max()*100:.1f}%]")

        # 3. 死币检测
        dead_coins = []
        for symbol in c.columns:
            price_flat = (c[symbol].diff().abs() < 1e-10).rolling(DEAD_COIN_HOURS).sum()
            if price_flat.max() >= DEAD_COIN_HOURS:
                dead_coins.append(symbol)
                continue
            if v is not None:
                zero_vol = (v[symbol].fillna(0) == 0).rolling(DEAD_COIN_HOURS).sum()
                if zero_vol.max() >= DEAD_COIN_HOURS:
                    dead_coins.append(symbol)

        self.dead_coins = dead_coins
        self._log(f"Dead/illiquid coins: {len(dead_coins)}")
        if dead_coins:
            self._log(f"  Examples: {dead_coins[:10]}")

        # 4. 低覆盖率币种
        coverage = c.notna().mean()
        low_coverage = coverage[coverage < MIN_DATA_COVERAGE].index.tolist()
        self._log(f"Low coverage (<{MIN_DATA_COVERAGE*100:.0f}%): {len(low_coverage)} coins")

        # 保存审计报告
        report_path = OUTPUT_DIR / 'data_audit_report.txt'
        report_path.write_text('\n'.join(self.audit_log), encoding='utf-8')

        return dead_coins

    # ------------------------------------------------------------------
    # Phase 1a: 数据清洗
    # ------------------------------------------------------------------
    def clean_data(self) -> None:
        """
        量化私募级数据清洗:
        1. 剔除死币
        2. 前向填充(限制6h)
        3. 收益率硬上限 ±15%/h
        4. 截面缩尾 (±3.5σ)
        5. 存活偏差处理
        """
        print("=" * 60)
        print("DATA CLEANING (Quant Fund Standard)")
        print("=" * 60)

        # --- Step 1: 剔除死币和低覆盖率币种 ---
        coverage = self.data['close'].notna().mean()
        bad_coins = set(self.dead_coins)
        bad_coins |= set(coverage[coverage < MIN_DATA_COVERAGE].index)
        self._log(f"Removing {len(bad_coins)} bad coins (dead + low coverage)")

        for name in self.data:
            cols_to_drop = [c for c in self.data[name].columns if c in bad_coins]
            if cols_to_drop:
                self.data[name] = self.data[name].drop(columns=cols_to_drop)

        self._log(f"Remaining symbols: {len(self.data['close'].columns)}")

        # --- Step 2: 前向填充 (限制FFILL_LIMIT小时) ---
        for name in ['open', 'high', 'low', 'close']:
            self.data[name] = self.data[name].ffill(limit=FFILL_LIMIT)
        self.data['volume'] = self.data['volume'].fillna(0)
        self.data['quote_volume'] = self.data['quote_volume'].fillna(0)

        # --- Step 3: 重新计算收益率 (从清洗后的close) ---
        # 不使用预计算的returns，从close重新计算避免任何预处理泄露
        raw_returns = self.data['close'].pct_change()
        self.data['returns'] = raw_returns
        self._log("Returns recomputed from cleaned close prices")

        # --- Step 4: 保存原始收益率 (回测结算用) ---
        self.data['returns_raw'] = raw_returns.copy()

        # --- Step 5: 收益率硬上限 ±15%/h ---
        n_capped = (raw_returns.abs() > MAX_RETURN_THRESHOLD).sum().sum()
        returns_capped = raw_returns.clip(-MAX_RETURN_THRESHOLD, MAX_RETURN_THRESHOLD)
        self._log(f"Return cap ±{MAX_RETURN_THRESHOLD*100:.0f}%/h: capped {n_capped:,} values")

        # --- Step 6: 截面缩尾 ---
        def winsorize_row(row):
            valid = row.dropna()
            if len(valid) < 5:
                return row
            mu = valid.mean()
            sigma = valid.std()
            if sigma < 1e-10:
                return row
            lower = mu - WINSORIZE_STD * sigma
            upper = mu + WINSORIZE_STD * sigma
            return row.clip(lower=lower, upper=upper)

        returns_clean = returns_capped.apply(winsorize_row, axis=1)
        n_winsorized = (returns_capped != returns_clean).sum().sum()
        self._log(f"Cross-sectional winsorize ±{WINSORIZE_STD}σ: {n_winsorized:,} values")

        self.data['returns_clean'] = returns_clean

        # --- Step 7: 存活偏差处理 ---
        # 对每个币种，找到最后有效数据的时间点，提前DELIST_BUFFER踢出
        close = self.data['close']
        last_valid = close.apply(lambda col: col.last_valid_index())
        max_time = close.index[-1]

        delist_mask = pd.DataFrame(True, index=close.index, columns=close.columns)
        for symbol in close.columns:
            lv = last_valid[symbol]
            if lv is not None and lv < max_time:
                # 这个币在数据结束前就消失了 → 退市
                buffer_start = lv - pd.Timedelta(hours=DELIST_BUFFER_HOURS)
                delist_mask.loc[buffer_start:, symbol] = False
                self._log(f"  Delist buffer: {symbol} removed from {buffer_start}")

        self.data['delist_mask'] = delist_mask

    # ------------------------------------------------------------------
    # Phase 1b: 宇宙构建
    # ------------------------------------------------------------------
    def build_universe(self) -> pd.DataFrame:
        """构建动态票池 (Mid-Cap + 流动性过滤 + 存活偏差)"""
        print("=" * 60)
        print("Building Dynamic Universe...")

        qv = self.data['quote_volume']
        close = self.data['close']

        # 1. 滚动成交额排名
        rolling_vol = qv.rolling(window=LOOKBACK_HOURS_VOLUME, min_periods=1).mean()
        vol_rank = rolling_vol.rank(axis=1, ascending=False)

        # 2. Mid-Cap: Rank 11-100
        mid_cap = (vol_rank > UNIVERSE_TOP_EXCLUDE) & (vol_rank <= UNIVERSE_BOTTOM_RANK)

        # 3. 流动性下限
        liquidity_ok = rolling_vol >= MIN_HOURLY_QUOTE_VOLUME

        # 4. 足够历史
        valid_count = close.notna().cumsum()
        has_history = valid_count >= MIN_HISTORY_HOURS

        # 5. 存活偏差mask
        delist_mask = self.data.get('delist_mask',
                                     pd.DataFrame(True, index=close.index, columns=close.columns))

        # 6. 数据有效
        has_data = close.notna()

        # 合并
        universe = mid_cap & liquidity_ok & has_history & delist_mask & has_data

        # 保留BTC用于对冲(不参与选币)
        if 'BTCUSDT' in universe.columns:
            universe['BTCUSDT'] = True
            self._log("BTCUSDT preserved for hedging")

        self.universe = universe

        avg_n = universe.sum(axis=1).mean()
        self._log(f"Universe: avg {avg_n:.0f} symbols, shape {universe.shape}")

        return universe

    # ------------------------------------------------------------------
    # Phase 1c: 市场指数 (成交额加权)
    # ------------------------------------------------------------------
    def build_market_index(self) -> pd.DataFrame:
        """
        成交额加权市场指数 (替代中位数)

        原因: 中位数指数在大量垃圾币环境下收益率严重偏负(-99%+),
        导致alpha虚高，IC被严重高估。成交额加权更合理。
        """
        print("=" * 60)
        print("Building Volume-Weighted Market Index...")

        returns = self.data['returns_clean']
        qv = self.data['quote_volume']

        # 只用宇宙内的币种
        masked_returns = returns.where(self.universe)
        masked_qv = qv.where(self.universe).fillna(0)

        # 成交额加权
        total_qv = masked_qv.sum(axis=1).replace(0, np.nan)
        weights = masked_qv.div(total_qv, axis=0)
        index_returns = (masked_returns * weights).sum(axis=1)

        # 回填NaN
        index_returns = index_returns.fillna(0)

        # 累计净值
        index_price = (1 + index_returns).cumprod() * 1000

        self.market_index = pd.DataFrame({
            'returns': index_returns,
            'price': index_price
        })

        total_ret = (index_price.iloc[-1] / index_price.iloc[0] - 1) * 100
        ann_vol = index_returns.std() * np.sqrt(24 * 365) * 100
        self._log(f"Index total return: {total_ret:.2f}%")
        self._log(f"Index ann. volatility: {ann_vol:.2f}%")

        return self.market_index

    # ------------------------------------------------------------------
    # Phase 1d: 输出
    # ------------------------------------------------------------------
    def get_panel_data(self) -> Dict[str, pd.DataFrame]:
        return {
            'open': self.data['open'],
            'high': self.data['high'],
            'low': self.data['low'],
            'close': self.data['close'],
            'volume': self.data['volume'],
            'quote_volume': self.data['quote_volume'],
            'returns': self.data['returns_raw'],
            'returns_clean': self.data['returns_clean'],
            'universe': self.universe,
            'market_index': self.market_index,
        }

    def run_pipeline(self) -> Dict[str, pd.DataFrame]:
        """运行完整数据引擎"""
        print("\n" + "=" * 70)
        print(" PHASE 0+1: DATA ENGINE (Quant Fund Standard)")
        print("=" * 70)

        self.load_data()
        self.run_audit()
        self.clean_data()
        self.build_universe()
        self.build_market_index()

        panel_data = self.get_panel_data()

        # 保存中间结果
        self.market_index.to_parquet(OUTPUT_DIR / 'market_index.parquet')
        self.universe.to_parquet(OUTPUT_DIR / 'universe_mask.parquet')

        print("\n" + "=" * 60)
        print("Data Engine completed!")
        print("=" * 60)

        return panel_data


if __name__ == "__main__":
    engine = DataEngine()
    panel = engine.run_pipeline()
    print(f"\nClose: {panel['close'].shape}")
    print(f"Universe avg: {panel['universe'].sum(axis=1).mean():.0f}")
    print(f"Index return: {(panel['market_index']['price'].iloc[-1]/panel['market_index']['price'].iloc[0]-1)*100:.2f}%")
