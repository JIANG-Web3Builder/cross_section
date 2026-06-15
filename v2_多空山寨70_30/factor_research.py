"""
Phase 2.5a: Factor Research 因子研究模块 (私募级)

核心检验:
1. IC / Rank IC / ICIR 基础检验
2. IC Decay 分析 — 确定最优持仓周期
3. 分组收益单调性检验 — 确认因子不只两端有效
4. 滚动IC方向稳定性检验 — 排除方向翻转因子
5. 因子自相关/换手率分析 — 验证换仓频率是否合理
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from config import (
    FORWARD_HOURS, OUTPUT_DIR,
    FACTOR_VALIDATION_QUANTILES,
    FACTOR_VALIDATION_MIN_CROSS_SECTION,
    FACTOR_VALIDATION_MIN_TIMESTAMPS,
)

RESEARCH_OUTPUT_DIR = OUTPUT_DIR / 'factor_research'
RESEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class FactorResearcher:
    """
    因子研究引擎 — 私募级因子检验

    输出:
    - factor_research_report.csv   全因子诊断报告
    - ic_decay_analysis.csv        IC Decay 矩阵
    - selected_features.json       筛选后因子列表
    """

    def __init__(self, panel_data: Dict[str, pd.DataFrame],
                 factors_normalized: Dict[str, pd.DataFrame],
                 factor_matrix: pd.DataFrame):
        self.returns = panel_data['returns']
        self.universe = panel_data['universe']
        self.market_index = panel_data['market_index']
        self.factors_normalized = factors_normalized
        self.factor_matrix = factor_matrix

        self.report: pd.DataFrame = None
        self.ic_decay_matrix: pd.DataFrame = None
        self.selected_features: List[str] = []

    # ------------------------------------------------------------------
    # 1. 基础 IC / Rank IC
    # ------------------------------------------------------------------
    def _compute_ic_series(self, factor_df: pd.DataFrame,
                           forward_alpha: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """逐截面计算 Pearson IC 和 Rank IC"""
        masked = factor_df.where(self.universe)
        aligned = ((masked.notna()) & (forward_alpha.notna())).sum(axis=1)
        eligible = aligned >= FACTOR_VALIDATION_MIN_CROSS_SECTION

        ic_list, ric_list, ts_list = [], [], []
        for ts in masked.index[eligible]:
            f_row = masked.loc[ts].dropna()
            r_row = forward_alpha.loc[ts].reindex(f_row.index).dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) < FACTOR_VALIDATION_MIN_CROSS_SECTION:
                continue
            f_vals = f_row[common]
            r_vals = r_row[common]
            ic_list.append(f_vals.corr(r_vals))
            ric_list.append(f_vals.rank().corr(r_vals.rank()))
            ts_list.append(ts)

        ic_series = pd.Series(ic_list, index=ts_list, dtype=float)
        ric_series = pd.Series(ric_list, index=ts_list, dtype=float)
        return ic_series, ric_series

    # ------------------------------------------------------------------
    # 2. IC Decay — 确定最优持仓周期
    # ------------------------------------------------------------------
    def compute_ic_decay(self, lags: List[int] = None) -> pd.DataFrame:
        """
        对每个因子，计算不同 forward horizon 的 Rank IC.
        确定因子信息的最优衰减速度 -> 指导 FORWARD_HOURS 的选择.
        """
        if lags is None:
            lags = [1, 3, 6, 12, 24, 48, 72]

        print("=" * 60)
        print("IC Decay Analysis — determining optimal holding period...")
        print(f"  Lags tested: {lags}")

        index_ret = self.market_index['returns']
        decay_rows = []

        factor_items = list(self.factors_normalized.items())
        for idx, (name, fdf) in enumerate(factor_items, 1):
            row = {'factor': name}
            for lag in lags:
                fwd = self.returns.shift(-lag).rolling(lag).sum()
                idx_fwd = index_ret.shift(-lag).rolling(lag).sum()
                alpha_fwd = fwd.sub(idx_fwd, axis=0).where(self.universe)
                _, ric = self._compute_ic_series(fdf, alpha_fwd)
                row[f'ric_{lag}h'] = ric.mean() if len(ric) > 0 else np.nan
            decay_rows.append(row)
            if idx % 20 == 0 or idx == len(factor_items):
                print(f"  Progress: {idx}/{len(factor_items)}")

        decay_df = pd.DataFrame(decay_rows)
        self.ic_decay_matrix = decay_df
        decay_df.to_csv(RESEARCH_OUTPUT_DIR / 'ic_decay_analysis.csv', index=False)
        print(f"  IC Decay saved to {RESEARCH_OUTPUT_DIR / 'ic_decay_analysis.csv'}")
        return decay_df

    # ------------------------------------------------------------------
    # 3. 分组单调性检验
    # ------------------------------------------------------------------
    def _quantile_monotonicity(self, factor_df: pd.DataFrame,
                                forward_alpha: pd.DataFrame,
                                n_quantiles: int = None) -> Tuple[float, float, Dict]:
        """
        将截面按因子值分组, 检验分组平均收益是否单调递增/递减.

        Returns:
            monotonicity: Spearman 相关 (分组序号 vs 分组均收益)
            spread: Top组 - Bottom组 平均收益
            group_returns: {q0: mean_ret, q1: ..., ...}
        """
        if n_quantiles is None:
            n_quantiles = FACTOR_VALIDATION_QUANTILES
        masked = factor_df.where(self.universe)
        group_accum = {i: [] for i in range(n_quantiles)}
        spread_list = []

        for ts in masked.index:
            f_row = masked.loc[ts].dropna()
            r_row = forward_alpha.loc[ts].reindex(f_row.index).dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) < n_quantiles * 3:
                continue
            f_vals = f_row[common]
            r_vals = r_row[common]
            try:
                buckets = pd.qcut(f_vals.rank(method='first'), q=n_quantiles,
                                  labels=False, duplicates='drop')
            except ValueError:
                continue
            if buckets.nunique() < n_quantiles:
                continue
            group_mean = r_vals.groupby(buckets).mean().sort_index()
            for g_id, g_ret in group_mean.items():
                group_accum[int(g_id)].append(g_ret)
            spread_list.append(float(group_mean.iloc[-1] - group_mean.iloc[0]))

        avg_group = {k: np.nanmean(v) if v else np.nan for k, v in group_accum.items()}
        vals = pd.Series(avg_group)
        if vals.dropna().nunique() < 2:
            return np.nan, np.nan, avg_group
        from scipy.stats import spearmanr
        mono, _ = spearmanr(np.arange(len(vals)), vals.values)
        spread = np.nanmean(spread_list) if spread_list else np.nan
        return float(mono), float(spread), avg_group

    # ------------------------------------------------------------------
    # 4. 滚动 IC 方向稳定性
    # ------------------------------------------------------------------
    def _rolling_ic_stability(self, ric_series: pd.Series,
                               window: int = 720) -> Tuple[float, float, float]:
        """
        在滚动窗口内计算 IC 均值, 检查方向是否翻转.

        Returns:
            yearly_sign_consistency: 年度IC方向一致率
            rolling_positive_ratio: 滚动窗口内IC > 0 的比例
            half_life_ratio: 前半段 vs 后半段 IC 方向一致率
        """
        if len(ric_series) < FACTOR_VALIDATION_MIN_TIMESTAMPS:
            return np.nan, np.nan, np.nan

        overall_sign = np.sign(ric_series.mean())
        if overall_sign == 0:
            overall_sign = 1.0

        # 年度一致性
        yearly = ric_series.groupby(ric_series.index.year).mean()
        yearly_consistency = float((np.sign(yearly) == overall_sign).mean()) if len(yearly) > 0 else np.nan

        # 滚动窗口
        rolling_mean = ric_series.rolling(window, min_periods=window // 2).mean()
        rolling_positive = float((np.sign(rolling_mean.dropna()) == overall_sign).mean())

        # 前后半段
        mid = len(ric_series) // 2
        first_half_mean = ric_series.iloc[:mid].mean()
        second_half_mean = ric_series.iloc[mid:].mean()
        half_consistent = 1.0 if np.sign(first_half_mean) == np.sign(second_half_mean) else 0.0

        return yearly_consistency, rolling_positive, half_consistent

    # ------------------------------------------------------------------
    # 5. 因子自相关 / 换手率
    # ------------------------------------------------------------------
    def _factor_autocorrelation(self, factor_df: pd.DataFrame,
                                 lag: int = 12) -> Tuple[float, float]:
        """
        截面级因子排名的自相关 — 衡量因子换手率.
        高自相关 = 低换手 = 适合低频换仓
        低自相关 = 高换手 = 需要更频繁换仓

        Returns:
            rank_autocorr: 截面排名自相关均值
            turnover: 平均持仓重叠率 (top quantile)
        """
        masked = factor_df.where(self.universe)
        rank_df = masked.rank(axis=1, pct=True)

        autocorr_list = []
        overlap_list = []
        top_pct = 1 - 1.0 / FACTOR_VALIDATION_QUANTILES

        timestamps = rank_df.index
        for i in range(lag, len(timestamps), lag):
            curr = rank_df.iloc[i].dropna()
            prev = rank_df.iloc[i - lag].dropna()
            common = curr.index.intersection(prev.index)
            if len(common) < FACTOR_VALIDATION_MIN_CROSS_SECTION:
                continue
            autocorr_list.append(curr[common].corr(prev[common]))
            curr_top = set(curr[common][curr[common] >= top_pct].index)
            prev_top = set(prev[common][prev[common] >= top_pct].index)
            if len(curr_top) > 0 and len(prev_top) > 0:
                overlap = len(curr_top & prev_top) / max(len(curr_top), len(prev_top))
                overlap_list.append(overlap)

        autocorr = np.nanmean(autocorr_list) if autocorr_list else np.nan
        turnover = 1.0 - (np.nanmean(overlap_list) if overlap_list else 0.5)
        return float(autocorr), float(turnover)

    # ------------------------------------------------------------------
    # 综合研究: 单因子完整诊断
    # ------------------------------------------------------------------
    def _research_single_factor(self, name: str, fdf: pd.DataFrame,
                                 forward_alpha: pd.DataFrame) -> Dict:
        """对单个因子运行全部检验"""
        ic_s, ric_s = self._compute_ic_series(fdf, forward_alpha)

        if len(ric_s) < FACTOR_VALIDATION_MIN_TIMESTAMPS:
            return {
                'factor': name, 'ic_mean': np.nan, 'rank_ic_mean': np.nan,
                'rank_icir': np.nan, 'rank_ic_tstat': np.nan,
                'positive_ratio': np.nan, 'monotonicity': np.nan,
                'spread': np.nan, 'yearly_consistency': np.nan,
                'rolling_stability': np.nan, 'half_life_stable': np.nan,
                'rank_autocorr': np.nan, 'factor_turnover': np.nan,
                'direction': 0, 'n_periods': len(ric_s),
            }

        ic_mean = float(ic_s.mean())
        ric_mean = float(ric_s.mean())
        ric_std = float(ric_s.std())
        direction = 1 if ric_mean >= 0 else -1
        signed_ric = ric_s * direction

        icir = ric_mean / (ric_std + 1e-10)
        tstat = ric_mean / (ric_std / np.sqrt(len(ric_s))) if ric_std > 0 else np.nan
        pos_ratio = float((signed_ric > 0).mean())

        mono, spread, _ = self._quantile_monotonicity(fdf, forward_alpha)
        yearly_con, rolling_stab, half_stable = self._rolling_ic_stability(ric_s)
        autocorr, turnover = self._factor_autocorrelation(fdf, lag=FORWARD_HOURS)

        return {
            'factor': name,
            'ic_mean': ic_mean,
            'rank_ic_mean': ric_mean,
            'rank_icir': icir,
            'rank_ic_tstat': tstat,
            'positive_ratio': pos_ratio,
            'monotonicity': float(mono) * direction if pd.notna(mono) else np.nan,
            'spread': float(spread) * direction if pd.notna(spread) else np.nan,
            'yearly_consistency': yearly_con,
            'rolling_stability': rolling_stab,
            'half_life_stable': half_stable,
            'rank_autocorr': autocorr,
            'factor_turnover': turnover,
            'direction': direction,
            'n_periods': len(ric_s),
        }

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run_research(self) -> pd.DataFrame:
        """运行完整因子研究"""
        print("\n" + "=" * 70)
        print(" PHASE 2.5a: FACTOR RESEARCH (私募级因子检验)")
        print("=" * 70)

        # 构建目标收益
        index_ret = self.market_index['returns']
        fwd = self.returns.shift(-FORWARD_HOURS).rolling(FORWARD_HOURS).sum()
        idx_fwd = index_ret.shift(-FORWARD_HOURS).rolling(FORWARD_HOURS).sum()
        forward_alpha = fwd.sub(idx_fwd, axis=0).where(self.universe)

        # 逐因子研究
        print("Running single-factor diagnostics...")
        rows = []
        items = list(self.factors_normalized.items())
        for idx, (name, fdf) in enumerate(items, 1):
            rows.append(self._research_single_factor(name, fdf, forward_alpha))
            if idx % 20 == 0 or idx == len(items):
                print(f"  Progress: {idx}/{len(items)}")

        report = pd.DataFrame(rows)
        report['abs_ric'] = report['rank_ic_mean'].abs()

        # 综合评分
        report['research_score'] = (
            report['abs_ric'].fillna(0) * 100
            + report['positive_ratio'].fillna(0) * 10
            + report['monotonicity'].abs().fillna(0) * 15
            + report['yearly_consistency'].fillna(0) * 10
            + report['rolling_stability'].fillna(0) * 10
            + report['spread'].abs().clip(upper=0.01).fillna(0) * 500
        )
        report = report.sort_values('research_score', ascending=False).reset_index(drop=True)
        self.report = report

        # 打印摘要
        print("\n  Top 15 factors by research score:")
        cols = ['factor', 'rank_ic_mean', 'rank_icir', 'positive_ratio',
                'monotonicity', 'yearly_consistency', 'rolling_stability',
                'factor_turnover', 'research_score']
        print(report[cols].head(15).to_string(index=False))

        # IC Decay
        self.compute_ic_decay()

        # 保存
        report.to_csv(RESEARCH_OUTPUT_DIR / 'factor_research_report.csv', index=False)
        print(f"\n  Full research report saved to {RESEARCH_OUTPUT_DIR}")

        # 打印 IC Decay 推荐
        self._print_ic_decay_recommendation()

        return report

    def _print_ic_decay_recommendation(self):
        """基于IC Decay数据给出持仓周期建议"""
        if self.ic_decay_matrix is None:
            return
        print("\n" + "-" * 60)
        print("IC Decay Summary (top 10 factors by 12h Rank IC):")
        decay = self.ic_decay_matrix.copy()
        decay = decay.sort_values('ric_12h', key=abs, ascending=False)
        lag_cols = [c for c in decay.columns if c.startswith('ric_')]
        print(decay[['factor'] + lag_cols].head(10).to_string(index=False))

        # 找到IC峰值对应的lag
        avg_row = decay[lag_cols].abs().mean()
        peak_lag = avg_row.idxmax()
        print(f"\n  Average peak IC at: {peak_lag}")
        print(f"  Current FORWARD_HOURS: {FORWARD_HOURS}h")
        print("-" * 60)

    def get_report(self) -> pd.DataFrame:
        if self.report is None:
            raise RuntimeError("Must call run_research() first")
        return self.report
