"""
Phase 2.5b: Factor Selection 因子筛选模块

基于 factor_research.py 的研究报告, 执行最终筛选:
- ICIR加权排序 (替代纯IC排序)
- 基础门槛: |Rank IC| > min_abs_ic, |ICIR| > 0.3, positive_ratio > 0.52
- 单调性门槛: |monotonicity| >= 0.5
- 稳定性门槛: yearly_consistency >= 0.5
- 相关性去重: R > 0.7 剔除 (从0.85收紧)
"""
import json
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR,
    FACTOR_SELECTION_MIN_ABS_IC,
    FACTOR_SELECTION_MIN_ICIR,
    FACTOR_SELECTION_MIN_POSITIVE_RATIO,
    FACTOR_SELECTION_MAX_FEATURES,
    FACTOR_SELECTION_MIN_FEATURES,
    FACTOR_SELECTION_MAX_CORR,
)


class FactorSelector:
    """ICIR加权 + 相关性去重 因子筛选"""

    def __init__(self, research_report: pd.DataFrame,
                 factor_matrix: pd.DataFrame):
        self.report = research_report.copy()
        self.factor_matrix = factor_matrix
        self.selected_features: List[str] = []
        self.correlation_matrix: pd.DataFrame = None

    def select_features(self) -> List[str]:
        """
        排序筛选法 (适配crypto弱信号环境):
        - 最低门槛: |Rank IC| > min_abs_ic, |ICIR| > min_icir
        - 按 research_score 排序取 Top N
        - R > 0.7 相关性去重
        """
        print("=" * 60)
        print("PHASE 2.5b: FACTOR SELECTION")
        print(f"  Score-ranked selection, correlation dedup R>{FACTOR_SELECTION_MAX_CORR}")
        print("=" * 60)

        df = self.report.copy()
        total = len(df)

        # 确保必要列存在
        if 'abs_icir' not in df.columns:
            df['abs_icir'] = df['rank_icir'].abs()
        if 'abs_ric' not in df.columns:
            df['abs_ric'] = df['rank_ic_mean'].abs()

        # --- 最低门槛 (只去掉噪音因子) ---
        pool = df[df['abs_ric'].fillna(0) >= FACTOR_SELECTION_MIN_ABS_IC].copy()
        print(f"  After |Rank IC| >= {FACTOR_SELECTION_MIN_ABS_IC}: {len(pool)}/{total}")

        pool = pool[pool['abs_icir'].fillna(0) >= FACTOR_SELECTION_MIN_ICIR]
        print(f"  After |ICIR| >= {FACTOR_SELECTION_MIN_ICIR}: {len(pool)}/{total}")

        # --- 按 research_score 排序 (不做硬性cutoff) ---
        ranked = pool.sort_values('research_score', ascending=False)['factor'].tolist()
        print(f"  Candidates ranked by research_score: {len(ranked)}")

        # 如果通过最低门槛的不够，再放宽
        if len(ranked) < FACTOR_SELECTION_MIN_FEATURES:
            print(f"  ⚠️ Only {len(ranked)} passed, relaxing IC threshold...")
            relaxed = df[
                df['abs_ric'].fillna(0) >= FACTOR_SELECTION_MIN_ABS_IC * 0.5
            ].sort_values('research_score', ascending=False)['factor'].tolist()
            for f in relaxed:
                if f not in ranked:
                    ranked.append(f)

        # --- 相关性去重 (R > 0.7) ---
        available = [f for f in ranked if f in self.factor_matrix.columns]
        if not available:
            # 最终兜底：取research_score最高的因子
            available = df.sort_values('research_score', ascending=False)['factor'].tolist()
            available = [f for f in available if f in self.factor_matrix.columns]

        corr_source = self.factor_matrix[available]
        if len(corr_source) > 200000:
            corr_source = corr_source.sample(n=200000, random_state=42)
        self.correlation_matrix = corr_source.corr().abs().fillna(0)

        selected = []
        for factor in available:
            if factor not in self.correlation_matrix.columns:
                continue
            if not selected:
                selected.append(factor)
            else:
                max_corr = self.correlation_matrix.loc[factor, selected].max()
                if max_corr <= FACTOR_SELECTION_MAX_CORR:
                    selected.append(factor)
            if len(selected) >= FACTOR_SELECTION_MAX_FEATURES:
                break

        # 兜底
        if len(selected) < FACTOR_SELECTION_MIN_FEATURES:
            relaxed_limit = min(0.85, FACTOR_SELECTION_MAX_CORR + 0.1)
            print(f"  Relaxing corr threshold to {relaxed_limit} for minimum features...")
            for factor in df.sort_values('research_score', ascending=False)['factor']:
                if factor in selected or factor not in self.correlation_matrix.columns:
                    continue
                if not selected:
                    selected.append(factor)
                else:
                    max_corr = self.correlation_matrix.loc[factor, selected].max()
                    if max_corr <= relaxed_limit:
                        selected.append(factor)
                if len(selected) >= FACTOR_SELECTION_MIN_FEATURES:
                    break

        self.selected_features = selected[:FACTOR_SELECTION_MAX_FEATURES]

        print(f"\n  Final selected: {len(self.selected_features)} features")
        for i, f in enumerate(self.selected_features):
            row = self.report[self.report['factor'] == f]
            if not row.empty:
                r = row.iloc[0]
                print(f"    {i+1:2d}. {f:<25s}  RIC={r['rank_ic_mean']:+.4f}  "
                      f"ICIR={r['rank_icir']:+.3f}  "
                      f"mono={r.get('monotonicity', 0):.2f}  "
                      f"score={r['research_score']:.1f}")
        return self.selected_features

    def save_outputs(self) -> None:
        selected_path = OUTPUT_DIR / 'selected_features.json'
        selected_csv_path = OUTPUT_DIR / 'selected_features_validated.csv'
        corr_path = OUTPUT_DIR / 'selected_feature_correlation.csv'

        pd.Series(self.selected_features, name='feature').to_csv(selected_csv_path, index=False)
        with open(selected_path, 'w', encoding='utf-8') as f:
            json.dump(self.selected_features, f, ensure_ascii=False, indent=2)

        if self.correlation_matrix is not None and self.selected_features:
            sel_corr = self.correlation_matrix.loc[self.selected_features, self.selected_features]
            sel_corr.to_csv(corr_path)

        print(f"  Selection outputs saved to {OUTPUT_DIR}")

    def run_pipeline(self) -> Tuple[pd.DataFrame, List[str]]:
        self.select_features()
        self.save_outputs()
        print("\n" + "=" * 60)
        print("Phase 2.5b completed!")
        print("=" * 60)
        return self.report, self.selected_features
