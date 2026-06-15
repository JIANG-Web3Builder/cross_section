"""Phase 2.5b: Factor Validation & Selection 因子筛选模块

基于 factor_research.py 的研究报告, 执行最终筛选:
- 基础门槛: |Rank IC| > min_abs_ic, positive_ratio > min_pos_ratio
- 单调性门槛: monotonicity > 0.5 (分组收益不能乱序)
- 稳定性门槛: yearly_consistency >= 0.5 (年度方向不翻转)
- 相关性去重: 贪心选低相关因子
"""
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR,
    FACTOR_SELECTION_MIN_ABS_IC,
    FACTOR_SELECTION_MIN_POSITIVE_RATIO,
    FACTOR_SELECTION_MAX_FEATURES,
    FACTOR_SELECTION_MIN_FEATURES,
    FACTOR_SELECTION_MAX_CORR,
)


class FactorValidator:
    """基于研究报告的因子筛选模块"""

    def __init__(self, research_report: pd.DataFrame,
                 factor_matrix: pd.DataFrame):
        self.report = research_report.copy()
        self.factor_matrix = factor_matrix
        self.selected_features: List[str] = []
        self.correlation_matrix: pd.DataFrame = None

    def select_features(self) -> List[str]:
        """多维度门槛筛选 + 相关性去重"""
        print("=" * 60)
        print("PHASE 2.5b: FACTOR SELECTION (research-informed)")
        print("=" * 60)

        df = self.report.copy()
        total = len(df)

        # --- 门槛1: 基础IC ---
        pool = df[df['abs_ric'].fillna(0) >= FACTOR_SELECTION_MIN_ABS_IC]
        print(f"  After |Rank IC| >= {FACTOR_SELECTION_MIN_ABS_IC}: {len(pool)}/{total}")

        # --- 门槛2: 方向一致性 ---
        pool = pool[pool['positive_ratio'].fillna(0) >= FACTOR_SELECTION_MIN_POSITIVE_RATIO]
        print(f"  After positive_ratio >= {FACTOR_SELECTION_MIN_POSITIVE_RATIO}: {len(pool)}/{total}")

        # --- 门槛3: 分组单调性 ---
        mono_thresh = 0.5
        pool = pool[pool['monotonicity'].abs().fillna(0) >= mono_thresh]
        print(f"  After |monotonicity| >= {mono_thresh}: {len(pool)}/{total}")

        # --- 门槛4: 年度方向稳定性 ---
        stab_thresh = 0.5
        pool = pool[pool['yearly_consistency'].fillna(0) >= stab_thresh]
        print(f"  After yearly_consistency >= {stab_thresh}: {len(pool)}/{total}")

        # --- 门槛5: spread > 0 ---
        pool = pool[pool['spread'].fillna(0) > 0]
        print(f"  After spread > 0: {len(pool)}/{total}")

        # 按research_score排序
        ranked = pool.sort_values('research_score', ascending=False)['factor'].tolist()

        # 如果通过门槛的因子不够, 放宽到仅IC+positive_ratio
        if len(ranked) < FACTOR_SELECTION_MIN_FEATURES:
            print(f"  ⚠️ Only {len(ranked)} passed strict thresholds, relaxing...")
            relaxed = df[
                (df['abs_ric'].fillna(0) >= FACTOR_SELECTION_MIN_ABS_IC * 0.5)
                & (df['positive_ratio'].fillna(0) >= 0.50)
            ].sort_values('research_score', ascending=False)['factor'].tolist()
            for f in relaxed:
                if f not in ranked:
                    ranked.append(f)

        # --- 相关性去重 ---
        available = [f for f in ranked if f in self.factor_matrix.columns]
        corr_source = self.factor_matrix[available]
        if len(corr_source) > 200000:
            corr_source = corr_source.sample(n=200000, random_state=42)
        self.correlation_matrix = corr_source.corr().abs().fillna(0)

        selected = []
        for factor in available:
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
            relaxed_limit = min(0.95, FACTOR_SELECTION_MAX_CORR + 0.1)
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
                      f"mono={r['monotonicity']:.2f}  stab={r.get('yearly_consistency', 0):.2f}  "
                      f"score={r['research_score']:.1f}")
        return self.selected_features

    def save_outputs(self) -> None:
        """保存筛选结果"""
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
