import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from config_tft import (
    FACTOR_DIR,
    OUTPUT_DIR,
    START_DATE,
    END_DATE,
    CANDIDATE_FACTOR_FEATURES,
    FACTOR_RESEARCH_CONFIG,
)

warnings.filterwarnings('ignore')


class FactorResearchPipeline:
    def __init__(self, factor_dir: Path = None, output_dir: Path = None):
        self.factor_dir = factor_dir or FACTOR_DIR
        self.output_dir = output_dir or (OUTPUT_DIR / 'factor_research')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.selection_path = OUTPUT_DIR / 'selected_factors.json'
        self.config = FACTOR_RESEARCH_CONFIG.copy()
        self.panel = None
        self.metrics = None

    def _load_factor_files(self):
        factor_files = sorted(self.factor_dir.glob('*_factors.parquet'))
        return [path for path in factor_files if path.stem.replace('_factors', '') + '_1h.csv']

    def _extract_symbol(self, factor_file: Path) -> str:
        stem = factor_file.name.replace('_factors.parquet', '')
        return stem.replace('USDT', '')

    def _safe_corr(self, left: pd.Series, right: pd.Series) -> float:
        if left.nunique() < 2 or right.nunique() < 2:
            return np.nan
        return left.corr(right)

    def load_panel(self) -> pd.DataFrame:
        panel_parts = []
        factor_files = sorted(self.factor_dir.glob('*_factors.parquet'))
        horizon = int(self.config['forward_return_hours'])

        for factor_file in factor_files:
            symbol = self._extract_symbol(factor_file)
            price_file = self.factor_dir / f'{symbol}USDT_1h.csv'
            if not price_file.exists():
                continue

            factor_df = pd.read_parquet(factor_file)
            if 'timestamp' not in factor_df.columns:
                continue

            factor_df['timestamp'] = pd.to_datetime(factor_df['timestamp'])
            available_factors = [col for col in CANDIDATE_FACTOR_FEATURES if col in factor_df.columns]
            if not available_factors:
                continue
            factor_df = factor_df[['timestamp'] + available_factors].copy()

            price_df = pd.read_csv(price_file, usecols=['open_time', 'close'])
            price_df['timestamp'] = pd.to_datetime(price_df['open_time'])
            price_df = price_df.sort_values('timestamp')
            price_df['forward_return'] = np.log(
                price_df['close'].shift(-horizon) / price_df['close'].replace(0, np.nan)
            )
            merged = factor_df.merge(
                price_df[['timestamp', 'forward_return']],
                on='timestamp',
                how='inner'
            )
            merged['symbol'] = symbol
            merged = merged[(merged['timestamp'] >= START_DATE) & (merged['timestamp'] < END_DATE)]
            merged = merged.replace([np.inf, -np.inf], np.nan)
            panel_parts.append(merged)

        if not panel_parts:
            raise ValueError('No factor panel could be built from factor directory.')

        panel = pd.concat(panel_parts, ignore_index=True)
        panel = panel.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        self.panel = panel
        return panel

    def _compute_quantile_diagnostics(self, panel: pd.DataFrame, factor_name: str) -> tuple:
        spreads = []
        monotonicities = []
        quantile_returns = {str(i): [] for i in range(int(self.config['quantiles']))}

        for _, group in panel.groupby('timestamp'):
            sample = group[[factor_name, 'forward_return']].dropna()
            if len(sample) < self.config['min_assets_per_timestamp']:
                continue
            if sample[factor_name].nunique() < 2 or sample['forward_return'].nunique() < 2:
                continue

            bucket_count = int(min(self.config['quantiles'], len(sample)))
            if bucket_count < 2:
                continue

            ranks = sample[factor_name].rank(method='first')
            try:
                buckets = pd.qcut(ranks, q=bucket_count, labels=False, duplicates='drop')
            except ValueError:
                continue

            if buckets.nunique() < 2:
                continue

            bucket_returns = sample.groupby(buckets)['forward_return'].mean().sort_index()
            spreads.append(bucket_returns.iloc[-1] - bucket_returns.iloc[0])

            bucket_index = pd.Series(np.arange(len(bucket_returns)), index=bucket_returns.index, dtype=float)
            monotonicity = self._safe_corr(bucket_index, bucket_returns)
            if pd.notna(monotonicity):
                monotonicities.append(monotonicity)

            for bucket_id, bucket_ret in bucket_returns.items():
                quantile_returns[str(int(bucket_id))].append(bucket_ret)

        mean_quantile_returns = {
            key: float(np.nanmean(values)) if values else np.nan
            for key, values in quantile_returns.items()
        }
        spread_mean = float(np.nanmean(spreads)) if spreads else np.nan
        monotonicity_mean = float(np.nanmean(monotonicities)) if monotonicities else np.nan
        return spread_mean, monotonicity_mean, mean_quantile_returns

    def _analyze_single_factor(self, factor_name: str) -> dict:
        sample = self.panel[['timestamp', 'symbol', factor_name, 'forward_return']].dropna()
        coverage = float(self.panel[factor_name].notna().mean()) if factor_name in self.panel.columns else 0.0

        ic_values = []
        rank_ic_values = []
        timestamps = []

        for timestamp, group in sample.groupby('timestamp'):
            if len(group) < self.config['min_assets_per_timestamp']:
                continue
            if group[factor_name].nunique() < 2 or group['forward_return'].nunique() < 2:
                continue

            ic = self._safe_corr(group[factor_name], group['forward_return'])
            rank_ic = self._safe_corr(group[factor_name].rank(method='average'), group['forward_return'].rank(method='average'))

            if pd.notna(ic):
                ic_values.append(ic)
            if pd.notna(rank_ic):
                rank_ic_values.append(rank_ic)
                timestamps.append(timestamp)

        ic_series = pd.Series(ic_values, dtype=float)
        rank_ic_series = pd.Series(rank_ic_values, index=pd.to_datetime(timestamps), dtype=float)

        ic_mean = float(ic_series.mean()) if not ic_series.empty else np.nan
        rank_ic_mean = float(rank_ic_series.mean()) if not rank_ic_series.empty else np.nan
        ic_std = float(ic_series.std(ddof=0)) if len(ic_series) > 1 else np.nan
        rank_ic_std = float(rank_ic_series.std(ddof=0)) if len(rank_ic_series) > 1 else np.nan
        ic_ir = float(ic_mean / ic_std) if pd.notna(ic_std) and abs(ic_std) > 1e-12 else np.nan
        rank_ic_ir = float(rank_ic_mean / rank_ic_std) if pd.notna(rank_ic_std) and abs(rank_ic_std) > 1e-12 else np.nan

        effective_sign = np.sign(rank_ic_mean) if pd.notna(rank_ic_mean) and rank_ic_mean != 0 else 1.0
        non_zero_rank_ic = rank_ic_series[rank_ic_series != 0]
        sign_consistency = float((np.sign(non_zero_rank_ic) == effective_sign).mean()) if not non_zero_rank_ic.empty else np.nan

        yearly_rank_ic = rank_ic_series.groupby(rank_ic_series.index.year).mean() if not rank_ic_series.empty else pd.Series(dtype=float)
        yearly_sign_consistency = float((np.sign(yearly_rank_ic[yearly_rank_ic != 0]) == effective_sign).mean()) if not yearly_rank_ic.empty else np.nan
        spread_mean, monotonicity_mean, quantile_returns = self._compute_quantile_diagnostics(sample, factor_name)

        selection_score = np.nansum([
            abs(rank_ic_mean) * 100 if pd.notna(rank_ic_mean) else 0.0,
            abs(rank_ic_ir) * 10 if pd.notna(rank_ic_ir) else 0.0,
            abs(spread_mean) * 100 if pd.notna(spread_mean) else 0.0,
            (sign_consistency or 0.0) * 5,
        ])

        return {
            'factor': factor_name,
            'coverage': coverage,
            'ic_mean': ic_mean,
            'ic_ir': ic_ir,
            'rank_ic_mean': rank_ic_mean,
            'rank_ic_ir': rank_ic_ir,
            'sign_consistency': sign_consistency,
            'yearly_sign_consistency': yearly_sign_consistency,
            'long_short_spread_mean': spread_mean,
            'monotonicity_mean': monotonicity_mean,
            'valid_timestamps': int(len(rank_ic_series)),
            'direction': int(1 if effective_sign >= 0 else -1),
            'selection_score': float(selection_score),
            'quantile_returns': quantile_returns,
        }

    def analyze_factors(self) -> pd.DataFrame:
        if self.panel is None:
            self.load_panel()

        metrics = []
        for factor_name in CANDIDATE_FACTOR_FEATURES:
            if factor_name not in self.panel.columns:
                continue
            metrics.append(self._analyze_single_factor(factor_name))

        metrics_df = pd.DataFrame(metrics)
        if metrics_df.empty:
            raise ValueError('No factor diagnostics were generated.')

        metrics_df = metrics_df.sort_values('selection_score', ascending=False).reset_index(drop=True)
        self.metrics = metrics_df
        return metrics_df

    def _passes_thresholds(self, row: pd.Series) -> bool:
        return (
            row['coverage'] >= self.config['min_coverage']
            and abs(row['rank_ic_mean']) >= self.config['min_abs_rank_ic']
            and abs(row['rank_ic_ir']) >= self.config['min_abs_rank_ic_ir']
            and row['sign_consistency'] >= self.config['min_sign_consistency']
        )

    def select_factors(self) -> dict:
        if self.metrics is None:
            self.analyze_factors()

        candidate_rows = self.metrics[self.metrics.apply(self._passes_thresholds, axis=1)].copy()
        correlation_panel = self.panel[[col for col in CANDIDATE_FACTOR_FEATURES if col in self.panel.columns]].copy()
        correlation_matrix = correlation_panel.corr().fillna(0.0)

        selected = []
        directions = {}
        rejected = []

        for _, row in candidate_rows.sort_values('selection_score', ascending=False).iterrows():
            factor_name = row['factor']
            too_close = False
            for chosen in selected:
                if factor_name not in correlation_matrix.index or chosen not in correlation_matrix.columns:
                    continue
                if abs(correlation_matrix.loc[factor_name, chosen]) >= self.config['max_factor_correlation']:
                    too_close = True
                    break

            if too_close:
                rejected.append(factor_name)
                continue

            selected.append(factor_name)
            directions[factor_name] = int(row['direction'])
            if len(selected) >= int(self.config['top_n_factors']):
                break

        if not selected and not self.metrics.empty:
            fallback_row = self.metrics.iloc[0]
            selected = [fallback_row['factor']]
            directions = {fallback_row['factor']: int(fallback_row['direction'])}

        payload = {
            'selected_factors': selected,
            'factor_directions': directions,
            'candidate_factors': [row['factor'] for _, row in candidate_rows.iterrows()],
            'rejected_by_correlation': rejected,
            'research_config': self.config,
        }
        return payload

    def save_outputs(self, selection_payload: dict):
        metrics_to_save = self.metrics.copy()
        metrics_to_save['quantile_returns'] = metrics_to_save['quantile_returns'].apply(json.dumps)
        metrics_path = self.output_dir / 'factor_diagnostics.csv'
        metrics_to_save.to_csv(metrics_path, index=False, encoding='utf-8-sig')

        summary_path = self.output_dir / 'factor_diagnostics.parquet'
        metrics_to_save.to_parquet(summary_path, index=False)

        self.selection_path.write_text(
            json.dumps(selection_payload, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

        correlation_panel = self.panel[[col for col in CANDIDATE_FACTOR_FEATURES if col in self.panel.columns]].copy()
        correlation_panel.corr().to_csv(self.output_dir / 'factor_correlation.csv', encoding='utf-8-sig')

    def run_pipeline(self) -> tuple:
        self.load_panel()
        metrics = self.analyze_factors()
        selection_payload = self.select_factors()
        self.save_outputs(selection_payload)
        return metrics, selection_payload


def main():
    pipeline = FactorResearchPipeline()
    metrics, selection_payload = pipeline.run_pipeline()
    print('=' * 80)
    print('FACTOR RESEARCH COMPLETED')
    print('=' * 80)
    print(f"Selected factors: {selection_payload['selected_factors']}")
    print(metrics[['factor', 'rank_ic_mean', 'rank_ic_ir', 'sign_consistency', 'selection_score']].head(10))


if __name__ == '__main__':
    main()
