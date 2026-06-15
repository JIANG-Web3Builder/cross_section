"""分析 2025-10 回撤来源：多头腿、空头腿、BTC/ETH overlay 以及持仓结构。"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from config import OUTPUT_DIR, DATA_FILES

positions = pd.read_parquet(OUTPUT_DIR / 'positions.parquet')
long_positions = pd.read_parquet(OUTPUT_DIR / 'long_positions.parquet')
short_positions = pd.read_parquet(OUTPUT_DIR / 'short_positions.parquet')
close = pd.read_parquet(DATA_FILES['close'])
returns = close.pct_change(fill_method=None)

common_idx = returns.index.intersection(positions.index)
common_cols = returns.columns.intersection(positions.columns)
returns = returns.loc[common_idx, common_cols]
positions = positions.loc[common_idx].reindex(columns=common_cols, fill_value=0.0)
long_positions = long_positions.loc[common_idx].reindex(columns=common_cols, fill_value=0.0)
short_positions = short_positions.loc[common_idx].reindex(columns=common_cols, fill_value=0.0)

window = (common_idx >= pd.Timestamp('2025-10-01')) & (common_idx < pd.Timestamp('2025-11-01'))
idx = common_idx[window]

alt_cols = [c for c in common_cols if c not in ['BTCUSDT', 'ETHUSDT']]

net = (positions.shift(1) * returns).sum(axis=1).loc[idx]
long_alt = (long_positions.shift(1).reindex(columns=alt_cols, fill_value=0.0) * returns[alt_cols]).sum(axis=1).loc[idx]
short_alt = (short_positions.shift(1).reindex(columns=alt_cols, fill_value=0.0) * returns[alt_cols]).sum(axis=1).loc[idx]
btc = pd.Series(0.0, index=common_idx)
eth = pd.Series(0.0, index=common_idx)
if 'BTCUSDT' in common_cols:
    btc = (positions['BTCUSDT'].shift(1) * returns['BTCUSDT']).reindex(common_idx).fillna(0.0)
if 'ETHUSDT' in common_cols:
    eth = (positions['ETHUSDT'].shift(1) * returns['ETHUSDT']).reindex(common_idx).fillna(0.0)

summary = {
    'net_compound': (1 + net).prod() - 1,
    'long_alt_compound': (1 + long_alt).prod() - 1,
    'short_alt_compound': (1 + short_alt).prod() - 1,
    'btc_sum': btc.loc[idx].sum(),
    'eth_sum': eth.loc[idx].sum(),
    'avg_btc_pos': positions['BTCUSDT'].loc[idx].mean() if 'BTCUSDT' in positions.columns else 0.0,
    'avg_eth_pos': positions['ETHUSDT'].loc[idx].mean() if 'ETHUSDT' in positions.columns else 0.0,
    'avg_alt_long_gross': long_positions.drop(columns=['BTCUSDT','ETHUSDT'], errors='ignore').clip(lower=0).sum(axis=1).loc[idx].mean(),
    'avg_alt_short_gross': short_positions.drop(columns=['BTCUSDT','ETHUSDT'], errors='ignore').abs().sum(axis=1).loc[idx].mean(),
    'avg_net_exposure': positions.sum(axis=1).loc[idx].mean(),
}

print('=' * 70)
print(' OCT 2025 DRAWDOWN ANALYSIS')
print('=' * 70)
for k, v in summary.items():
    if 'avg_' in k:
        print(f'{k:<22}: {v:+.4f}')
    else:
        print(f'{k:<22}: {v*100:+.2f}%')

# worst hours
worst = net.nsmallest(10)
print('\nWorst 10 hours:')
for ts, v in worst.items():
    print(f'  {ts}  net={v*100:+.2f}%  btc={btc.loc[ts]*100:+.2f}%  eth={eth.loc[ts]*100:+.2f}%  long_alt={long_alt.loc[ts]*100:+.2f}%  short_alt={short_alt.loc[ts]*100:+.2f}%')

# rebalance snapshots inside October
changes = positions.diff().abs().sum(axis=1)
rebalances = changes[changes > 0.01]
rebalances = rebalances[rebalances.index.isin(idx)]
print(f'\nRebalances in Oct 2025: {len(rebalances)}')
for ts in rebalances.index[:10]:
    row = positions.loc[ts]
    top_long = row[row > 0].drop(labels=[c for c in ['BTCUSDT','ETHUSDT'] if c in row.index], errors='ignore').sort_values(ascending=False).head(5)
    top_short = row[row < 0].drop(labels=[c for c in ['BTCUSDT','ETHUSDT'] if c in row.index], errors='ignore').sort_values().head(5)
    print(f'\n[{ts}] BTC={row.get("BTCUSDT",0):+.3f} ETH={row.get("ETHUSDT",0):+.3f} Net={row.sum():+.3f}')
    print('  Top long:', ', '.join([f'{k}:{val:+.3f}' for k, val in top_long.items()]))
    print('  Top short:', ', '.join([f'{k}:{val:+.3f}' for k, val in top_short.items()]))
