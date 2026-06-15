"""对比紧急通道对关键回撤月份的影响"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from config import OUTPUT_DIR, DATA_FILES, TAKER_FEE, SLIPPAGE

pos = pd.read_parquet(OUTPUT_DIR / 'positions.parquet')
lpos = pd.read_parquet(OUTPUT_DIR / 'long_positions.parquet')
spos = pd.read_parquet(OUTPUT_DIR / 'short_positions.parquet')
close = pd.read_parquet(DATA_FILES['close'])
ret = close.pct_change(fill_method=None)

cidx = ret.index.intersection(pos.index)
ccols = ret.columns.intersection(pos.columns)
cost_rate = TAKER_FEE + SLIPPAGE

gross = (pos.shift(1).loc[cidx, ccols] * ret.loc[cidx, ccols]).sum(axis=1)
prev = pos.shift(1).fillna(0)
real = ((prev == 0) & (pos != 0)) | ((prev != 0) & (pos == 0)) | ((prev * pos) < 0)
cost = (pos.diff().abs() * real.astype(float)).sum(axis=1) * cost_rate
net = gross - cost

alt_cols = [c for c in ccols if c not in ['BTCUSDT', 'ETHUSDT']]
long_alt = (lpos.shift(1).reindex(columns=alt_cols, fill_value=0).loc[cidx] * ret.loc[cidx, alt_cols]).sum(axis=1)
short_alt = (spos.shift(1).reindex(columns=alt_cols, fill_value=0).loc[cidx] * ret.loc[cidx, alt_cols]).sum(axis=1)
btc_pnl = (pos['BTCUSDT'].shift(1).loc[cidx] * ret.loc[cidx, 'BTCUSDT']) if 'BTCUSDT' in ccols else pd.Series(0, index=cidx)
eth_pnl = (pos['ETHUSDT'].shift(1).loc[cidx] * ret.loc[cidx, 'ETHUSDT']) if 'ETHUSDT' in ccols else pd.Series(0, index=cidx)

# Core scale tracking
core_gross = pos.drop(columns=['BTCUSDT', 'ETHUSDT'], errors='ignore').abs().sum(axis=1)

out = pd.DataFrame({
    'net': net, 'long_alt': long_alt, 'short_alt': short_alt,
    'btc': btc_pnl, 'eth': eth_pnl, 'cost': cost,
    'core_gross_exposure': core_gross,
    'btc_pos': pos['BTCUSDT'].reindex(cidx).fillna(0),
})

monthly = out.resample('ME').agg({
    'net': lambda x: (1+x).prod()-1,
    'long_alt': lambda x: (1+x).prod()-1,
    'short_alt': lambda x: (1+x).prod()-1,
    'btc': 'sum', 'eth': 'sum', 'cost': 'sum',
    'core_gross_exposure': 'mean',
    'btc_pos': 'mean',
})

# Previous baseline (from earlier session)
baseline = {
    '2025-02': {'net': -0.1528, 'long_alt': -0.2970, 'short_alt': 0.1745, 'btc': -0.0198, 'eth': -0.0288},
    '2025-10': {'net': -0.3069, 'long_alt': -0.1973, 'short_alt': -0.1694, 'btc': -0.0207, 'eth': -0.0149},
}

print("=" * 70)
print(" EMERGENCY CHANNEL IMPACT ANALYSIS")
print("=" * 70)

for month in ['2025-02', '2025-10']:
    m = monthly[monthly.index.strftime('%Y-%m') == month]
    if len(m) == 0:
        continue
    row = m.iloc[0]
    b = baseline.get(month, {})
    
    print(f"\n--- {month} ---")
    print(f"  {'Metric':<20s} {'Baseline':>10s} {'With Emergency':>15s} {'Delta':>10s}")
    print(f"  {'─'*55}")
    print(f"  {'Net':<20s} {b.get('net',0)*100:>+9.1f}% {row['net']*100:>+14.1f}% {(row['net']-b.get('net',0))*100:>+9.1f}%")
    print(f"  {'Long alt':<20s} {b.get('long_alt',0)*100:>+9.1f}% {row['long_alt']*100:>+14.1f}%")
    print(f"  {'Short alt':<20s} {b.get('short_alt',0)*100:>+9.1f}% {row['short_alt']*100:>+14.1f}%")
    print(f"  {'BTC overlay':<20s} {b.get('btc',0)*100:>+9.1f}% {row['btc']*100:>+14.1f}%")
    print(f"  {'ETH overlay':<20s} {b.get('eth',0)*100:>+9.1f}% {row['eth']*100:>+14.1f}%")
    print(f"  {'Avg core exposure':<20s} {'':>10s} {row['core_gross_exposure']:>14.2f}")
    print(f"  {'Avg BTC position':<20s} {'':>10s} {row['btc_pos']:>+14.3f}")

# Overall comparison
eq = 100000 * (1 + net).cumprod()
dd = eq / eq.cummax() - 1
ann_ret = (eq.iloc[-1]/eq.iloc[0])**(1/1.34) - 1
ann_vol = net.std() * np.sqrt(8760)

print(f"\n--- Overall Performance ---")
print(f"  {'Metric':<25s} {'Baseline':>10s} {'With Emergency':>15s}")
print(f"  {'─'*50}")
print(f"  {'Total Return':<25s} {'80.12%':>10s} {(eq.iloc[-1]/eq.iloc[0]-1)*100:>14.2f}%")
print(f"  {'Sharpe':<25s} {'1.119':>10s} {ann_ret/ann_vol:>14.3f}")
print(f"  {'Max DD':<25s} {'-54.05%':>10s} {dd.min()*100:>14.2f}%")
print(f"  {'Turnover':<25s} {'1.90%':>10s} {pos.diff().abs().sum(axis=1).mean()*100:>14.2f}%")
