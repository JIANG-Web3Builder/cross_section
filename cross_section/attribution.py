"""
详细归因分析脚本 — 策略收益分解与风险分析
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from config import OUTPUT_DIR, DATA_FILES, TAKER_FEE, SLIPPAGE

# Load data
pos = pd.read_parquet(OUTPUT_DIR / 'positions.parquet')
lpos = pd.read_parquet(OUTPUT_DIR / 'long_positions.parquet')
spos = pd.read_parquet(OUTPUT_DIR / 'short_positions.parquet')
raw_close = pd.read_parquet(DATA_FILES['close'])
raw_ret = raw_close.pct_change(fill_method=None)
mkt = pd.read_parquet(OUTPUT_DIR / 'market_index.parquet')

cidx = raw_ret.index.intersection(pos.index)
ccols = raw_ret.columns.intersection(pos.columns)
cost_rate = TAKER_FEE + SLIPPAGE

# === Gross & Net ===
gross = (pos.shift(1).loc[cidx, ccols] * raw_ret.loc[cidx, ccols]).sum(axis=1)
prev = pos.shift(1).fillna(0)
real_trades = ((prev == 0) & (pos != 0)) | ((prev != 0) & (pos == 0)) | ((prev * pos) < 0)
tc = (pos.diff().abs() * real_trades.astype(float)).sum(axis=1) * cost_rate
net = gross - tc
eq = 100000 * (1 + net).cumprod()

print("=" * 60)
print(" DETAILED ATTRIBUTION ANALYSIS")
print("=" * 60)

# === 1. PnL Decomposition ===
alt_cols = [c for c in ccols if c not in ['BTCUSDT', 'ETHUSDT']]
long_pnl = (lpos.shift(1).reindex(columns=alt_cols, fill_value=0).loc[cidx] * raw_ret.loc[cidx, alt_cols]).sum(axis=1)
short_pnl = (spos.shift(1).reindex(columns=alt_cols, fill_value=0).loc[cidx] * raw_ret.loc[cidx, alt_cols]).sum(axis=1)

btc_pnl = pd.Series(0.0, index=cidx)
eth_pnl = pd.Series(0.0, index=cidx)
if 'BTCUSDT' in ccols:
    btc_pnl = pos['BTCUSDT'].shift(1).loc[cidx] * raw_ret.loc[cidx, 'BTCUSDT']
if 'ETHUSDT' in ccols:
    eth_pnl = pos['ETHUSDT'].shift(1).loc[cidx] * raw_ret.loc[cidx, 'ETHUSDT']

print("\n--- 1. PnL Decomposition ---")
print(f"  Long altcoins (cum):  {(1+long_pnl).prod()-1:>+8.2%}")
print(f"  Short altcoins (cum): {(1+short_pnl).prod()-1:>+8.2%}")
print(f"  BTC overlay (sum):    {btc_pnl.sum():>+8.2%}")
print(f"  ETH overlay (sum):    {eth_pnl.sum():>+8.2%}")
print(f"  Total gross (cum):    {(1+gross).prod()-1:>+8.2%}")
print(f"  Total cost:           {-tc.sum():>+8.2%}")
print(f"  Total net (cum):      {(1+net).prod()-1:>+8.2%}")

# === 2. Monthly Returns ===
net_monthly = net.resample('M').apply(lambda x: (1+x).prod()-1)
print("\n--- 2. Monthly Returns ---")
for dt, r in net_monthly.items():
    n_plus = max(0, int(r * 200))
    n_minus = max(0, int(-r * 200))
    bar = "+" * n_plus + "-" * n_minus
    print(f"  {dt.strftime('%Y-%m')}: {r:>+7.2%} {bar}")

# === 3. Quarterly Returns ===
net_quarterly = net.resample('Q').apply(lambda x: (1+x).prod()-1)
print("\n--- 3. Quarterly Returns ---")
for dt, r in net_quarterly.items():
    q = (dt.month - 1) // 3 + 1
    print(f"  {dt.year}-Q{q}: {r:>+7.2%}")

# === 4. Rolling Metrics ===
roll_window = 720  # 30 days
roll_ret = net.rolling(roll_window).apply(lambda x: (1+x).prod()-1, raw=False)
roll_vol = net.rolling(roll_window).std() * np.sqrt(8760)
roll_sharpe = net.rolling(roll_window).apply(
    lambda x: x.mean()/x.std()*np.sqrt(8760) if x.std()>0 else 0, raw=False)

print("\n--- 4. Rolling 30-day Stats ---")
print(f"  Return range: [{roll_ret.min():.2%}, {roll_ret.max():.2%}], median={roll_ret.median():.2%}")
print(f"  Sharpe range: [{roll_sharpe.min():.2f}, {roll_sharpe.max():.2f}], median={roll_sharpe.median():.2f}")
print(f"  % time Sharpe > 0: {(roll_sharpe > 0).mean():.1%}")

# === 5. Turnover ===
diff = pos.diff().abs().sum(axis=1)
rebal = diff[diff > 0.01]
print("\n--- 5. Turnover ---")
print(f"  Rebalance events: {len(rebal)}")
print(f"  Avg TO/rebalance: {rebal.mean()*100:.1f}%")
print(f"  Annual cost: {tc.sum()/1.34:.2%}")

# Overlap
rebal_idx = rebal.index
overlaps_l, overlaps_s = [], []
for i in range(1, len(rebal_idx)):
    prev_p = pos.loc[rebal_idx[i-1]]
    curr_p = pos.loc[rebal_idx[i]]
    pl = set(prev_p[prev_p > 0.01].index)
    cl = set(curr_p[curr_p > 0.01].index)
    ps = set(prev_p[prev_p < -0.01].index)
    cs = set(curr_p[curr_p < -0.01].index)
    if pl and cl:
        overlaps_l.append(len(pl & cl) / len(pl))
    if ps and cs:
        overlaps_s.append(len(ps & cs) / len(ps))

print(f"  Avg long overlap: {np.mean(overlaps_l):.1%}")
print(f"  Avg short overlap: {np.mean(overlaps_s):.1%}")

# === 6. Risk ===
dd = eq / eq.cummax() - 1
mkt_ret = mkt['returns'].loc[cidx]
port_beta = net.cov(mkt_ret) / mkt_ret.var() if mkt_ret.var() > 0 else 0
net_exp = pos.sum(axis=1)

print("\n--- 6. Risk ---")
print(f"  Max drawdown: {dd.min():.2%}")
print(f"  DD duration > 20%: {(dd < -0.2).sum()} hours")
print(f"  Avg net exposure: {net_exp.mean():.3f}")
print(f"  Portfolio beta: {port_beta:.3f}")
print(f"  Win rate: {(net > 0).mean():.1%}")

# === 7. Top/Bottom coins ===
# Average weight per symbol
avg_long_w = lpos.mean()
avg_short_w = spos.mean()
top_long = avg_long_w[avg_long_w > 0.001].sort_values(ascending=False).head(15)
top_short = avg_short_w[avg_short_w < -0.001].sort_values().head(15)

print("\n--- 7. Most Held Coins ---")
print("  Long:")
for sym, w in top_long.items():
    print(f"    {sym:<15s} avg_weight={w:.4f}")
print("  Short:")
for sym, w in top_short.items():
    print(f"    {sym:<15s} avg_weight={w:.4f}")

# === 8. Summary ===
ann_ret = (eq.iloc[-1]/eq.iloc[0])**(1/1.34) - 1
ann_vol = net.std() * np.sqrt(8760)

print("\n" + "=" * 60)
print(" FINAL SUMMARY")
print("=" * 60)
print(f"  Strategy: L/S Altcoin Core + BTC/ETH Timing Overlay")
print(f"  Period: {cidx[0].strftime('%Y-%m-%d')} to {cidx[-1].strftime('%Y-%m-%d')} ({1.34:.1f} years)")
print(f"  Total Return:     {(eq.iloc[-1]/eq.iloc[0]-1):>+8.2%}")
print(f"  Ann. Return:      {ann_ret:>+8.2%}")
print(f"  Ann. Volatility:  {ann_vol:>8.2%}")
print(f"  Sharpe Ratio:     {ann_ret/ann_vol:>8.3f}")
print(f"  Max Drawdown:     {dd.min():>8.2%}")
print(f"  Calmar Ratio:     {ann_ret/abs(dd.min()):>8.3f}")
print(f"  Avg Turnover:     {diff.mean()*100:>8.2f}%")
print(f"  Annual Cost:      {tc.sum()/1.34:>8.2%}")
print(f"  Beta:             {port_beta:>8.3f}")
print("=" * 60)
