"""诊断RSI牛熊期间山寨币的实际表现, 判断timing方向是否正确"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from config import DATA_FILES, TIMING_RSI_PERIOD, TIMING_RSI_BULL, TIMING_RSI_BEAR, TIMING_EMA_PERIOD_12H

btc = pd.read_parquet(DATA_FILES['close'])['BTCUSDT']
returns = pd.read_parquet(DATA_FILES['returns'])

# BTC Daily RSI
daily = btc.resample('1D').last().dropna()
delta = daily.diff()
gain = delta.where(delta > 0, 0)
loss = (-delta).where(delta < 0, 0)
ag = gain.ewm(span=TIMING_RSI_PERIOD, adjust=False).mean()
al = loss.ewm(span=TIMING_RSI_PERIOD, adjust=False).mean()
rs = ag / (al + 1e-10)
rsi = 100 - (100 / (1 + rs))
rsi_hourly = rsi.reindex(btc.index, method='ffill')

# 12h EMA(120)
p12h = btc.resample('12h').last().dropna()
ema12h = p12h.ewm(span=TIMING_EMA_PERIOD_12H, adjust=False).mean()
ema_hourly = ema12h.reindex(btc.index, method='ffill')

# Regimes
bull = rsi_hourly > TIMING_RSI_BULL
bear = rsi_hourly < TIMING_RSI_BEAR
neutral = (~bull) & (~bear)
ema_bull = neutral & (btc > ema_hourly)
ema_bear = neutral & (btc <= ema_hourly)

# BTC returns
btc_ret = btc.pct_change()

# Alt returns (equal weight, exclude BTC/ETH)
alt_cols = [c for c in returns.columns if c not in ['BTCUSDT', 'ETHUSDT']]
alt_avg_ret = returns[alt_cols].mean(axis=1)

# Top10 - Bottom10 spread (simple proxy for cross-sectional alpha)
# Using rolling rank
top10_ret = returns[alt_cols].apply(lambda x: x.nlargest(10).mean(), axis=1)
bot10_ret = returns[alt_cols].apply(lambda x: x.nsmallest(10).mean(), axis=1)
ls_spread = top10_ret - bot10_ret  # cross-sectional alpha

# Focus on test period only (last ~30%)
test_start = btc.index[int(len(btc) * 0.7)]
mask = btc.index >= test_start

print("=" * 70)
print(f" RSI TIMING REGIME ANALYSIS (Test period: {test_start.strftime('%Y-%m-%d')} onward)")
print("=" * 70)

for name, regime_mask in [
    (f'RSI>{TIMING_RSI_BULL} (Bull)', bull & mask),
    (f'RSI<{TIMING_RSI_BEAR} (Bear)', bear & mask),
    ('Neutral + EMA Bull', ema_bull & mask),
    ('Neutral + EMA Bear', ema_bear & mask),
]:
    n = regime_mask.sum()
    if n == 0:
        continue
    
    btc_avg = btc_ret[regime_mask].mean() * 100
    alt_avg = alt_avg_ret[regime_mask].mean() * 100
    spread_avg = ls_spread[regime_mask].mean() * 100
    btc_cum = (1 + btc_ret[regime_mask]).prod() - 1
    alt_cum = (1 + alt_avg_ret[regime_mask]).prod() - 1
    
    print(f"\n--- {name} ({n} hours, {n/mask.sum()*100:.1f}%) ---")
    print(f"  BTC avg hourly: {btc_avg:+.4f}%  (cum: {btc_cum*100:+.1f}%)")
    print(f"  Alt avg hourly: {alt_avg:+.4f}%  (cum: {alt_cum*100:+.1f}%)")
    print(f"  Alt-BTC spread: {(alt_avg-btc_avg):+.4f}%")
    print(f"  L/S spread:     {spread_avg:+.4f}%")
    
    # What the strategy does in this regime
    if 'Bull' in name:
        print(f"  → Strategy: net LONG alts + SHORT BTC/ETH")
        print(f"    Expected PnL contribution: {'POSITIVE' if alt_avg > 0 else 'NEGATIVE'} (alts {'up' if alt_avg > 0 else 'down'})")
    elif 'Bear' in name:
        print(f"  → Strategy: net SHORT alts + LONG BTC/ETH")
        print(f"    Expected PnL contribution: {'POSITIVE' if alt_avg < 0 else 'NEGATIVE'} (alts {'down' if alt_avg < 0 else 'up'})")
