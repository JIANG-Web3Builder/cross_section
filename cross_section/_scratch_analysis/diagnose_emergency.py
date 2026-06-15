"""诊断紧急通道触发频率"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from config import DATA_FILES, EMERGENCY_BTC_WINDOW, EMERGENCY_BTC_THRESHOLD, EMERGENCY_COOLDOWN

btc = pd.read_parquet(DATA_FILES['close'])['BTCUSDT']
ret12 = btc.pct_change(EMERGENCY_BTC_WINDOW)

crashes = ret12 < -EMERGENCY_BTC_THRESHOLD
rallies = ret12 > EMERGENCY_BTC_THRESHOLD

print(f"BTC 12h crashes (< -{EMERGENCY_BTC_THRESHOLD*100:.0f}%): {crashes.sum()} hours")
print(f"BTC 12h rallies (> +{EMERGENCY_BTC_THRESHOLD*100:.0f}%): {rallies.sum()} hours")
print(f"Total hours: {len(ret12)}")
print(f"Crash pct: {crashes.mean()*100:.1f}%")
print(f"Rally pct: {rallies.mean()*100:.1f}%")

# With cooldown simulation
triggered = []
last_trigger = -EMERGENCY_COOLDOWN
for i in range(len(ret12)):
    r = ret12.iloc[i]
    if abs(r) >= EMERGENCY_BTC_THRESHOLD and (i - last_trigger) >= EMERGENCY_COOLDOWN:
        triggered.append((ret12.index[i], r, 'CRASH' if r < 0 else 'RALLY'))
        last_trigger = i

print(f"\nWith {EMERGENCY_COOLDOWN}h cooldown: {len(triggered)} triggers")
for ts, r, typ in triggered:
    print(f"  {ts.strftime('%Y-%m-%d %H:%M')} | {typ:6s} | BTC 12h: {r*100:+.1f}%")
