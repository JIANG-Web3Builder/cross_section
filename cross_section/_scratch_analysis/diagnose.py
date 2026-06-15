"""
全面诊断脚本 — 排查持仓为0、IC decay缺失、持续亏损的根因
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path
from config import OUTPUT_DIR, LONG_N, REBALANCE_HOURS, FORWARD_HOURS

print("=" * 70)
print(" DIAGNOSTIC REPORT")
print("=" * 70)

# =============================================
# 1. 检查 predictions
# =============================================
print("\n\n>>> 1. PREDICTIONS ANALYSIS")
try:
    pred_file = OUTPUT_DIR / 'predictions.parquet'
    preds = pd.read_parquet(pred_file)
    print(f"  Shape: {preds.shape}")
    print(f"  Index (timestamps): {preds.index[0]} to {preds.index[-1]}, len={len(preds.index)}")
    print(f"  Columns (symbols): {len(preds.columns)}")
    print(f"  NaN ratio: {preds.isna().sum().sum() / preds.size * 100:.1f}%")
    print(f"  Non-NaN per row (mean): {preds.notna().sum(axis=1).mean():.1f}")
    print(f"  Non-NaN per row (min): {preds.notna().sum(axis=1).min()}")
    print(f"  Value range: [{preds.min().min():.4f}, {preds.max().max():.4f}]")
    print(f"  Mean: {preds.stack().mean():.4f}, Std: {preds.stack().std():.4f}")
    
    # 检查每个时间截面的预测分布
    sample_ts = preds.index[len(preds)//2]
    sample_pred = preds.loc[sample_ts].dropna()
    print(f"\n  Sample timestamp {sample_ts}:")
    print(f"    Non-NaN: {len(sample_pred)}")
    print(f"    Top 5: {sample_pred.nlargest(5).to_dict()}")
    print(f"    Bottom 5: {sample_pred.nsmallest(5).to_dict()}")
except Exception as e:
    print(f"  ERROR: {e}")

# =============================================
# 2. 检查 positions
# =============================================
print("\n\n>>> 2. POSITIONS ANALYSIS")
try:
    pos = pd.read_parquet(OUTPUT_DIR / 'positions.parquet')
    long_pos = pd.read_parquet(OUTPUT_DIR / 'long_positions.parquet')
    short_pos = pd.read_parquet(OUTPUT_DIR / 'short_positions.parquet')
    
    print(f"  Positions shape: {pos.shape}")
    print(f"  Long positions shape: {long_pos.shape}")
    print(f"  Short positions shape: {short_pos.shape}")
    
    # 持仓数量统计
    n_long = (long_pos > 0).sum(axis=1)
    n_short = (short_pos < 0).sum(axis=1)  # short权重是负数
    n_net = (pos != 0).sum(axis=1)
    
    print(f"\n  Long positions per bar: mean={n_long.mean():.1f}, min={n_long.min()}, max={n_long.max()}")
    print(f"  Short positions per bar: mean={n_short.mean():.1f}, min={n_short.min()}, max={n_short.max()}")
    print(f"  Non-zero positions per bar: mean={n_net.mean():.1f}, min={n_net.min()}, max={n_net.max()}")
    
    # 检查前几个非零时间点
    nonzero_bars = n_net[n_net > 0]
    print(f"\n  Bars with any position: {len(nonzero_bars)} / {len(n_net)}")
    if len(nonzero_bars) > 0:
        print(f"  First non-zero at: {nonzero_bars.index[0]}")
        print(f"  Last non-zero at: {nonzero_bars.index[-1]}")
        
        sample_bar = nonzero_bars.index[0]
        print(f"\n  Sample positions at {sample_bar}:")
        pos_at_bar = pos.loc[sample_bar]
        print(f"    Long: {pos_at_bar[pos_at_bar > 0].to_dict()}")
        print(f"    Short: {pos_at_bar[pos_at_bar < 0].to_dict()}")
    else:
        print("  *** ALL POSITIONS ARE ZERO! ***")
        
    # 权重总和检查
    total_long_weight = long_pos.sum(axis=1)
    total_short_weight = short_pos.sum(axis=1)
    total_net = pos.sum(axis=1)
    print(f"\n  Total long weight: mean={total_long_weight.mean():.4f}")
    print(f"  Total short weight: mean={total_short_weight.mean():.4f}")
    print(f"  Total net weight: mean={total_net.mean():.4f}")
    
except Exception as e:
    print(f"  ERROR: {e}")

# =============================================
# 3. 检查 universe 和 predictions 对齐
# =============================================
print("\n\n>>> 3. UNIVERSE vs PREDICTIONS ALIGNMENT")
try:
    universe = pd.read_parquet(OUTPUT_DIR / 'universe_mask.parquet')
    
    print(f"  Universe shape: {universe.shape}")
    print(f"  Universe index: {universe.index[0]} to {universe.index[-1]}")
    print(f"  Predictions index: {preds.index[0]} to {preds.index[-1]}")
    
    # 检查时间对齐
    common_ts = preds.index.intersection(universe.index)
    print(f"  Common timestamps: {len(common_ts)}")
    
    # 检查列对齐
    common_cols = set(preds.columns) & set(universe.columns)
    pred_only = set(preds.columns) - set(universe.columns)
    univ_only = set(universe.columns) - set(preds.columns)
    print(f"  Common symbols: {len(common_cols)}")
    print(f"  In predictions but not universe: {len(pred_only)}")
    print(f"  In universe but not predictions: {len(univ_only)}")
    if pred_only:
        print(f"    Pred-only examples: {list(pred_only)[:5]}")
    if univ_only:
        print(f"    Univ-only examples: {list(univ_only)[:5]}")
    
    # 在测试期内, 宇宙中有多少有效币
    test_universe = universe.loc[common_ts]
    n_in_univ = test_universe.sum(axis=1)
    print(f"  Universe size during test: mean={n_in_univ.mean():.0f}, min={n_in_univ.min()}")
    
except Exception as e:
    print(f"  ERROR: {e}")

# =============================================
# 4. 模拟 generate_positions 逻辑
# =============================================
print("\n\n>>> 4. SIMULATING generate_positions LOGIC")
try:
    rebalance_count = 0
    skip_timing = 0
    skip_universe = 0
    skip_too_few = 0
    successful = 0
    
    last_rebalance_idx = -REBALANCE_HOURS
    
    for i, ts in enumerate(preds.index):
        # 换仓频率控制
        if i - last_rebalance_idx < REBALANCE_HOURS:
            continue
        
        # 宇宙检查
        if ts not in universe.index:
            skip_universe += 1
            continue
        
        universe_mask = universe.loc[ts]
        pred = preds.loc[ts]
        
        # 过滤宇宙内 + 非NaN
        valid_syms = universe_mask.index[universe_mask]
        common = pred.index.intersection(valid_syms)
        pred_filtered = pred[common].dropna()
        pred_filtered = pred_filtered.drop('BTCUSDT', errors='ignore')
        
        if len(pred_filtered) < LONG_N * 2:
            skip_too_few += 1
            if i < 50 or skip_too_few <= 3:
                print(f"    ts={ts}: only {len(pred_filtered)} valid preds (need {LONG_N*2})")
            continue
        
        successful += 1
        last_rebalance_idx = i
        
        if successful <= 3:
            long_picks = pred_filtered.nlargest(LONG_N).index.tolist()
            short_picks = pred_filtered.nsmallest(LONG_N).index.tolist()
            print(f"    ts={ts}: {len(pred_filtered)} valid, "
                  f"long={long_picks[:3]}..., short={short_picks[:3]}...")
    
    total_bars = len(preds.index)
    possible_rebalances = total_bars // REBALANCE_HOURS
    print(f"\n  Total bars: {total_bars}")
    print(f"  Possible rebalances (every {REBALANCE_HOURS}h): {possible_rebalances}")
    print(f"  Skipped (universe miss): {skip_universe}")
    print(f"  Skipped (too few symbols): {skip_too_few}")
    print(f"  Successful rebalances: {successful}")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# =============================================
# 5. 收益归因分析
# =============================================
print("\n\n>>> 5. RETURN ATTRIBUTION ANALYSIS")
try:
    equity = pd.read_csv(OUTPUT_DIR / 'equity_curve.csv', index_col=0, parse_dates=True)
    equity = equity.iloc[:, 0] if isinstance(equity, pd.DataFrame) else equity
    
    metrics = pd.read_csv(OUTPUT_DIR / 'metrics.csv')
    print(f"  Total Return: {metrics['total_return'].iloc[0]*100:.2f}%")
    print(f"  Long Return: {metrics['long_return'].iloc[0]*100:.2f}%")
    print(f"  Short Return: {metrics['short_return'].iloc[0]*100:.2f}%")
    print(f"  Avg Turnover: {metrics['avg_turnover'].iloc[0]*100:.2f}%")
    
    # 加载原始收益率
    from config import DATA_FILES
    returns = pd.read_parquet(DATA_FILES['close']).pct_change()
    
    # 分析持仓的实际收益
    if len(nonzero_bars) > 0:
        # 取几个样本时间点，看持仓币种的下一步收益
        sample_bars = nonzero_bars.index[::max(1, len(nonzero_bars)//10)][:10]
        
        long_ret_list = []
        short_ret_list = []
        for bar in sample_bars:
            p = pos.loc[bar]
            long_syms = p[p > 0].index.tolist()
            short_syms = p[p < 0].index.tolist()
            
            # 下一个bar的收益
            bar_loc = returns.index.get_loc(bar)
            if bar_loc + 1 < len(returns):
                next_ret = returns.iloc[bar_loc + 1]
                if long_syms:
                    long_ret = next_ret[long_syms].mean()
                    long_ret_list.append(long_ret)
                if short_syms:
                    short_ret = next_ret[short_syms].mean()
                    short_ret_list.append(short_ret)
        
        if long_ret_list:
            print(f"\n  Sample long picks avg next-bar return: {np.nanmean(long_ret_list)*100:.4f}%")
        if short_ret_list:
            print(f"  Sample short picks avg next-bar return: {np.nanmean(short_ret_list)*100:.4f}%")
            print(f"  Spread (long - short): {(np.nanmean(long_ret_list) - np.nanmean(short_ret_list))*100:.4f}%")
    
    # 交易成本影响
    from config import TAKER_FEE, SLIPPAGE
    cost_rate = TAKER_FEE + SLIPPAGE
    avg_turnover = metrics['avg_turnover'].iloc[0]
    n_hours = metrics['n_hours'].iloc[0]
    n_years = metrics['n_years'].iloc[0]
    annual_cost = avg_turnover * cost_rate * 24 * 365  # 年化交易成本
    print(f"\n  Cost analysis:")
    print(f"    Single trade cost: {cost_rate*100:.3f}%")
    print(f"    Avg hourly turnover: {avg_turnover*100:.2f}%")
    print(f"    Estimated annual cost: {annual_cost*100:.1f}%")
    
    # 如果没有交易成本, 收益如何?
    print(f"\n  Without costs (gross return): analyzing...")
    pos_shifted = pos.shift(1)
    common_idx = returns.index.intersection(pos_shifted.index)
    common_cols = returns.columns.intersection(pos_shifted.columns)
    gross_ret = (pos_shifted.loc[common_idx, common_cols] * returns.loc[common_idx, common_cols]).sum(axis=1)
    gross_equity = 100000 * (1 + gross_ret).cumprod()
    gross_total = gross_equity.iloc[-1] / gross_equity.iloc[0] - 1
    print(f"    Gross total return (no costs): {gross_total*100:.2f}%")
    print(f"    Net total return (with costs): {metrics['total_return'].iloc[0]*100:.2f}%")
    print(f"    Cost impact: {(gross_total - metrics['total_return'].iloc[0])*100:.2f}%")
    
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# =============================================
# 6. 因子方向检查
# =============================================
print("\n\n>>> 6. FACTOR DIRECTION CHECK")
try:
    report = pd.read_csv(OUTPUT_DIR / 'factor_research' / 'factor_research_report.csv')
    
    # 所有因子的IC方向
    pos_ic = (report['rank_ic_mean'] > 0).sum()
    neg_ic = (report['rank_ic_mean'] < 0).sum()
    print(f"  Factors with positive IC: {pos_ic}")
    print(f"  Factors with negative IC: {neg_ic}")
    
    # 选中的因子
    try:
        import json
        sel_features = json.load(open(OUTPUT_DIR / 'selected_features.json'))
        sel_report = report[report['factor'].isin(sel_features)]
        print(f"\n  Selected factors ({len(sel_features)}):")
        for _, row in sel_report.iterrows():
            direction = "↑" if row['rank_ic_mean'] > 0 else "↓"
            print(f"    {row['factor']:<25s} IC={row['rank_ic_mean']:+.4f} {direction}  "
                  f"ICIR={row['rank_icir']:+.3f}  "
                  f"positive_ratio={row.get('positive_ratio', 0):.3f}")
    except:
        pass
    
    # 模型预测 vs 标签 相关性
    rolling_pred = pd.read_parquet(OUTPUT_DIR / 'rolling_predictions.parquet')
    print(f"\n  Rolling predictions shape: {rolling_pred.shape}")
    print(f"  Prediction value range: [{rolling_pred['prediction'].min():.4f}, {rolling_pred['prediction'].max():.4f}]")
    print(f"  Prediction mean: {rolling_pred['prediction'].mean():.4f}")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# =============================================
# 7. 检查 visualizer 输出
# =============================================
print("\n\n>>> 7. OUTPUT FILES CHECK")
output_files = list(OUTPUT_DIR.glob('*'))
for f in sorted(output_files):
    if f.is_file():
        size = f.stat().st_size
        print(f"  {f.name:<45s} {size:>10,} bytes")
    elif f.is_dir():
        n_files = len(list(f.glob('*')))
        print(f"  {f.name + '/':<45s} {n_files} files")

print("\n" + "=" * 70)
print(" DIAGNOSIS COMPLETE")
print("=" * 70)
