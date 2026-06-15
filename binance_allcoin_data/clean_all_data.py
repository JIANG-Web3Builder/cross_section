"""
BTC/ETH数据清洗脚本
1. 刚性时间网格 - resample到1H
2. 极值截断 (Winsorization) - 0.5%/99.5%
3. 精度压缩 - float32
4. 保存为Parquet (snappy)
"""

import pandas as pd
import numpy as np
import os


def clean_and_resample(symbol: str, base_dir: str, output_dir: str):
    """清洗并resample单个币种数据"""
    prefix = symbol.replace("USDT", "")
    
    print(f"\n{'='*50}")
    print(f"处理 {symbol}")
    print(f"{'='*50}")
    
    # 1. 加载 fundingRate
    fr_path = os.path.join(base_dir, f"data_{prefix}_fundingRate", f"{symbol}_fundingRate_combined.csv")
    print(f"\n加载 fundingRate: {fr_path}")
    fr_df = pd.read_csv(fr_path)
    fr_df['calc_time'] = pd.to_datetime(fr_df['calc_time'])
    fr_df = fr_df.set_index('calc_time').sort_index()
    print(f"  原始: {len(fr_df)} 条, {fr_df.index.min()} ~ {fr_df.index.max()}")
    
    # 2. 加载 metrics
    m_path = os.path.join(base_dir, f"data_{prefix}_metric", f"{symbol}_metrics_combined.csv")
    print(f"\n加载 metrics: {m_path}")
    m_df = pd.read_csv(m_path)
    m_df['create_time'] = pd.to_datetime(m_df['create_time'])
    if 'symbol' in m_df.columns:
        m_df = m_df.drop(columns=['symbol'])
    m_df = m_df.set_index('create_time').sort_index()
    print(f"  原始: {len(m_df)} 条, {m_df.index.min()} ~ {m_df.index.max()}")
    
    # 3. 建立完整1H时间索引
    start_time = min(fr_df.index.min(), m_df.index.min()).floor('h')
    end_time = max(fr_df.index.max(), m_df.index.max()).ceil('h')
    full_idx = pd.date_range(start=start_time, end=end_time, freq='1h')
    print(f"\n完整时间索引: {len(full_idx)} 小时 ({start_time} ~ {end_time})")
    
    # 4. Resample fundingRate (每8h一条 -> 1h ffill)
    print("\nResample fundingRate -> 1H (ffill)")
    fr_1h = fr_df.resample('1h').first().reindex(full_idx).ffill()
    
    # 5. Resample metrics (每5min一条 -> 1h聚合)
    print("Resample metrics -> 1H (mean/last)")
    agg_dict = {col: 'last' if 'open_interest' in col else 'mean' for col in m_df.columns}
    m_1h = m_df.resample('1h').agg(agg_dict).reindex(full_idx).ffill()
    
    # 6. 极值截断 (Winsorization 0.5%/99.5%)
    print("\n极值截断 (0.5% ~ 99.5%)")
    for df in [fr_1h, m_1h]:
        for col in df.select_dtypes(include=[np.number]).columns:
            lower, upper = df[col].quantile(0.005), df[col].quantile(0.995)
            clipped = ((df[col] < lower) | (df[col] > upper)).sum()
            if clipped > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                print(f"  {col}: 截断 {clipped} 个")
    
    # 7. 转float32
    print("\n转换为 float32")
    for df in [fr_1h, m_1h]:
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].astype(np.float32)
    
    # 8. 合并
    combined = pd.concat([fr_1h, m_1h], axis=1)
    
    # 9. 保存
    os.makedirs(output_dir, exist_ok=True)
    
    fr_out = os.path.join(output_dir, f"{symbol}_fundingRate_1H.parquet")
    fr_1h.to_parquet(fr_out, compression='snappy')
    print(f"\n✓ 保存: {fr_out} ({len(fr_1h)} 行)")
    
    m_out = os.path.join(output_dir, f"{symbol}_metrics_1H.parquet")
    m_1h.to_parquet(m_out, compression='snappy')
    print(f"✓ 保存: {m_out} ({len(m_1h)} 行)")
    
    comb_out = os.path.join(output_dir, f"{symbol}_combined_1H.parquet")
    combined.to_parquet(comb_out, compression='snappy')
    print(f"✓ 保存: {comb_out} ({len(combined)} 行)")
    
    # 验证
    print(f"\n验证:")
    gaps = combined.index.to_series().diff()
    max_gap = gaps.max()
    missing = combined.isnull().sum().sum()
    print(f"  最大时间间隔: {max_gap}")
    print(f"  缺失值: {missing}")
    
    return combined


def main():
    base_dir = r"D:\strategy\binance_allcoin_data"
    output_dir = os.path.join(base_dir, "data_cleaned_1H")
    
    print("="*60)
    print("数据清洗: Resample to 1H + Winsorization + float32")
    print("="*60)
    
    for symbol in ["BNBUSDT","SOLUSDT"]:
        clean_and_resample(symbol, base_dir, output_dir)
    
    print("\n" + "="*60)
    print("清洗完成!")
    print(f"输出目录: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
