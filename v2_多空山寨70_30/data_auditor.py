"""
数据审计与清洗模块
基于解决方案.md中的数据清洗指南实现
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from config import DATA_FILES, DATA_DIR, OUTPUT_DIR


class DataAuditor:
    """数据质量审计与清洗"""
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        """
        传入包含 open, high, low, close, volume 的字典
        """
        self.data = data_dict
        self.report = []
        self.dead_coins: list = []
        
    def log(self, msg):
        print(f"[Audit] {msg}")
        self.report.append(msg)
        
    def check_physical_integrity(self):
        """第一层：物理完整性检查"""
        self.log("=" * 60)
        self.log("--- 1. Physical Integrity Check (物理完整性检查) ---")
        
        # 1. 检查形状是否对齐
        shapes = {k: v.shape for k, v in self.data.items()}
        self.log(f"Data Shapes: {shapes}")
        if len(set(shapes.values())) > 1:
            self.log("❌ CRITICAL: Data shapes are NOT aligned!")
        else:
            self.log("✅ Shapes are aligned.")
            
        # 2. 检查NaN比例
        close = self.data['close']
        nan_pct = close.isna().mean().mean() * 100
        self.log(f"Overall Missing Data (NaN): {nan_pct:.2f}%")
        
        # 3. 检查时间索引连续性
        full_idx = pd.date_range(start=close.index[0], end=close.index[-1], freq='1h')
        missing_timestamps = full_idx.difference(close.index)
        if len(missing_timestamps) > 0:
            self.log(f"❌ Found {len(missing_timestamps)} missing timestamps!")
            self.log(f"   First few missing: {list(missing_timestamps[:5])}")
        else:
            self.log("✅ Timestamp continuity is perfect.")
            
        # 4. 检查重复索引
        duplicate_idx = close.index.duplicated().sum()
        if duplicate_idx > 0:
            self.log(f"❌ Found {duplicate_idx} duplicate timestamps!")
        else:
            self.log("✅ No duplicate timestamps.")
            
    def check_logic_errors(self):
        """第二层：金融逻辑检查 (OHLC一致性)"""
        self.log("\n" + "=" * 60)
        self.log("--- 2. Financial Logic Check (金融逻辑检查) ---")
        
        o = self.data['open']
        h = self.data['high']
        l = self.data['low']
        c = self.data['close']
        v = self.data.get('volume', self.data.get('quote_volume'))
        
        # 检查 H >= L
        invalid_hl = (h < l).sum().sum()
        if invalid_hl > 0:
            self.log(f"❌ Found {invalid_hl} cases where High < Low")
        else:
            self.log("✅ High >= Low: OK")
            
        # 检查 H >= O, C
        invalid_h_o = (h < o).sum().sum()
        invalid_h_c = (h < c).sum().sum()
        if invalid_h_o > 0 or invalid_h_c > 0:
            self.log(f"❌ Found {invalid_h_o + invalid_h_c} cases where High < Open/Close")
        else:
            self.log("✅ High >= Open, Close: OK")
            
        # 检查 L <= O, C
        invalid_l_o = (l > o).sum().sum()
        invalid_l_c = (l > c).sum().sum()
        if invalid_l_o > 0 or invalid_l_c > 0:
            self.log(f"❌ Found {invalid_l_o + invalid_l_c} cases where Low > Open/Close")
        else:
            self.log("✅ Low <= Open, Close: OK")
            
        # 检查零/负价格
        zeros_o = (o <= 0).sum().sum()
        zeros_h = (h <= 0).sum().sum()
        zeros_l = (l <= 0).sum().sum()
        zeros_c = (c <= 0).sum().sum()
        total_zeros = zeros_o + zeros_h + zeros_l + zeros_c
        if total_zeros > 0:
            self.log(f"❌ Found {total_zeros} cases with Price <= 0")
            self.log(f"   Open<=0: {zeros_o}, High<=0: {zeros_h}, Low<=0: {zeros_l}, Close<=0: {zeros_c}")
        else:
            self.log("✅ All prices > 0: OK")
            
        # 检查负成交量
        if v is not None:
            neg_vol = (v < 0).sum().sum()
            if neg_vol > 0:
                self.log(f"❌ Found {neg_vol} cases with Volume < 0")
            else:
                self.log("✅ All volumes >= 0: OK")
                
    def check_outliers(self):
        """第三层：统计与异常值检查"""
        self.log("\n" + "=" * 60)
        self.log("--- 3. Statistical Outliers Check (异常值检查) ---")
        
        c = self.data['close']
        returns = c.pct_change()
        
        # 定义阈值
        upper_threshold = 1.0   # +100%
        lower_threshold = -0.8  # -80%
        
        # 统计异常数量
        huge_pumps = (returns > upper_threshold).sum().sum()
        huge_dumps = (returns < lower_threshold).sum().sum()
        
        self.log(f"🚀 Pumps (>100% in 1h): {huge_pumps} occurrences")
        self.log(f"💥 Dumps (<-50% in 1h): {huge_dumps} occurrences")
        
        # 更宽松阈值统计
        pumps_50 = (returns > 0.5).sum().sum()
        dumps_30 = (returns < -0.3).sum().sum()
        self.log(f"   Pumps >50%: {pumps_50}, Dumps <-30%: {dumps_30}")
        
        # 打印最极端的几个值
        flat_ret = returns.stack()
        max_val = flat_ret.max()
        min_val = flat_ret.min()
        
        self.log(f"Extreme Max Return: {max_val*100:.2f}%")
        self.log(f"Extreme Min Return: {min_val*100:.2f}%")
        
        # 找出最极端的币种
        if max_val > 2.0:
            max_idx = flat_ret.idxmax()
            self.log(f"   Max occurs at: {max_idx}")
        if min_val < -0.9:
            min_idx = flat_ret.idxmin()
            self.log(f"   Min occurs at: {min_idx}")
            
        if max_val > 2.0 or min_val < -0.9:
            self.log("⚠️ WARNING: Extremely dirty data detected! Cleaning required.")
            
    def check_dead_coins(self, threshold_hours: int = 72):
        """检测停牌/死币"""
        self.log("\n" + "=" * 60)
        self.log("--- 4. Dead Coins Check (死币检测) ---")
        
        c = self.data['close']
        v = self.data.get('volume', self.data.get('quote_volume'))
        
        dead_coins = []
        for symbol in c.columns:
            # 检查连续N小时价格不变
            price_unchanged = (c[symbol].diff() == 0).rolling(threshold_hours).sum()
            max_unchanged = price_unchanged.max()
            
            # 检查连续N小时零成交
            if v is not None:
                zero_volume = (v[symbol] == 0).rolling(threshold_hours).sum()
                max_zero_vol = zero_volume.max()
            else:
                max_zero_vol = 0
                
            if max_unchanged >= threshold_hours or max_zero_vol >= threshold_hours:
                dead_coins.append(symbol)
                
        self.dead_coins = dead_coins
        if dead_coins:
            self.log(f"⚠️ Found {len(dead_coins)} potential dead/illiquid coins:")
            self.log(f"   {dead_coins[:10]}...")
        else:
            self.log("✅ No dead coins detected.")
            
    def run_full_audit(self):
        """运行完整审计"""
        self.log("\n" + "=" * 60)
        self.log("DATA QUALITY AUDIT REPORT")
        self.log("=" * 60)
        
        self.check_physical_integrity()
        self.check_logic_errors()
        self.check_outliers()
        self.check_dead_coins()
        
        self.log("\n" + "=" * 60)
        self.log("AUDIT COMPLETE")
        self.log("=" * 60)
        
        return self.report
        
    def clean_data(self, max_iterations: int = 5) -> Dict[str, pd.DataFrame]:
        """执行数据清洗（迭代清洗直到无极端值）"""
        self.log("\n" + "=" * 60)
        self.log("--- DATA CLEANING ---")
        self.log("=" * 60)
        
        cleaned_data = {}
        
        # 1. 强制对齐时间索引 (填充缺失时间戳)
        close = self.data['close']
        full_idx = pd.date_range(start=close.index[0], end=close.index[-1], freq='1h')
        
        self.log(f"Original index length: {len(close.index)}")
        self.log(f"Full index length: {len(full_idx)}")
        
        for name, df in self.data.items():
            # 重建索引
            df_clean = df.reindex(full_idx)
            
            # 2. 处理价格 <= 0 的情况
            if name in ['open', 'high', 'low', 'close']:
                invalid_count = (df_clean <= 0).sum().sum()
                df_clean[df_clean <= 0] = np.nan
                if invalid_count > 0:
                    self.log(f"  {name}: Set {invalid_count} non-positive values to NaN")
                    
            cleaned_data[name] = df_clean
            
        # 3. 修复OHLC逻辑错误
        o = cleaned_data['open']
        h = cleaned_data['high']
        l = cleaned_data['low']
        c = cleaned_data['close']
        
        # High < Low -> 整行设为NaN
        invalid_hl_mask = h < l
        invalid_hl_count = invalid_hl_mask.sum().sum()
        if invalid_hl_count > 0:
            for name in ['open', 'high', 'low', 'close']:
                cleaned_data[name] = cleaned_data[name].where(~invalid_hl_mask)
            self.log(f"  Fixed {invalid_hl_count} High < Low errors (set to NaN)")
            
        # 4. 迭代清洗极端收益率（处理NaN边界产生的新极端值）
        total_extreme_removed = 0
        for iteration in range(max_iterations):
            c = cleaned_data['close']
            returns = c.pct_change(fill_method=None)
            
            # 标记极端收益 (>100% 或 <-80%)
            extreme_mask = (returns > 1.0) | (returns < -0.8)
            extreme_count = extreme_mask.sum().sum()
            
            if extreme_count == 0:
                self.log(f"  Iteration {iteration+1}: No more extreme values found.")
                break
                
            # 将极端收益时刻的价格设为NaN
            for name in ['open', 'high', 'low', 'close']:
                cleaned_data[name] = cleaned_data[name].where(~extreme_mask)
            total_extreme_removed += extreme_count
            self.log(f"  Iteration {iteration+1}: Removed {extreme_count} extreme points")
            
        self.log(f"  Total extreme return points removed: {total_extreme_removed}")
            
        # 5. 重新计算returns（使用fill_method=None避免警告）
        cleaned_returns = cleaned_data['close'].pct_change(fill_method=None)
        
        # 6. 对returns进行clip截断（保守处理剩余的中等异常值）
        # 单小时收益限制在 [-50%, +50%]
        cleaned_returns = cleaned_returns.clip(lower=-0.5, upper=0.5)
        cleaned_data['returns'] = cleaned_returns
        
        # 统计最终结果
        final_max = cleaned_returns.max().max()
        final_min = cleaned_returns.min().min()
        self.log(f"  Final returns range: [{final_min*100:.2f}%, {final_max*100:.2f}%]")
        
        self.log("✅ Data cleaning completed.")
        
        return cleaned_data
        
    def save_cleaned_data(self, cleaned_data: Dict[str, pd.DataFrame], output_dir: Path):
        """保存清洗后的数据"""
        output_dir.mkdir(exist_ok=True)
        
        self.log(f"\nSaving cleaned data to {output_dir}...")
        
        for name, df in cleaned_data.items():
            output_path = output_dir / f'cross_section_{name}_cleaned.parquet'
            df.to_parquet(output_path)
            self.log(f"  Saved: {output_path.name} ({df.shape})")
            
        self.log("✅ All cleaned data saved.")


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """加载原始数据"""
    data = {}
    print("Loading raw data...")
    for name, path in DATA_FILES.items():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        data[name] = df
        print(f"  {name}: {df.shape}")
    return data


def main():
    """主函数：审计并清洗数据"""
    # 1. 加载原始数据
    raw_data = load_raw_data()
    
    # 2. 创建审计器
    auditor = DataAuditor(raw_data)
    
    # 3. 运行审计
    auditor.run_full_audit()
    
    # 4. 执行清洗
    cleaned_data = auditor.clean_data()
    
    # 5. 保存清洗后的数据
    auditor.save_cleaned_data(cleaned_data, DATA_DIR)
    
    # 6. 再次审计清洗后的数据
    print("\n" + "=" * 60)
    print("RE-AUDITING CLEANED DATA...")
    print("=" * 60)
    auditor2 = DataAuditor(cleaned_data)
    auditor2.run_full_audit()
    
    return cleaned_data


if __name__ == "__main__":
    main()
