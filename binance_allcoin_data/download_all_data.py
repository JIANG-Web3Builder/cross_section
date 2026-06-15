"""
下载BTC/ETH fundingRate和metrics数据
从 data.binance.vision 获取2021年至今的数据
只保存combined文件
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import requests
import zipfile
import io
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET


class DataDownloader:
    """fundingRate和metrics数据下载器"""
    
    VISION_URL = "https://data.binance.vision/data/futures/um"
    S3_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def get_available_dates(self, symbol: str, data_type: str, freq: str = "daily") -> List[str]:
        """获取可用日期"""
        prefix = f"data/futures/um/{freq}/{data_type}/{symbol}/"
        all_dates = []
        marker = ""
        
        print(f"  获取 {symbol} {data_type} 可用日期...", end="")
        
        while True:
            url = f"{self.S3_URL}?prefix={prefix}&marker={marker}"
            try:
                response = requests.get(url, timeout=30)
                if response.status_code != 200:
                    break
                
                root = ET.fromstring(response.content)
                ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
                contents = root.findall('.//s3:Contents/s3:Key', ns)
                
                for c in contents:
                    key = c.text
                    if key.endswith('.zip'):
                        filename = key.split('/')[-1]
                        parts = filename.replace('.zip', '').split('-')
                        if freq == "monthly" and len(parts) >= 3:
                            all_dates.append(f"{parts[-2]}-{parts[-1]}")
                        elif len(parts) >= 4:
                            all_dates.append(f"{parts[-3]}-{parts[-2]}-{parts[-1]}")
                
                is_truncated = root.find('.//s3:IsTruncated', ns)
                if is_truncated is not None and is_truncated.text == 'true':
                    marker = contents[-1].text if contents else ""
                    if not marker:
                        break
                else:
                    break
            except Exception as e:
                print(f" 错误: {e}")
                break
        
        print(f" 找到 {len(all_dates)} 个")
        return sorted(all_dates)
    
    def download_zip(self, url: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """下载并解析zip文件"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=60)
                if response.status_code != 200:
                    return None
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    for filename in z.namelist():
                        with z.open(filename) as f:
                            return pd.read_csv(f)
            except:
                if attempt < max_retries - 1:
                    time.sleep(1)
        return None
    
    def download_funding_rate(self, symbol: str, start_date: str, end_date: str, max_workers: int = 10) -> pd.DataFrame:
        """下载fundingRate数据"""
        start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
        
        available_months = self.get_available_dates(symbol, "fundingRate", freq="monthly")
        filtered = [m for m in available_months if start_dt.year <= int(m.split('-')[0]) <= end_dt.year]
        
        print(f"  下载 {len(filtered)} 个月数据...", end="")
        
        all_data = []
        def dl(m):
            return self.download_zip(f"{self.VISION_URL}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip")
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for df in ex.map(dl, filtered):
                if df is not None and len(df) > 0:
                    all_data.append(df)
        
        if not all_data:
            print(" 无数据")
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True).drop_duplicates()
        
        # 时间转换
        time_col = next((c for c in result.columns if 'time' in c.lower()), None)
        if time_col:
            result[time_col] = pd.to_numeric(result[time_col], errors='coerce')
            if result[time_col].iloc[0] > 1e12:
                result[time_col] = pd.to_datetime(result[time_col], unit='ms')
            else:
                result[time_col] = pd.to_datetime(result[time_col], unit='s')
            result = result.sort_values(time_col)
            result = result[(result[time_col] >= start_dt) & (result[time_col] <= end_dt)]
        
        print(f" {len(result)} 条记录")
        return result.reset_index(drop=True)
    
    def download_metrics(self, symbol: str, start_date: str, end_date: str, max_workers: int = 10) -> pd.DataFrame:
        """下载metrics数据"""
        start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
        
        available_dates = self.get_available_dates(symbol, "metrics", freq="daily")
        filtered = [d for d in available_dates if start_dt <= pd.to_datetime(d) <= end_dt]
        
        print(f"  下载 {len(filtered)} 天数据...", end=" ")
        
        all_data = []
        completed = 0
        
        def dl(d):
            return self.download_zip(f"{self.VISION_URL}/daily/metrics/{symbol}/{symbol}-metrics-{d}.zip")
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = list(ex.map(dl, filtered))
            for df in futures:
                if df is not None and len(df) > 0:
                    all_data.append(df)
        
        if not all_data:
            print("无数据")
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True).drop_duplicates()
        
        time_col = 'create_time' if 'create_time' in result.columns else None
        if time_col:
            result[time_col] = pd.to_datetime(result[time_col])
            result = result.sort_values(time_col)
            result = result[(result[time_col] >= start_dt) & (result[time_col] <= end_dt)]
        
        print(f"{len(result)} 条记录")
        return result.reset_index(drop=True)


def main():
    output_dir = r"D:\strategy\binance_allcoin_data"
    downloader = DataDownloader(output_dir)
    
    start_date = "2021-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    symbols = ['BNBUSDT','SOLUSDT','']
    
    print("="*60)
    print(f"下载 fundingRate & metrics 数据")
    print(f"时间范围: {start_date} ~ {end_date}")
    print("="*60)
    
    for symbol in symbols:
        prefix = symbol.replace("USDT", "")
        
        print(f"\n[{symbol}]")
        
        # fundingRate
        fr_dir = os.path.join(output_dir, f"data_{prefix}_fundingRate")
        os.makedirs(fr_dir, exist_ok=True)
        
        funding_df = downloader.download_funding_rate(symbol, start_date, end_date)
        if not funding_df.empty:
            path = os.path.join(fr_dir, f"{symbol}_fundingRate_combined.csv")
            funding_df.to_csv(path, index=False)
            print(f"  ✓ 保存: {path}")
        
        # metrics
        m_dir = os.path.join(output_dir, f"data_{prefix}_metric")
        os.makedirs(m_dir, exist_ok=True)
        
        metrics_df = downloader.download_metrics(symbol, start_date, end_date)
        if not metrics_df.empty:
            path = os.path.join(m_dir, f"{symbol}_metrics_combined.csv")
            metrics_df.to_csv(path, index=False)
            print(f"  ✓ 保存: {path}")
    
    print("\n" + "="*60)
    print("下载完成!")
    print("="*60)


if __name__ == "__main__":
    main()
