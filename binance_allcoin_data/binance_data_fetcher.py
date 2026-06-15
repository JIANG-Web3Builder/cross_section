"""
币安合约数据获取模块 - 高性能版本

特性:
1. 并行下载 - 多线程加速10倍+
2. Parquet存储 - 比CSV快10倍，压缩率高
3. 增量更新 - 只下载缺失数据
4. 历史数据 - 支持从data.binance.vision批量下载

API限制:
- 公共API: 1200次/分钟
- 认证API: 更高限制
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import requests
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import io
import warnings
warnings.filterwarnings('ignore')

try:
    from binance.client import Client
    HAS_BINANCE = True
except ImportError:
    HAS_BINANCE = False
    print("提示: pip install python-binance 可使用认证API")

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("提示: pip install pyarrow 可使用Parquet存储(更快)")


# ============= 你的API配置 =============
API_KEY = "HttTcAH5vB4kmCAmyarcZKD5DIkwGasxpu7LVMC4rh3zelLhDzcVfWqFvsRBBq1O"
API_SECRET = "cGGOLCrfFSdIP1KV7amEYZ1pRxCPPxpRLUrPItHr6yHIVeMUrQJtTI5wjeiTXivv"
# ========================================


class BinanceFuturesDataFetcher:
    """币安永续合约数据获取器 - 高性能版本"""
    
    BASE_URL = "https://fapi.binance.com"
    VISION_URL = "https://data.binance.vision/data/futures/um"
    
    def __init__(
        self, 
        data_dir: str = "data",
        api_key: str = None,
        api_secret: str = None,
        use_parquet: bool = True
    ):
        """
        初始化
        
        Args:
            data_dir: 数据存储目录
            api_key: 币安API Key
            api_secret: 币安API Secret
            use_parquet: 是否使用Parquet格式存储
        """
        self.data_dir = data_dir
        self.use_parquet = use_parquet and HAS_PARQUET
        os.makedirs(data_dir, exist_ok=True)
        
        # 初始化客户端 (可选，用于需要认证的接口)
        self.client = None
        self.api_key = api_key
        self.api_secret = api_secret
        
        if api_key and api_secret and HAS_BINANCE:
            try:
                # 使用合约API基础URL
                self.client = Client(api_key, api_secret, tld='com')
                self.client.API_URL = 'https://fapi.binance.com/fapi'
                print("✓ 币安API客户端已初始化")
            except Exception as e:
                print(f"⚠ API客户端初始化失败(不影响公开接口): {e}")
                self.client = None
        
        # 存储目录
        self.klines_dir = os.path.join(data_dir, "klines")
        self.daily_cap_dir = os.path.join(data_dir, "daily_cap")
        os.makedirs(self.klines_dir, exist_ok=True)
        os.makedirs(self.daily_cap_dir, exist_ok=True)
    
    def get_all_symbols(self, max_retries: int = 3) -> List[str]:
        """获取所有USDT永续合约交易对"""
        url = f"{self.BASE_URL}/fapi/v1/exchangeInfo"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15)
                data = response.json()
                
                if 'symbols' not in data:
                    print(f"⚠ 响应格式异常，重试 ({attempt+1}/{max_retries})")
                    time.sleep(2)
                    continue
                
                symbols = [
                    s['symbol'] for s in data['symbols']
                    if s['contractType'] == 'PERPETUAL' 
                    and s['quoteAsset'] == 'USDT'
                    and s['status'] == 'TRADING'
                ]
                
                print(f"✓ 获取到 {len(symbols)} 个USDT永续合约")
                return symbols
                
            except Exception as e:
                print(f"⚠ 获取交易对失败 ({attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        
        print("✗ 获取交易对失败，请检查网络")
        return []
    
    def _get_klines_single(
        self, 
        symbol: str, 
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """获取单个交易对的K线数据 (内部方法)"""
        url = f"{self.BASE_URL}/fapi/v1/klines"
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': int(current_start.timestamp() * 1000),
                'endTime': int(end_time.timestamp() * 1000),
                'limit': 1500
            }
            
            try:
                response = requests.get(url, params=params, timeout=15)
                data = response.json()
                
                if isinstance(data, dict) and 'code' in data:
                    break
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # 更新起始时间
                last_close_time = data[-1][6]
                current_start = datetime.fromtimestamp(last_close_time / 1000) + timedelta(milliseconds=1)
                
                if len(data) < 1500:
                    break
                    
                time.sleep(0.05)  # 避免限流
                
            except Exception as e:
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        
        df['symbol'] = symbol
        
        # 去重
        df = df.drop_duplicates(subset=['open_time', 'symbol'])
        
        return df
    
    def download_klines_parallel(
        self,
        symbols: List[str] = None,
        interval: str = "1h",
        days: int = 365,
        max_workers: int = 10
    ) -> pd.DataFrame:
        """
        并行下载多个交易对的K线数据
        
        Args:
            symbols: 交易对列表，None表示所有
            interval: K线周期 (1m, 5m, 15m, 1h, 4h, 1d)
            days: 下载多少天
            max_workers: 并行线程数
            
        Returns:
            合并后的DataFrame
        """
        if symbols is None:
            symbols = self.get_all_symbols()
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        print(f"\n开始并行下载 {len(symbols)} 个交易对的 {interval} K线数据")
        print(f"时间范围: {start_time.date()} ~ {end_time.date()}")
        print(f"并行线程: {max_workers}")
        
        all_data = []
        completed = 0
        failed = []
        
        def download_one(symbol):
            try:
                df = self._get_klines_single(symbol, interval, start_time, end_time)
                return symbol, df, None
            except Exception as e:
                return symbol, None, str(e)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one, s): s for s in symbols}
            
            for future in as_completed(futures):
                symbol, df, error = future.result()
                completed += 1
                
                if error:
                    failed.append(symbol)
                elif df is not None and len(df) > 0:
                    all_data.append(df)
                
                print(f"\r  进度: {completed}/{len(symbols)} | 成功: {len(all_data)} | 失败: {len(failed)}", end="")
        
        print()
        
        if not all_data:
            print("✗ 未获取到任何数据")
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        
        # 保存
        self._save_klines(result, interval)
        
        print(f"✓ 下载完成: {len(result)} 条记录, {result['symbol'].nunique()} 个币种")
        
        return result
    
    def _save_klines(self, df: pd.DataFrame, interval: str):
        """保存K线数据"""
        date_str = datetime.now().strftime('%Y%m%d')
        
        if self.use_parquet:
            filepath = os.path.join(self.klines_dir, f"klines_{interval}_{date_str}.parquet")
            df.to_parquet(filepath, index=False, compression='snappy')
        else:
            filepath = os.path.join(self.klines_dir, f"klines_{interval}_{date_str}.csv")
            df.to_csv(filepath, index=False)
        
        print(f"✓ 数据已保存: {filepath}")
        return filepath
    
    def load_klines(self, interval: str = "1h") -> pd.DataFrame:
        """
        加载K线数据 (自动选择最新文件)
        
        Args:
            interval: K线周期
            
        Returns:
            K线DataFrame
        """
        pattern = f"klines_{interval}_"
        
        # 查找最新文件
        files = []
        for f in os.listdir(self.klines_dir):
            if f.startswith(pattern):
                files.append(os.path.join(self.klines_dir, f))
        
        if not files:
            print(f"未找到 {interval} K线数据，请先下载")
            return pd.DataFrame()
        
        # 按修改时间排序，取最新
        latest_file = max(files, key=os.path.getmtime)
        
        print(f"加载K线数据: {latest_file}")
        
        if latest_file.endswith('.parquet'):
            df = pd.read_parquet(latest_file)
        else:
            df = pd.read_csv(latest_file)
            df['open_time'] = pd.to_datetime(df['open_time'])
            df['close_time'] = pd.to_datetime(df['close_time'])
        
        return df
    
    # ============= 每日市值快照 =============
    
    def get_daily_market_cap(self, date: datetime = None) -> pd.DataFrame:
        """
        获取某一天的市值代理数据
        
        Args:
            date: 日期，None表示今天
            
        Returns:
            市值DataFrame
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y%m%d')
        
        # 检查缓存
        cache_file = os.path.join(self.daily_cap_dir, f"market_cap_{date_str}.parquet" if self.use_parquet else f"market_cap_{date_str}.csv")
        
        if os.path.exists(cache_file):
            if self.use_parquet:
                return pd.read_parquet(cache_file)
            else:
                return pd.read_csv(cache_file)
        
        # 获取24h行情
        url = f"{self.BASE_URL}/fapi/v1/ticker/24hr"
        
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            
            df = pd.DataFrame(data)
            
            # 筛选USDT永续
            df = df[df['symbol'].str.endswith('USDT')]
            
            # 转换数值
            for col in ['lastPrice', 'volume', 'quoteVolume', 'priceChangePercent']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['date'] = date.date()
            df['market_cap_proxy'] = df['quoteVolume']  # 24h成交额作为市值代理
            df['cap_rank'] = df['market_cap_proxy'].rank(ascending=False)
            
            # 保存
            if self.use_parquet:
                df.to_parquet(cache_file, index=False)
            else:
                df.to_csv(cache_file, index=False)
            
            return df
            
        except Exception as e:
            print(f"获取市值数据失败: {e}")
            return pd.DataFrame()
    
    def download_historical_market_cap(
        self, 
        days: int = 365,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        下载历史每日市值数据
        
        原理: 使用日K线的成交额作为每日市值代理
        
        Args:
            days: 天数
            interval: 使用的K线周期 (建议1d)
            
        Returns:
            历史市值DataFrame
        """
        print(f"下载 {days} 天历史市值数据...")
        
        # 先下载日K线
        df = self.download_klines_parallel(interval=interval, days=days, max_workers=10)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # 按日期提取市值
        df['date'] = df['open_time'].dt.date
        
        daily_cap = df.groupby(['date', 'symbol']).agg({
            'quote_volume': 'sum',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        daily_cap['market_cap_proxy'] = daily_cap['quote_volume']
        
        # 计算每日排名
        daily_cap['cap_rank'] = daily_cap.groupby('date')['market_cap_proxy'].rank(ascending=False)
        daily_cap['cap_percentile'] = daily_cap.groupby('date')['market_cap_proxy'].rank(pct=True)
        
        # 保存
        if self.use_parquet:
            filepath = os.path.join(self.daily_cap_dir, f"historical_cap_{days}d.parquet")
            daily_cap.to_parquet(filepath, index=False)
        else:
            filepath = os.path.join(self.daily_cap_dir, f"historical_cap_{days}d.csv")
            daily_cap.to_csv(filepath, index=False)
        
        print(f"✓ 历史市值数据已保存: {filepath}")
        print(f"  - 日期范围: {daily_cap['date'].min()} ~ {daily_cap['date'].max()}")
        print(f"  - 币种数量: {daily_cap['symbol'].nunique()}")
        
        return daily_cap
    
    # ============= data.binance.vision 批量下载 =============
    
    def get_vision_symbols(self, interval: str = "1h", data_type: str = "daily") -> List[str]:
        """
        从 data.binance.vision 获取所有币种列表
        
        这个方法会获取到所有有历史数据的币种，包括已退市的
        
        Args:
            interval: K线周期
            data_type: 'daily' 或 'monthly'
            
        Returns:
            币种列表
        """
        import xml.etree.ElementTree as ET
        
        # data.binance.vision 使用S3格式，可以通过XML列出目录
        base_url = f"https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
        prefix = f"data/futures/um/{data_type}/klines/"
        
        all_symbols = set()
        marker = ""
        
        print(f"从 data.binance.vision 获取全部币种列表...")
        
        while True:
            url = f"{base_url}?prefix={prefix}&delimiter=/&marker={marker}"
            
            try:
                response = requests.get(url, timeout=30)
                if response.status_code != 200:
                    print(f"请求失败: {response.status_code}")
                    break
                
                # 解析XML
                root = ET.fromstring(response.content)
                
                # 提取命名空间
                ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
                
                # 获取CommonPrefixes (子目录)
                prefixes = root.findall('.//s3:CommonPrefixes/s3:Prefix', ns)
                
                for p in prefixes:
                    # 格式: data/futures/um/daily/klines/BTCUSDT/
                    path = p.text
                    symbol = path.split('/')[-2]  # 获取币种名
                    if symbol.endswith('USDT'):
                        all_symbols.add(symbol)
                
                # 检查是否有更多
                is_truncated = root.find('.//s3:IsTruncated', ns)
                if is_truncated is not None and is_truncated.text == 'true':
                    next_marker = root.find('.//s3:NextMarker', ns)
                    if next_marker is not None:
                        marker = next_marker.text
                    else:
                        # 使用最后一个prefix作为marker
                        if prefixes:
                            marker = prefixes[-1].text
                        else:
                            break
                else:
                    break
                    
            except Exception as e:
                print(f"获取币种列表失败: {e}")
                break
        
        symbols = sorted(list(all_symbols))
        print(f"✓ 获取到 {len(symbols)} 个币种 (包含已退市)")
        
        return symbols
    
    def get_vision_available_dates(self, symbol: str, interval: str = "1h") -> List[str]:
        """
        获取某个币种在vision上可用的所有日期
        
        Args:
            symbol: 交易对
            interval: K线周期
            
        Returns:
            可用日期列表 ['2023-01-01', '2023-01-02', ...]
        """
        import xml.etree.ElementTree as ET
        
        base_url = f"https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
        prefix = f"data/futures/um/daily/klines/{symbol}/{interval}/"
        
        all_dates = []
        marker = ""
        
        while True:
            url = f"{base_url}?prefix={prefix}&marker={marker}"
            
            try:
                response = requests.get(url, timeout=30)
                if response.status_code != 200:
                    break
                
                root = ET.fromstring(response.content)
                ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
                
                # 获取文件列表
                contents = root.findall('.//s3:Contents/s3:Key', ns)
                
                for c in contents:
                    # 格式: data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2023-01-01.zip
                    key = c.text
                    if key.endswith('.zip'):
                        # 提取日期
                        filename = key.split('/')[-1]  # BTCUSDT-1h-2023-01-01.zip
                        date_part = filename.replace(f'{symbol}-{interval}-', '').replace('.zip', '')
                        all_dates.append(date_part)
                
                # 检查是否有更多
                is_truncated = root.find('.//s3:IsTruncated', ns)
                if is_truncated is not None and is_truncated.text == 'true':
                    if contents:
                        marker = contents[-1].text
                    else:
                        break
                else:
                    break
                    
            except Exception as e:
                break
        
        return sorted(all_dates)
    
    def download_from_vision(
        self,
        symbols: List[str],
        interval: str = "1h",
        start_date: str = "2023-01-01",
        end_date: str = None,
        data_type: str = "daily",  # daily 或 monthly
        max_workers: int = 5
    ) -> pd.DataFrame:
        """
        从 data.binance.vision 批量下载历史数据
        
        适合下载超过1年的完整历史数据
        
        Args:
            symbols: 交易对列表
            interval: K线周期
            start_date: 开始日期
            end_date: 结束日期，None表示今天
            data_type: 'daily' 按天下载, 'monthly' 按月下载
            max_workers: 并行数
            
        Returns:
            合并后的DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        print(f"\n从 data.binance.vision 下载历史数据")
        print(f"币种: {len(symbols)} 个")
        print(f"周期: {interval}")
        print(f"范围: {start_date} ~ {end_date}")
        
        all_data = []
        
        def download_symbol_data(symbol):
            """下载单个币种的全部历史数据"""
            symbol_data = []
            
            if data_type == "monthly":
                # 按月下载
                current = start.replace(day=1)
                while current <= end:
                    month_str = current.strftime('%Y-%m')
                    url = f"{self.VISION_URL}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month_str}.zip"
                    
                    df = self._download_vision_zip(url, symbol)
                    if df is not None:
                        symbol_data.append(df)
                    
                    # 下个月
                    if current.month == 12:
                        current = current.replace(year=current.year + 1, month=1)
                    else:
                        current = current.replace(month=current.month + 1)
            else:
                # 按天下载
                current = start
                while current <= end:
                    date_str = current.strftime('%Y-%m-%d')
                    url = f"{self.VISION_URL}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
                    
                    df = self._download_vision_zip(url, symbol)
                    if df is not None:
                        symbol_data.append(df)
                    
                    current += timedelta(days=1)
            
            if symbol_data:
                return pd.concat(symbol_data, ignore_index=True)
            return None
        
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_symbol_data, s): s for s in symbols}
            
            for future in as_completed(futures):
                symbol = futures[future]
                completed += 1
                
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        all_data.append(df)
                        print(f"\r  [{completed}/{len(symbols)}] {symbol}: {len(df)} 条", end="")
                except Exception as e:
                    print(f"\r  [{completed}/{len(symbols)}] {symbol}: 失败 - {e}", end="")
        
        print()
        
        if not all_data:
            print("✗ 未下载到任何数据")
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['open_time', 'symbol'])
        result = result.sort_values(['symbol', 'open_time'])
        
        # 保存
        filename = f"vision_{interval}_{start_date}_{end_date}.parquet" if self.use_parquet else f"vision_{interval}_{start_date}_{end_date}.csv"
        filepath = os.path.join(self.klines_dir, filename)
        
        if self.use_parquet:
            result.to_parquet(filepath, index=False)
        else:
            result.to_csv(filepath, index=False)
        
        print(f"✓ 下载完成: {len(result)} 条记录")
        print(f"  保存至: {filepath}")
        
        return result
    
    def _download_vision_zip(self, url: str, symbol: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """下载并解析vision zip文件"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=60)
                if response.status_code != 200:
                    return None
                
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    for filename in z.namelist():
                        with z.open(filename) as f:
                            # 先读取看看第一行是否为header
                            df = pd.read_csv(f, header=None)
                            
                            # 检查第一行是否为列名 (open_time是字符串)
                            if str(df.iloc[0, 0]) == 'open_time':
                                df = df.iloc[1:]  # 跳过header行
                                df = df.reset_index(drop=True)
                            
                            # 只取前12列
                            df = df.iloc[:, :12]
                            df.columns = [
                                'open_time', 'open', 'high', 'low', 'close', 'volume',
                                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                                'taker_buy_quote_volume', 'ignore'
                            ]
                            
                            df['symbol'] = symbol
                            
                            # 转换时间戳
                            df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
                            df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce')
                            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                            
                            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            # 删除无效行
                            df = df.dropna(subset=['open_time', 'close'])
                            
                            return df
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue
        return None
    
    def download_all_vision_data(
        self,
        interval: str = "1h",
        start_date: str = None,
        end_date: str = None,
        max_workers: int = 10,
        save_per_symbol: bool = True
    ) -> pd.DataFrame:
        """
        从 data.binance.vision 下载全部币种的完整历史数据
        
        这是官方完整数据，包含所有上市过的币种（含已退市）
        
        Args:
            interval: K线周期 (1m, 5m, 15m, 1h, 4h, 1d等)
            start_date: 开始日期，None表示从最早开始
            end_date: 结束日期，None表示到最新
            max_workers: 并行线程数
            save_per_symbol: 是否每个币种单独保存
            
        Returns:
            合并后的DataFrame
        """
        print("\n" + "="*60)
        print("    从 data.binance.vision 下载完整历史数据")
        print("="*60)
        
        # 1. 获取全部币种
        symbols = self.get_vision_symbols(interval=interval, data_type="daily")
        
        if not symbols:
            print("✗ 未获取到币种列表")
            return pd.DataFrame()
        
        print(f"\n开始下载 {len(symbols)} 个币种的 {interval} 数据")
        if start_date:
            print(f"时间范围: {start_date} ~ {end_date or '最新'}")
        else:
            print("时间范围: 全部历史")
        print(f"并行线程: {max_workers}")
        print("-"*60)
        
        # 2. 准备日期过滤
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else datetime.now()
        
        all_data = []
        completed = 0
        success_count = 0
        total_records = 0
        
        def download_symbol_all_dates(symbol):
            """下载单个币种的全部可用数据"""
            # 获取该币种的所有可用日期
            available_dates = self.get_vision_available_dates(symbol, interval)
            
            if not available_dates:
                return None, 0
            
            # 过滤日期范围
            if start_dt or end_dt:
                filtered_dates = []
                for d in available_dates:
                    try:
                        dt = pd.to_datetime(d)
                        if start_dt and dt < start_dt:
                            continue
                        if end_dt and dt > end_dt:
                            continue
                        filtered_dates.append(d)
                    except:
                        continue
                available_dates = filtered_dates
            
            if not available_dates:
                return None, 0
            
            # 下载每一天的数据
            symbol_data = []
            for date_str in available_dates:
                url = f"{self.VISION_URL}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
                df = self._download_vision_zip(url, symbol, max_retries=2)
                if df is not None and len(df) > 0:
                    symbol_data.append(df)
            
            if symbol_data:
                result = pd.concat(symbol_data, ignore_index=True)
                return result, len(result)
            return None, 0
        
        # 3. 并行下载
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_symbol_all_dates, s): s for s in symbols}
            
            for future in as_completed(futures):
                symbol = futures[future]
                completed += 1
                
                try:
                    df, count = future.result()
                    if df is not None and len(df) > 0:
                        success_count += 1
                        total_records += count
                        
                        if save_per_symbol:
                            # 单独保存每个币种
                            symbol_file = os.path.join(self.klines_dir, f"{symbol}_{interval}.parquet")
                            df.to_parquet(symbol_file, index=False)
                        else:
                            all_data.append(df)
                        
                        print(f"\r  [{completed}/{len(symbols)}] 成功: {success_count} | 总记录: {total_records:,}", end="")
                    else:
                        print(f"\r  [{completed}/{len(symbols)}] 成功: {success_count} | {symbol}: 无数据", end="")
                except Exception as e:
                    print(f"\r  [{completed}/{len(symbols)}] {symbol}: 失败", end="")
        
        print("\n" + "-"*60)
        
        # 4. 合并保存
        if save_per_symbol:
            print(f"✓ 数据已按币种保存到: {self.klines_dir}")
            print(f"  - 成功币种: {success_count}")
            print(f"  - 总记录数: {total_records:,}")
            
            # 返回合并数据的方法
            return self.load_all_symbol_data(interval)
        else:
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                result = result.drop_duplicates(subset=['open_time', 'symbol'])
                result = result.sort_values(['symbol', 'open_time'])
                
                # 保存
                filename = f"vision_all_{interval}.parquet"
                filepath = os.path.join(self.klines_dir, filename)
                result.to_parquet(filepath, index=False)
                
                print(f"✓ 全部数据已保存: {filepath}")
                print(f"  - 币种数: {result['symbol'].nunique()}")
                print(f"  - 记录数: {len(result):,}")
                
                return result
        
        return pd.DataFrame()
    
    def load_all_symbol_data(self, interval: str = "1h") -> pd.DataFrame:
        """
        加载所有单独保存的币种数据
        
        Args:
            interval: K线周期
            
        Returns:
            合并后的DataFrame
        """
        pattern = f"_{interval}.parquet"
        files = [f for f in os.listdir(self.klines_dir) if f.endswith(pattern) and not f.startswith('vision_')]
        
        if not files:
            print(f"未找到 {interval} 数据文件")
            return pd.DataFrame()
        
        print(f"加载 {len(files)} 个币种数据...")
        
        all_data = []
        for f in files:
            filepath = os.path.join(self.klines_dir, f)
            df = pd.read_parquet(filepath)
            all_data.append(df)
        
        result = pd.concat(all_data, ignore_index=True)
        print(f"✓ 加载完成: {len(result):,} 条记录, {result['symbol'].nunique()} 个币种")
        
        return result
    
    # ============= 增量更新 =============
    
    def update_klines(self, interval: str = "1h") -> pd.DataFrame:
        """
        增量更新K线数据
        
        自动检测已有数据的最新时间，只下载缺失部分
        
        Args:
            interval: K线周期
            
        Returns:
            更新后的完整DataFrame
        """
        existing_df = self.load_klines(interval)
        
        if len(existing_df) == 0:
            print("无现有数据，执行完整下载")
            return self.download_klines_parallel(interval=interval, days=365)
        
        # 找到最新时间
        latest_time = existing_df['open_time'].max()
        days_to_update = (datetime.now() - latest_time).days + 1
        
        if days_to_update <= 0:
            print("数据已是最新")
            return existing_df
        
        print(f"增量更新: 从 {latest_time} 开始，更新 {days_to_update} 天")
        
        # 下载新数据
        new_df = self.download_klines_parallel(interval=interval, days=days_to_update)
        
        if len(new_df) == 0:
            return existing_df
        
        # 合并去重
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['open_time', 'symbol'])
        combined = combined.sort_values(['symbol', 'open_time'])
        
        # 保存
        self._save_klines(combined, interval)
        
        print(f"✓ 更新完成: 总计 {len(combined)} 条记录")
        
        return combined


def quick_download(days: int = 365, interval: str = "1h"):
    """快速下载脚本"""
    fetcher = BinanceFuturesDataFetcher(
        api_key=API_KEY,
        api_secret=API_SECRET
    )
    
    # 下载K线
    df = fetcher.download_klines_parallel(interval=interval, days=days, max_workers=10)
    
    return df


if __name__ == "__main__":
    # 初始化 (公开接口不需要API Key)
    fetcher = BinanceFuturesDataFetcher()
    
    print("\n" + "="*60)
    print("        币安合约数据下载器")
    print("="*60)
    
    # 获取所有交易对
    symbols = fetcher.get_all_symbols()
    
    # 选项1: 下载过去1年的1小时K线 (约需要10-15分钟)
    print("\n选项1: 下载1年1小时K线数据")
    print("  命令: df = fetcher.download_klines_parallel(interval='1h', days=365)")
    
    # 选项2: 下载历史市值数据
    print("\n选项2: 下载历史每日市值数据")
    print("  命令: cap_df = fetcher.download_historical_market_cap(days=365)")
    
    # 选项3: 从vision下载更长历史
    print("\n选项3: 从data.binance.vision下载更长历史(如3年)")
    print("  命令: fetcher.download_from_vision(symbols[:50], '1h', '2022-01-01')")
    
    # 演示: 下载最近30天数据作为测试
    print("\n" + "-"*60)
    print("正在下载最近30天数据作为测试...")
    
    df = fetcher.download_klines_parallel(interval='1d', days=30, max_workers=10)
    
    if len(df) > 0:
        print(f"\n数据概览:")
        print(f"  - 时间范围: {df['open_time'].min()} ~ {df['open_time'].max()}")
        print(f"  - 币种数量: {df['symbol'].nunique()}")
        print(f"  - 总记录数: {len(df)}")
