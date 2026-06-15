
import pandas as pd
import os

files_to_check = [
    "BTCUSDT_combined_1H.parquet",
    "ETHUSDT_combined_1H.parquet",
    "btc_dominance_1h.parquet",
    "known_time.parquet"
]

data_dir = r"D:\strategy\TFT\factor"

for file in files_to_check:
    file_path = os.path.join(data_dir, file)
    try:
        df = pd.read_parquet(file_path)
        print(f"--- {file} ---")
        print(f"Index name: {df.index.name}")
        print(f"Index type: {type(df.index)}")
        print(f"Index head: {df.index[:3]}")
        print(f"Index min: {df.index.min()}")
        print(f"Index max: {df.index.max()}")
        print("\n")
    except Exception as e:
        print(f"Error reading {file}: {e}")
