
import pandas as pd
import os

file_path = r"D:\strategy\TFT\factor\merged_tft_data.parquet"

try:
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")
    
    df = pd.read_parquet(file_path)
    print("Successfully read parquet file.")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Index: {df.index.min()} to {df.index.max()}")
    print("Content check passed.")
except Exception as e:
    print(f"Error reading file: {e}")
