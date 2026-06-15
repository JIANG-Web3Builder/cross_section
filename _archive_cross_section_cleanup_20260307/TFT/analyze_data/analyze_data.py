
import pandas as pd
import os
import json

data_dir = r"D:\strategy\TFT\factor"
files = [f for f in os.listdir(data_dir) if f.endswith('.csv') or f.endswith('.parquet')]

analysis_results = {}

for file in files:
    file_path = os.path.join(data_dir, file)
    try:
        if file.endswith('.csv'):
            df = pd.read_csv(file_path, nrows=5) # Read a few rows to infer schema
            # Re-read full file for null checks if small enough, or just get info
            # For 3MB files it's fine to read all
            df = pd.read_csv(file_path)
        else:
            df = pd.read_parquet(file_path)
        
        info = {
            "columns": list(df.columns),
            "shape": df.shape,
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "head": df.head(3).to_dict(orient='records'),
            "time_column_candidates": [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        }
        
        # Try to identify time range if possible
        for col in info['time_column_candidates']:
            try:
                # Convert to datetime if not already
                if df[col].dtype == 'object' or 'int' in str(df[col].dtype):
                    # explicit conversion often needed
                    pass 
                # Just taking min/max as string for now to avoid parsing errors in this quick script
                info[f"{col}_min"] = str(df[col].min())
                info[f"{col}_max"] = str(df[col].max())
            except:
                pass

        analysis_results[file] = info
        print(f"Successfully analyzed {file}")
        
    except Exception as e:
        analysis_results[file] = {"error": str(e)}
        print(f"Error analyzing {file}: {e}")

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

# Output results to json for me to read
with open(r"D:\strategy\TFT\factor\data_analysis_temp.json", "w") as f:
    json.dump(analysis_results, f, indent=4, cls=DateTimeEncoder)
