# Data Description and Interpretation for TFT Training

This document describes the data files located in `D:\strategy\TFT\factor`, their contents, schemas, and how they should be processed for Temporal Fusion Transformer (TFT) training.

## 1. Data Inventory & Schema

### 1.1 Price & Volume Data (OHLCV)
**Files**: `BTCUSDT_1h.csv`, `ETHUSDT_1h.csv`
- **Description**: Hourly spot price data from Binance.
- **Key Columns**:
  - `open_time`: Timestamp string (e.g., "2020-09-30 16:00:00"). **Needs parsing**.
  - `open`, `high`, `low`, `close`: Price data (float).
  - `volume`: Trading volume.
- **Time Range**: ~Sep 2020 to Jan 2026.
- **Processing**: Convert `open_time` to datetime, set as index. Rename columns to include symbol prefix (e.g., `BTC_close`).

### 1.2 Market Metrics (Funding, Open Interest, Sentiment)
**Files**: `BTCUSDT_combined_1H.parquet`, `ETHUSDT_combined_1H.parquet`
- **Description**: Derivatives market data including funding rates and open interest.
- **Key Columns**:
  - `funding_interval_hours`, `last_funding_rate`: Funding metrics.
  - `sum_open_interest`, `sum_open_interest_value`: Market depth/participation.
  - `long_short_ratio` variants: Sentiment indicators.
- **Index**: `DatetimeIndex` is already present.
- **Time Range**: Jan 2022 to Jan 2026.
- **Notes**: Significant missing values (NaN) in early periods or specific columns (`long_short_ratio`). Requires imputation or forward-filling.

### 1.3 Factor Data (Technical Indicators & Alpha)
**Files**: `BTCUSDT_factors.parquet`, `ETHUSDT_factors.parquet`
- **Description**: Pre-calculated technical indicators and alpha factors.
- **Key Columns**:
  - `timestamp`: Datetime column.
  - `log_return_1h`, `log_return_4h`, `log_return_24h`: Targets or features.
  - `rsi_14_normalized`, `macd_hist_norm`, `volatility_30`: Standard technicals.
  - `alphaXXX`: Proprietary alpha factors.
- **Time Range**: Jan 2021 to Jan 2026.
- **Processing**: Align on `timestamp`.

### 1.4 Global/Macro Features
**File**: `btc_dominance_1h.parquet`
- **Description**: Bitcoin dominance metrics.
- **Columns**: `btcdom_change_zscore`, `btcdom_level_zscore`.
- **Index**: `DatetimeIndex` (UTC).
- **Time Range**: Jan 2022 to Jan 2026.

**File**: `btc_spx_correlation.parquet`
- **Description**: Correlation between BTC and S&P 500 (Macro correlation).
- **Columns**: `timestamp`, `corr_7d`, `corr_30d`, `corr_90d`.
- **Time Range**: Jan 2021 to Jan 2026.

### 1.5 Time Covariates
**File**: `known_time.parquet`
- **Description**: Pre-calculated time embeddings (sine/cosine of hour, day, month) and market events (halving, US market hours).
- **Issue**: Uses a `RangeIndex` (0 to 52560) with no explicit timestamp mapping.
- **Recommendation**: **Regenerate these features** dynamically during the merge process to ensure 100% alignment with the actual timestamps of the merged dataset. Do not rely on the row order of this file without a join key.

## 2. Merge Strategy for AI

1.  **Standardization**:
    -   Convert all time columns to `datetime64[ns]` (UTC).
    -   Resample all data to strict 1H frequency to handle any missing rows (fill gaps).

2.  **Alignment**:
    -   Use `outer` join on the timestamp index to preserve all data, but trim to the common useful period (likely starting **Jan 1, 2022** where most metrics overlap, or **Jan 1, 2021** with filled NaNs for metrics that start later).
    -   Primary spine: `BTCUSDT_1h.csv` timestamps.

3.  **Naming Convention**:
    -   Prefix asset-specific columns: `BTC_close`, `ETH_funding_rate`, etc.
    -   Keep global columns as is: `btc_spx_corr_30d`.

4.  **Missing Value Handling**:
    -   **Price/Volume**: Should be continuous. Forward fill small gaps.
    -   **Factors**: Forward fill.
    -   **Metrics (Funding/OI)**: Zero fill might be misleading; Forward fill is preferred.
    -   **Correlations**: Forward fill.

5.  **Target Generation**:
    -   Ensure `log_return_Xh` columns are correctly aligned (looking forward or backward? usually backward in factors, but for training labels we need future returns). *Note: The existing factor files likely contain PAST returns. Future targets need to be shifted.*


## 3. Output Requirements
The final merged file should be a single Parquet file (`merged_tft_data.parquet`) with:
-   Index: `timestamp`
-   Columns: ~100+ features combining all inputs.
-   Metadata: A separate JSON or CSV report detailing missing value counts per column.

## 4. Merge Analysis & Data Quality Report (Post-Merge)

**Merged File**: `merged_tft_data.parquet`
**Time Range**: 2020-09-30 16:00:00 to 2026-01-05 15:00:00
**Total Rows**: 46,146
**Total Columns**: 101
**File Size**: ~22.5 MB (Optimized)
**Compression**: Snappy
**Precision**: float32 (for efficiency)

### 4.1 Missing Data Segments
Based on the merge report, there are distinct "start dates" for different feature groups. The dataset is **not** fully populated for the entire history.

1.  **Price/Volume (OHLCV)**:
    -   Available from **Sep 2020**.
    -   Fully populated (0 missing after fill).

2.  **Market Metrics (Funding/OI/LongShort)**:
    -   Missing for the first **~11,000 rows** (approx. Sep 2020 to Jan 2022).
    -   **Action**: These features are NaN for the first 1.3 years. If they are critical for the model, training should likely start from **Jan 1, 2022**.

3.  **Factor Data & Correlations**:
    -   Missing for the first **~2,200 - 4,500 rows** (Sep 2020 to early 2021).
    -   **Action**: Safe to use from roughly **Jan 2021** onwards.

### 4.2 Recommendations for TFT Training
*   **Option A (Full History)**: Use all data from 2020, but fill the missing metrics with a static value (e.g., -1 or 0) and add a binary feature `metrics_available` (0 or 1). This allows learning price dynamics from 2020-2022 but ignores funding/sentiment signals.
*   **Option B (High Quality)**: Filter the dataset to start from **2022-01-01**. This ensures all 101 features are available and valid. This provides ~4 years of high-quality dense data (2022-2026), which is usually sufficient for crypto timeframe training.

**Generated Files**:
-   `merged_tft_data.parquet`: The complete merged dataset.
-   `merge_report.json`: detailed missing value counts and schema info.
-   `merge_strategy_data.py`: The reproducible script used to generate the data.
