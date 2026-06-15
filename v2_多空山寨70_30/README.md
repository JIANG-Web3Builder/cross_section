# 截面多因子量化策略 - V2 多空山寨对冲版

## 目录
- [策略概述](#策略概述)
- [项目结构](#项目结构)
- [运行方式](#运行方式)
- [回测流程详解](#回测流程详解)
  - [Phase 1: 数据工程](#phase-1-数据工程-data_engineeringpy)
  - [Phase 2: 因子工程](#phase-2-因子工程-factor_engineeringpy)
  - [Phase 3: 模型训练](#phase-3-模型训练-model_trainingpy)
  - [Phase 4: 策略回测](#phase-4-策略回测-strategy_backtestpy)
  - [Phase 5: 可视化](#phase-5-可视化-visualizationpy)
- [配置参数详解](#配置参数详解)
- [输出文件说明](#输出文件说明)

---

## 策略概述

**策略类型**: 截面多因子量化策略 (Cross-Sectional Multi-Factor Strategy)

**核心思路**:
- **市场中性**: 做多预测最强的Top N个山寨币 + 做空预测最弱的Bottom N个山寨币
- **纯Alpha策略**: 完全对冲市场Beta风险，只赚取选币能力的Alpha收益
- **滚动训练**: Walk-Forward Analysis，每月更新模型，避免未来函数

**策略特点**:
- Mid-Cap Universe: 选取成交额排名11-100的腰部资产（剔除Top 10大票和流动性差的小票）
- 双模融合: LightGBM + CatBoost 集成学习
- 波动率倒数加权: 给高波动币种更低权重
- 换仓缓冲: 降低换手率，每12小时换仓一次
- BTC剔除: BTC不参与排序，不作为对冲端

---

## 项目结构

```
v2_多空山寨70_30/
├── config.py              # 配置文件（所有参数）
├── main.py                # 主入口，运行完整流水线
├── data_engineering.py    # Phase 1: 数据工程
├── factor_engineering.py  # Phase 2: 因子工程
├── model_training.py      # Phase 3: 模型训练
├── strategy_backtest.py   # Phase 4: 策略回测
├── visualization.py       # Phase 5: 可视化
├── output/                # 输出目录
│   ├── equity_curve.csv   # 净值曲线
│   ├── positions.parquet  # 持仓记录
│   ├── trades.csv         # 交易记录
│   ├── metrics.csv        # 绩效指标
│   └── *.png              # 可视化图表
└── README.md              # 本文档
```

---

## 运行方式

```bash
# 运行完整流水线（使用缓存）
python main.py

# 不使用因子缓存，重新计算所有因子
python main.py --no-cache

# 仅生成可视化（需要先运行过完整流水线）
python main.py --visualize
```

---

## 回测流程详解

### Phase 1: 数据工程 (`data_engineering.py`)

**目的**: 加载原始数据，构建动态票池，创建大盘指数，数据清洗

#### 步骤1.1: 加载数据 (`load_data`)
- **数据来源**: 从 `crypto_截面/data/` 目录加载清洗后的parquet文件
- **数据类型**: open, high, low, close, volume, quote_volume, returns
- **处理**: 排除问题币种（USDCUSDT, USTCUSDT, LUNAUSDT, FTTUSDT）

#### 步骤1.2: 构建动态票池 (`build_dynamic_universe`)
- **逻辑**:
  1. 计算24小时滚动成交额
  2. 对每个时间截面进行排名
  3. 选取排名11-100的币种（Mid-Cap Universe）
  4. 剔除历史数据不足720小时的币种
  5. **强制保留BTCUSDT**用于对冲（虽然本策略不用BTC对冲）
- **使用的参数**:
  - `LOOKBACK_HOURS_VOLUME`: 计算成交额的回看窗口
  - `MIN_HISTORY_HOURS`: 最少需要的历史数据小时数
  - `EXCLUDE_SYMBOLS`: 排除的币种列表

#### 步骤1.3: 构建合成大盘指数 (`build_market_index`)
- **逻辑**: 使用票池内资产收益率的**中位数**（更稳健，抗极端值）
- **输出**: 大盘指数的收益率和累计价格

#### 步骤1.4: 数据去极值 (`winsorize_data`)
- **逻辑**:
  1. 对收益率进行截面缩尾处理（超过N倍标准差的值拉回）
  2. 限制单小时收益不超过50%
  3. **保留原始收益率**用于回测结算，清洗后的收益率仅用于因子计算
- **使用的参数**:
  - `WINSORIZE_STD`: 缩尾处理的标准差倍数
  - `MAX_RETURN_THRESHOLD`: 单小时最大收益率阈值

#### 输出
- `panel_data`: 包含所有数据的字典
- `market_index.parquet`: 大盘指数
- `universe_mask.parquet`: 动态票池掩码

---

### Phase 2: 因子工程 (`factor_engineering.py`)

**目的**: 计算技术因子，截面标准化，创建因子矩阵

#### 步骤2.1: 生成所有因子 (`generate_all_factors`)
- **因子类别**:
  1. **Alpha 101 精选因子** (15个): alpha001-alpha101
  2. **动量类因子** (15个): ROC, Bias, Slope, RSI, MACD, TRIX, CCI, APO, CMO等
  3. **波动率类因子** (8个): ATR, NATR, StdDev, Ulcer Index, BB_Width, Parkinson Vol等
  4. **量价与流动性因子** (10个): VWAP_Dist, OBV, MFI, Vol_Ratio, PVT, Liquidity Ratio等
  5. **统计分布特征** (9个): Skewness, Kurtosis, Z-Score, Ts_ArgMax, Ts_ArgMin, Corr_PV等
  6. **K线形态特征** (5个): Shadow_Upper, Shadow_Lower, Body_Size, Gap, Log_Return
  7. **扩展因子** (16个): Momentum, Williams %R, Stochastic K, Realized Vol, HL Ratio等
- **缓存机制**: 因子计算后保存到 `crypto_截面/output/factors_cache/`，避免重复计算
- **Numba加速**: 复杂滚动计算使用Numba JIT编译加速

#### 步骤2.2: 截面标准化 (`cross_sectional_normalize`)
- **逻辑**:
  1. 对每个时间切片，计算票池内因子值的均值和标准差
  2. Z-Score标准化: `(x - mean) / std`
  3. 缩尾处理: 限制在±3倍标准差内
- **目的**: 消除因子量纲差异，使不同因子可比

#### 步骤2.3: 创建因子矩阵 (`create_factor_matrix`)
- **格式**: 长表格式，MultiIndex为 (timestamp, symbol)
- **过滤**: 只保留票池内的记录

#### 输出
- `factors_normalized`: 标准化后的因子字典
- `factor_matrix.parquet`: 因子矩阵

---

### Phase 3: 模型训练 (`model_training.py`)

**目的**: 构建标签，滚动训练模型，特征重要性分析

#### 步骤3.1: 标签构建 (`build_labels`)
- **逻辑**:
  1. 计算未来36小时收益率 (`FORWARD_HOURS`)
  2. 减去大盘指数收益（去Beta）
  3. 转换为截面Rank (0-1之间)
- **使用的参数**:
  - `FORWARD_HOURS`: 预测的未来收益周期

#### 步骤3.2: 滚动窗口训练 (`run_rolling_training`)
- **Walk-Forward Analysis逻辑**:
  1. 训练窗口: 过去6个月 (`ROLLING_TRAIN_SIZE`)
  2. 测试窗口: 未来1个月 (`ROLLING_TEST_SIZE`)
  3. 隔离期: 36小时 (`ROLLING_EMBARGO`)，防止Label泄露
  4. 每月滚动一次，模拟真实"定期更新模型"
- **模型训练**:
  1. 训练集最后20%作为验证集
  2. 训练 LightGBM 模型（Early Stopping）
  3. 训练 CatBoost 模型（Early Stopping）
  4. Z-Score融合两个模型的预测
- **使用的参数**:
  - `ROLLING_TRAIN_SIZE`: 训练窗口大小
  - `ROLLING_TEST_SIZE`: 测试窗口大小
  - `ROLLING_EMBARGO`: 隔离期大小
  - `LGBM_PARAMS`: LightGBM模型参数
  - `CATBOOST_PARAMS`: CatBoost模型参数
  - `ENSEMBLE_WEIGHTS`: 融合权重

#### 步骤3.3: 特征重要性分析 (`analyze_feature_importance`)
- **逻辑**: 基于最后一个Fold的模型，归一化LGBM和CatBoost的特征重要性，取平均

#### 输出
- `rolling_predictions.parquet`: 滚动预测结果
- `selected_features.csv`: 使用的特征列表
- `feature_importance.csv`: 特征重要性

---

### Phase 4: 策略回测 (`strategy_backtest.py`)

**目的**: 生成持仓信号，执行回测，计算绩效指标

#### 步骤4.1: 择时模块 (`TimingModule`)
- **逻辑**: 计算大盘波动率，在极端波动（>95%分位）时暂停交易
- **使用的参数**:
  - `TIMING_MA_PERIOD`: 大盘MA周期
  - `TIMING_VOL_PERIOD`: 波动率计算周期
  - `TIMING_VOL_QUANTILE`: 波动率危险阈值分位数

#### 步骤4.2: 持仓管理 (`MarketNeutralPositionManager`)
- **纯截面多空策略逻辑**:
  1. **剔除BTC**: BTC不参与排序
  2. **做多**: 预测分最高的Top N个币（等权分配，总权重+1.0）
  3. **做空**: 预测分最低的Bottom N个币（等权分配，总权重-1.0）
  4. **换仓频率控制**: 每12小时换仓一次
- **辅助功能**:
  - 波动率倒数加权 (`calc_inverse_vol_weights`)
  - 滚动Beta计算 (`calc_rolling_beta`)
  - 换仓缓冲机制 (`apply_buffer_mechanism`)
- **使用的参数**:
  - `LONG_N`: 做多/做空数量
  - `TOTAL_HOLD`: 总持仓数量
  - `REBALANCE_HOURS`: 换仓频率
  - `BETA_LOOKBACK`: Beta计算回看周期

#### 步骤4.3: 回测引擎 (`BacktestEngine`)
- **止损逻辑**:
  1. 跟踪每个持仓的累计收益
  2. 当单币种亏损超过10%时立即平仓
  3. 平仓后在下次换仓前不再开仓该币种
- **收益计算**:
  1. 多头收益: 正权重 × 正收益
  2. 空头收益: 负权重 × 正收益（做空赚反向收益）
  3. 扣除交易成本
- **使用的参数**:
  - `MAKER_FEE`: Maker费率
  - `TAKER_FEE`: Taker费率
  - `SLIPPAGE`: 滑点
  - `STOP_LOSS_PCT`: 止损阈值

#### 步骤4.4: 绩效指标计算 (`_calc_metrics`)
- 总收益率、年化收益率
- 年化波动率、最大回撤
- 夏普比率、卡尔玛比率、信息比率
- 胜率、盈亏比、平均换手率
- 多头收益、空头收益、组合Beta

#### 输出
- `equity_curve.csv`: 净值曲线
- `positions.parquet`: 合并后的净持仓
- `long_positions.parquet`: 多头持仓
- `short_positions.parquet`: 空头持仓
- `trades.csv`: 交易记录
- `metrics.csv`: 绩效指标
- `predictions.parquet`: 模型预测

---

### Phase 5: 可视化 (`visualization.py`)

**目的**: 生成策略分析图表

#### 图表类型
1. **净值曲线** (`plot_equity_curve`): 策略净值 vs 大盘指数，回撤曲线，滚动夏普
2. **月度收益热力图** (`plot_monthly_returns`): 按年月展示收益率
3. **特征重要性** (`plot_feature_importance`): Top 25重要因子
4. **持仓分析** (`plot_position_analysis`): 持仓数量、权重、最常持仓币种、换手率分布
5. **收益分布** (`plot_return_distribution`): 收益直方图、Q-Q图

---

## 配置参数详解

### 路径配置

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `PROJECT_ROOT` | 当前目录 | 项目根目录 | 所有模块 |
| `DATA_DIR` | `crypto_截面/data` | 数据目录 | `data_engineering.py` |
| `OUTPUT_DIR` | `output/` | 输出目录 | 所有模块 |

### 数据文件配置

| 参数 | 说明 | 使用位置 |
|------|------|----------|
| `DATA_FILES` | 清洗后的数据文件路径字典 | `data_engineering.py: load_data()` |
| `DATA_FILES_RAW` | 原始数据文件路径字典 | 备用 |

### 动态票池参数

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `LOOKBACK_HOURS_VOLUME` | 24 | 计算成交额的回看窗口(小时) | `data_engineering.py: build_dynamic_universe()` |
| `MIN_HISTORY_HOURS` | 720 | 最少需要720小时历史数据 | `data_engineering.py: build_dynamic_universe()` |

### 数据清洗参数

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `WINSORIZE_STD` | 10.0 | 缩尾处理：超过N倍标准差的值拉回 | `data_engineering.py: winsorize_data()` |
| `MAX_RETURN_THRESHOLD` | 0.5 | 单小时最大收益率阈值(50%) | `data_engineering.py: winsorize_data()` |
| `EXCLUDE_SYMBOLS` | ['USDCUSDT', 'USTCUSDT', 'LUNAUSDT', 'FTTUSDT'] | 排除的币种 | `data_engineering.py: load_data()` |

### 标签构建参数

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `FORWARD_HOURS` | 36 | 预测未来36小时收益 | `model_training.py: build_labels()` |

### 滚动训练参数 (Walk-Forward Analysis)

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `ROLLING_TRAIN_SIZE` | 4320 | 训练窗口: 6个月 (约180天×24小时) | `model_training.py: run_rolling_training()` |
| `ROLLING_TEST_SIZE` | 720 | 测试窗口: 1个月 (约30天×24小时) | `model_training.py: run_rolling_training()` |
| `ROLLING_EMBARGO` | 36 | 隔离期: 36小时 (防止Label泄露) | `model_training.py: run_rolling_training()` |

### LightGBM 参数

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `objective` | 'regression' | 回归任务 | `model_training.py` |
| `metric` | 'rmse' | 评估指标 | `model_training.py` |
| `n_estimators` | 5000 | 最大迭代次数（依赖early_stopping） | `model_training.py` |
| `learning_rate` | 0.005 | 学习率（压制过拟合） | `model_training.py` |
| `num_leaves` | 31 | 叶子数（从63降到31，粗粒度判断） | `model_training.py` |
| `max_depth` | 5 | 最大深度（从7降到5，浅树泛化更强） | `model_training.py` |
| `min_child_samples` | 500 | 叶子最小样本数（防止过拟合个股） | `model_training.py` |
| `reg_alpha` | 10 | L1正则（过滤不重要特征） | `model_training.py` |
| `reg_lambda` | 10 | L2正则（防止权重过大） | `model_training.py` |
| `min_split_gain` | 0.1 | 分裂最小收益 | `model_training.py` |
| `subsample` | 0.7 | 样本采样率 | `model_training.py` |
| `colsample_bytree` | 0.6 | 特征采样率（防止泄露特征主导） | `model_training.py` |

### CatBoost 参数

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `iterations` | 5000 | 最大迭代次数 | `model_training.py` |
| `learning_rate` | 0.005 | 学习率 | `model_training.py` |
| `depth` | 5 | 树深度 | `model_training.py` |
| `l2_leaf_reg` | 30 | L2正则（抗噪核心，从5提到30） | `model_training.py` |
| `rsm` | 0.6 | 特征采样率 | `model_training.py` |
| `subsample` | 0.7 | 样本采样率 | `model_training.py` |
| `od_wait` | 200 | Early Stopping耐心值 | `model_training.py` |

### 融合权重

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `ENSEMBLE_WEIGHTS['lgbm']` | 0.5 | LightGBM权重 | `model_training.py: run_rolling_training()` |
| `ENSEMBLE_WEIGHTS['catboost']` | 0.5 | CatBoost权重 | `model_training.py: run_rolling_training()` |

### 择时参数

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `TIMING_MA_PERIOD` | 168 | 大盘MA周期(1周) | `strategy_backtest.py: TimingModule` |
| `TIMING_VOL_PERIOD` | 24 | 波动率计算周期 | `strategy_backtest.py: TimingModule` |
| `TIMING_VOL_QUANTILE` | 0.90 | 波动率危险阈值分位数 | `strategy_backtest.py: TimingModule` (未直接使用，用0.95) |

### 持仓参数

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `LONG_N` | 10 | 做多/做空数量 (Top 10 / Bottom 10) | `strategy_backtest.py: MarketNeutralPositionManager` |
| `TOTAL_HOLD` | 10 | 总持仓数量 | `strategy_backtest.py` (备用) |
| `REBALANCE_HOURS` | 12 | 换仓频率(12小时) | `strategy_backtest.py: generate_positions()` |

### Beta中性参数

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `BETA_LOOKBACK` | 168 | Beta计算回看周期(1周) | `strategy_backtest.py: calc_rolling_beta()` |

### 交易成本

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `MAKER_FEE` | 0.0002 | 0.02% Maker费率 | `strategy_backtest.py: run_backtest()` |
| `TAKER_FEE` | 0.0004 | 0.04% Taker费率 | `strategy_backtest.py: run_backtest()` |
| `SLIPPAGE` | 0.001 | 0.1% 滑点 | `strategy_backtest.py: run_backtest()` |

### 风控参数

| 参数 | 值 | 说明 | 使用位置 |
|------|-----|------|----------|
| `STOP_LOSS_PCT` | 0.10 | 单币种止损阈值 (10%亏损触发止损) | `strategy_backtest.py: _apply_stop_loss()` |
| `MAX_POSITION_DRAWDOWN` | 0.15 | 单币种最大回撤阈值 (15%) | 备用，未实现 |

---

## 输出文件说明

| 文件 | 格式 | 说明 |
|------|------|------|
| `equity_curve.csv` | CSV | 净值曲线，index为时间戳 |
| `equity_curve.png` | PNG | 净值曲线图（含回撤、滚动夏普） |
| `positions.parquet` | Parquet | 合并后的净持仓权重 |
| `long_positions.parquet` | Parquet | 多头持仓权重 |
| `short_positions.parquet` | Parquet | 空头持仓权重（负数） |
| `trades.csv` | CSV | 交易记录 |
| `metrics.csv` | CSV | 绩效指标汇总 |
| `predictions.parquet` | Parquet | 模型预测值 |
| `rolling_predictions.parquet` | Parquet | 滚动训练的预测结果 |
| `market_index.parquet` | Parquet | 合成大盘指数 |
| `universe_mask.parquet` | Parquet | 动态票池掩码 |
| `factor_matrix.parquet` | Parquet | 因子矩阵 |
| `feature_importance.csv` | CSV | 特征重要性排名 |
| `selected_features.csv` | CSV | 使用的特征列表 |
| `monthly_returns.png` | PNG | 月度收益热力图 |
| `feature_importance.png` | PNG | 特征重要性图 |
| `position_analysis.png` | PNG | 持仓分析图 |
| `return_distribution.png` | PNG | 收益分布图 |

---

## 修改建议

### 如何调整持仓数量
修改 `config.py` 中的 `LONG_N` 参数，例如改为 5 则做多/做空各5个币。

### 如何调整换仓频率
修改 `config.py` 中的 `REBALANCE_HOURS` 参数，例如改为 24 则每24小时换仓一次。

### 如何调整预测周期
修改 `config.py` 中的 `FORWARD_HOURS` 参数，同时建议同步调整 `ROLLING_EMBARGO` 避免数据泄露。

### 如何调整模型复杂度
- 增加过拟合风险: 调大 `num_leaves`, `max_depth`，调小 `reg_alpha`, `reg_lambda`
- 降低过拟合风险: 调小 `num_leaves`, `max_depth`，调大 `reg_alpha`, `reg_lambda`, `min_child_samples`

### 如何添加新因子
1. 在 `factor_engineering.py` 中添加计算函数
2. 在 `_get_factor_definitions()` 方法中注册新因子
3. 运行 `python main.py --no-cache` 重新计算因子

### 如何调整交易成本
修改 `config.py` 中的 `MAKER_FEE`, `TAKER_FEE`, `SLIPPAGE` 参数。

---

## 注意事项

1. **数据依赖**: 本项目依赖 `crypto_截面/data/` 目录下的数据文件
2. **因子缓存**: 因子缓存存储在 `crypto_截面/output/factors_cache/`，修改因子计算逻辑后需要清除缓存
3. **未来函数**: 回测仅在测试集（滚动训练的out-of-sample部分）上进行，避免未来函数
4. **内存占用**: 因子矩阵较大，建议16GB以上内存
