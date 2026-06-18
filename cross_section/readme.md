# Cross-Sectional Multi-Factor Strategy

## 模块定位

本目录是 `cross_section` 仓库中的主线横截面多因子策略模块，用于完成加密资产小时级横截面选币、因子研究、机器学习排序预测、多空组合回测和结果可视化。

策略目标不是预测单个币种的绝对涨跌，而是识别同一时间截面中未来相对更强和相对更弱的资产。模型输出用于构建 Top N 多头与 Bottom N 空头组合，并通过交易成本、BTC/ETH overlay、崩盘保护和收益归因评估策略表现。

## 执行入口

主入口文件为 `main.py`。

```bash
cd D:\muti\cross_section\cross_section
python main.py
```

完整运行时，`main.py` 依次执行以下阶段：

```text
DataEngine
  -> FactorEngine
  -> FactorResearcher
  -> FactorSelector
  -> ModelTrainer
  -> StrategyAssembler
  -> StrategyVisualizer
```

命令行参数如下：

```bash
# 运行完整策略流水线，默认复用因子缓存
python main.py

# 禁用因子缓存，重新计算全部因子
python main.py --no-cache

# 仅基于已有 output 结果重新生成图表和摘要
python main.py --visualize
```

## 目录结构

```text
cross_section/
  main.py                         策略主入口
  config.py                       数据路径、清洗、因子、模型、组合和风控参数
  data_engine.py                  数据加载、审计、清洗、动态票池和市场指数
  factor_engine.py                因子计算、缓存、BTC 正交化、截面标准化和因子矩阵
  factor_research.py              因子 IC、Rank IC、ICIR、衰减、单调性和稳定性检验
  factor_selector.py              因子筛选、研究分数排序和相关性去重
  model_trainer.py                标签构建、LightGBM/CatBoost 滚动训练和预测
  backtester.py                   择时、持仓生成、回测、指标计算和结果保存
  visualizer.py                   净值、回撤、月度收益、因子、持仓和收益分布图表
  attribution.py                  收益分解、成本、敞口、换手和持仓归因
  rerun_backtest.py               基于已有预测和入选因子重新运行回测
  analyze_oct2025_drawdown.py     指定回撤区间诊断脚本
  _scratch_analysis/              临时诊断和实验脚本
  output/                         策略运行输出目录
```

## 数据依赖

数据路径由 `config.py` 中的 `DATA_DIR` 和 `DATA_FILES` 定义：

```python
STRATEGY_ROOT = PROJECT_ROOT.parent
DATA_DIR = STRATEGY_ROOT / "data" / "cross_section"
```

模块期望读取以下 parquet 文件：

```text
D:\muti\cross_section\data\cross_section\cross_section_open.parquet
D:\muti\cross_section\data\cross_section\cross_section_high.parquet
D:\muti\cross_section\data\cross_section\cross_section_low.parquet
D:\muti\cross_section\data\cross_section\cross_section_close.parquet
D:\muti\cross_section\data\cross_section\cross_section_volume.parquet
D:\muti\cross_section\data\cross_section\cross_section_quote_volume.parquet
D:\muti\cross_section\data\cross_section\cross_section_returns.parquet
```

数据表结构为时间索引加交易对列，主要用于小时级 OHLCV 横截面研究。`DataEngine` 在运行时重新从清洗后的 close 计算收益率，不直接依赖预存 returns 作为最终研究口径。

## 核心参数

### 数据清洗参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `MAX_RETURN_THRESHOLD` | `0.15` | 单小时收益率硬上限，限制在正负 15% |
| `WINSORIZE_STD` | `3.5` | 横截面缩尾标准差倍数 |
| `MIN_HOURLY_QUOTE_VOLUME` | `50000` | 小时最低成交额门槛 |
| `FFILL_LIMIT` | `6` | 价格字段最大前向填充小时数 |
| `MIN_DATA_COVERAGE` | `0.85` | 单币种最低数据覆盖率 |
| `DEAD_COIN_HOURS` | `48` | 连续低活跃状态识别窗口 |
| `DELIST_BUFFER_HOURS` | `72` | 退市或提前消失资产的缓冲剔除窗口 |

### 动态票池参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `LOOKBACK_HOURS_VOLUME` | `24` | 滚动成交额排名窗口 |
| `MIN_HISTORY_HOURS` | `720` | 入池前最低历史数据长度 |
| `UNIVERSE_TOP_EXCLUDE` | `10` | 剔除成交额排名前 10 的大票 |
| `UNIVERSE_BOTTOM_RANK` | `100` | 保留成交额排名 11-100 的资产 |

### 标签与训练参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `FORWARD_HOURS` | `12` | 标签预测窗口，未来 12 小时 |
| `ROLLING_TRAIN_SIZE` | `4320` | 滚动训练窗口，约 6 个月 |
| `ROLLING_TEST_SIZE` | `720` | 滚动测试窗口，约 1 个月 |
| `ROLLING_EMBARGO` | `24` | 训练与测试之间的时间隔离期 |
| `ENSEMBLE_WEIGHTS` | `0.5 / 0.5` | LightGBM 与 CatBoost 融合权重 |

### 因子筛选参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `FACTOR_SELECTION_MIN_ABS_IC` | `0.005` | 最低 Rank IC 绝对值 |
| `FACTOR_SELECTION_MIN_ICIR` | `0.10` | 最低 ICIR 绝对值 |
| `FACTOR_SELECTION_MAX_CORR` | `0.70` | 入选因子最大相关性阈值 |
| `FACTOR_SELECTION_MIN_FEATURES` | `8` | 最少入选因子数量 |
| `FACTOR_SELECTION_MAX_FEATURES` | `20` | 最多入选因子数量 |

### 组合与风控参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `LONG_N` | `10` | 多头与空头各选 10 个币种 |
| `REBALANCE_HOURS` | `96` | 每 96 小时调仓 |
| `INITIAL_CAPITAL` | `100000` | 回测初始资金 |
| `TAKER_FEE` | `0.0005` | Taker 手续费 |
| `SLIPPAGE` | `0.001` | 滑点参数 |
| `HEDGE_SYMBOLS` | `BTCUSDT, ETHUSDT` | overlay 标的 |
| `HEDGE_RATIO` | `0.2` | overlay 总仓位比例 |
| `CRASH_PROTECT_BTC_WINDOW` | `12` | BTC 崩盘保护观察窗口 |
| `CRASH_PROTECT_BTC_DROP` | `-0.05` | BTC 12 小时跌幅触发阈值 |

## 阶段一：数据引擎

实现文件：`data_engine.py`

核心类：`DataEngine`

`DataEngine` 将数据审计、数据清洗、动态票池和市场指数构建合并为一个闭环。执行方法为：

```python
engine = DataEngine()
panel_data = engine.run_pipeline()
```

### 主要方法

| 方法 | 职责 |
|---|---|
| `load_data()` | 读取 open、high、low、close、volume、quote_volume、returns parquet 文件 |
| `run_audit()` | 输出缺失率、极端收益、死币、低覆盖率币种等质量信息 |
| `clean_data()` | 剔除问题币种、有限前向填充、收益率重算、硬上限和缩尾处理 |
| `build_universe()` | 按成交额排名、流动性、历史长度、退市缓冲和有效价格构建动态票池 |
| `build_market_index()` | 构建成交额加权市场指数 |
| `get_panel_data()` | 汇总后续模块需要的面板数据 |
| `run_pipeline()` | 串联完整数据阶段 |

### 数据审计输出

当前已有 `output/data_audit_report.txt` 中记录：

```text
Missing data: 37.0%
Pumps >100%/h: 0, Dumps <-50%/h: 0
Return range: [-50.0%, 49.3%]
Dead/illiquid coins: 404
Low coverage (<85%): 326 coins
```

该审计结果说明样本中存在大量低覆盖率和低活跃资产，因此动态票池与清洗规则是策略数据口径的重要组成部分。

## 阶段二：因子工程

实现文件：`factor_engine.py`

核心类：`FactorEngine`

`FactorEngine` 根据清洗后的面板数据计算横截面因子，支持单因子缓存、BTC 残差正交化、截面标准化和长表因子矩阵输出。

```python
factor_engine = FactorEngine(panel_data, use_cache=True)
factors_normalized, factor_matrix = factor_engine.run_pipeline()
```

### 因子类别

| 类别 | 示例 |
|---|---|
| Alpha 101 精选因子 | `alpha001`, `alpha006`, `alpha009`, `alpha012`, `alpha040`, `alpha060`, `alpha101` |
| 动量与趋势因子 | `roc_6`, `momentum_72`, `slope_24`, `rsi_24`, `macd_hist` |
| 波动率因子 | `atr_14`, `natr_14`, `stddev_24`, `ulcer_index`, `parkinson_vol` |
| 量价与流动性因子 | `mfi_24`, `vol_ratio_24`, `pvt_norm`, `liquidity_ratio`, `davids_ratio` |
| 统计分布因子 | `skewness_72`, `kurtosis_72`, `zscore_close`, `ts_argmax_24`, `corr_pv_24` |
| K 线结构因子 | `shadow_upper`, `shadow_lower`, `body_size`, `gap`, `log_return` |
| 扩展强弱因子 | `williams_r_72`, `stoch_k_72`, `realized_vol_72`, `max_dd_168`, `up_down_ratio_72` |

### 处理流程

1. `_get_factor_definitions()` 注册全部可计算因子。
2. `generate_all_factors()` 读取缓存或重新计算因子。
3. `orthogonalize_factors()` 对因子做 BTC 收益残差正交化。
4. `cross_sectional_normalize()` 在每个时间截面对票池内因子做 z-score 标准化，并限制到 `[-3, 3]`。
5. `create_factor_matrix()` 将宽表因子转为 `(timestamp, symbol)` MultiIndex 长表矩阵。

主要输出：

```text
output/factors_cache/*.parquet
output/factor_matrix.parquet
```

## 阶段三：因子研究

实现文件：`factor_research.py`

核心类：`FactorResearcher`

因子研究阶段用于评估单因子的横截面预测能力。研究目标不是直接生成交易信号，而是为后续入模因子筛选提供依据。

```python
researcher = FactorResearcher(panel_data, factors_normalized, factor_matrix)
research_report = researcher.run_research()
```

### 检验指标

| 指标 | 含义 |
|---|---|
| `ic_mean` | 因子值与未来 alpha 收益的 Pearson IC 均值 |
| `rank_ic_mean` | 因子排序与未来收益排序的 Rank IC 均值 |
| `rank_icir` | Rank IC 均值除以 Rank IC 标准差 |
| `positive_ratio` | 因子方向与整体方向一致的截面占比 |
| `monotonicity` | 分位数组合收益的单调性 |
| `spread` | 高低分位组合未来 alpha 收益差 |
| `yearly_consistency` | 年度 IC 方向一致性 |
| `rolling_stability` | 滚动窗口 IC 方向稳定性 |
| `rank_autocorr` | 因子排序自相关 |
| `factor_turnover` | 由自相关推导的因子换手特征 |

因子研究默认按照 `REBALANCE_HOURS` 采样截面，降低计算量并保持与调仓周期一致。IC decay 检验周期为 `6h`、`12h`、`24h`、`48h`。

主要输出：

```text
output/factor_research/factor_research_report.csv
output/factor_research/ic_decay_analysis.csv
```

## 阶段四：因子筛选

实现文件：`factor_selector.py`

核心类：`FactorSelector`

因子筛选阶段基于研究报告进行最终入模特征选择。

```python
selector = FactorSelector(research_report, factor_matrix)
_, selected_features = selector.run_pipeline()
```

### 筛选逻辑

1. 保留 `|Rank IC| >= 0.005` 的因子。
2. 保留 `|ICIR| >= 0.10` 的因子。
3. 按 `research_score` 从高到低排序。
4. 对候选因子做相关性去重，阈值为 `0.70`。
5. 控制最终特征数在 `8-20` 个之间。

当前输出中的入选因子：

```text
rsi_24
momentum_72
mfi_24
davids_ratio
alpha012
alpha040
slope_24
ulcer_index
alpha009
ts_argmax_24
corr_pv_12
up_down_ratio_72
slope_12
alpha060
alpha053
atr_14
roc_6
corr_pv_24
alpha101
skewness_72
```

主要输出：

```text
output/selected_features.json
output/selected_features_validated.csv
output/selected_feature_correlation.csv
```

## 阶段五：模型训练

实现文件：`model_trainer.py`

核心类：`ModelTrainer`

模型训练阶段使用未来 `12` 小时相对收益 rank 作为标签，并采用 walk-forward 滚动训练结构。

```python
trainer = ModelTrainer(
    panel_data,
    factors_normalized,
    factor_matrix,
    selected_features=selected_features
)
rolling_predictor, selected_features, test_times = trainer.run_pipeline()
```

### 标签定义

标签计算流程：

```text
个币未来 12 小时收益
  - 市场指数未来 12 小时收益
  -> forward alpha
  -> 每个时间截面做 percentile rank
  -> 仅保留动态票池内样本
```

该标签用于训练横截面相对强弱排序模型，不是绝对收益率预测。

### 滚动训练结构

| 部分 | 口径 |
|---|---|
| 训练窗口 | 过去 `4320` 小时 |
| 测试窗口 | 未来 `720` 小时 |
| Embargo | 训练与测试之间隔离 `24` 小时 |
| 验证集 | 每个训练窗口最后 20% 时间段 |
| 模型 | LightGBM Regressor + CatBoost Regressor |
| 融合 | 两个模型预测按时间截面 z-score 后等权融合 |

### 过拟合诊断

每个 fold 会记录 train IC、test IC 和 overfit ratio，并输出到：

```text
output/fold_overfit_diagnostics.csv
```

训练阶段还会输出损失曲线、滚动预测和特征重要性：

```text
output/rolling_predictions.parquet
output/selected_features_validated.csv
output/feature_importance.csv
output/loss_curves.png
```

## 阶段六：组合构建与回测

实现文件：`backtester.py`

核心类：

| 类 | 职责 |
|---|---|
| `TimingModule` | 根据 BTC 日线 RSI 生成 overlay regime |
| `DirectionalPositionManager` | 将预测排序转化为多空持仓和 BTC/ETH overlay |
| `BacktestEngine` | 计算组合收益、交易成本、净值曲线和绩效指标 |
| `StrategyAssembler` | 串联预测、择时、持仓、回测和结果保存 |

### 预测矩阵

`StrategyAssembler.generate_predictions()` 仅在滚动训练产生的 `test_times` 上生成回测预测，避免训练样本参与回测。

### 核心持仓规则

```text
每个调仓截面：
  1. 读取模型预测分数
  2. 剔除 BTCUSDT 与 ETHUSDT overlay 标的
  3. 选择预测最高的 Top 10 做多
  4. 选择预测最低的 Bottom 10 做空
  5. 多头总权重为 +1.0
  6. 空头总权重为 -1.0
  7. 非调仓时段沿用上一期持仓
```

持仓 buffer 机制会保留仍处于较优分位的原持仓，以降低组合换手。

### BTC/ETH overlay

`TimingModule` 使用 BTC 日线 RSI 判断市场状态：

| 条件 | 行为 |
|---|---|
| `RSI > 60` | 小仓位做空 BTC/ETH overlay |
| `RSI < 40` | 小仓位做多 BTC/ETH overlay |
| `40 <= RSI <= 60` | 不做 overlay |

overlay 总仓位由 `HEDGE_RATIO=0.2` 控制，BTC 与 ETH 权重比例为 `0.6 / 0.4`。

### 崩盘保护

当 BTC 在 `12` 小时窗口内跌幅小于等于 `-5%` 时，策略触发 crash protection：

```text
山寨币空头 gross 缩减到 50%
强制写入 BTC/ETH 多头 overlay
```

该规则用于控制极端下跌环境中空头挤压和市场相关性跃升风险。

### 交易成本

回测使用原始收益率进行结算。交易成本按以下方式扣除：

```text
trade_cost_rate = TAKER_FEE + SLIPPAGE
```

成本仅在真实交易事件上计入：

- 从 0 到非 0 的新开仓；
- 从非 0 到 0 的平仓；
- 多空方向切换。

### 绩效指标

`BacktestEngine._calc_metrics()` 输出：

```text
total_return
ann_return
ann_volatility
sharpe_ratio
max_drawdown
calmar_ratio
win_rate
profit_loss_ratio
avg_turnover
index_return
excess_return
info_ratio
long_return
short_return
portfolio_beta
n_hours
n_years
```

主要输出：

```text
output/equity_curve.csv
output/positions.parquet
output/long_positions.parquet
output/short_positions.parquet
output/trades.csv
output/metrics.csv
output/predictions.parquet
```

## 阶段七：可视化

实现文件：`visualizer.py`

核心类：`StrategyVisualizer`

```python
viz = StrategyVisualizer(result)
viz.market_index = panel_data["market_index"]
viz.feature_importance = trainer.feature_importance
viz.generate_report()
viz.print_summary()
```

可视化输出包括：

| 文件 | 内容 |
|---|---|
| `equity_curve.png` | 策略净值、BTC 对比、回撤和 30 日滚动 Sharpe |
| `monthly_returns.png` | 月度收益热力图 |
| `feature_importance.png` | Top 因子重要性 |
| `position_analysis.png` | 持仓数量、多空权重、常持仓币种和换手率 |
| `ic_decay.png` | 因子 IC decay 图 |
| `attribution_analysis.png` | 收益归因图 |
| `return_distribution.png` | 收益分布与尾部风险图 |

仅重绘图表时执行：

```bash
python main.py --visualize
```

## 收益归因

实现文件：`attribution.py`

该脚本读取已有 output 结果和原始 close 数据，对策略收益进行分解。

```bash
python attribution.py
```

分析内容包括：

- 山寨币多头累计收益；
- 山寨币空头累计收益；
- BTC overlay 收益；
- ETH overlay 收益；
- 总交易成本；
- 月度与季度收益；
- 滚动 30 日收益、波动和 Sharpe；
- 平均换手和持仓重合度；
- 组合 beta、净敞口和胜率；
- 最常持有的多头和空头币种。

## 诊断脚本

### `rerun_backtest.py`

用于在已有特征、预测或入选因子基础上重新运行回测。该脚本适用于组合规则、overlay 或回测成本口径变更后的快速复算。

```bash
python rerun_backtest.py
```

### `analyze_oct2025_drawdown.py`

用于诊断特定回撤区间。脚本读取持仓、收益和市场数据，定位某一时间段内的收益来源、风险暴露和异常资产。

```bash
python analyze_oct2025_drawdown.py
```

### `_scratch_analysis/`

该目录包含临时实验和诊断脚本，例如 RSI overlay 诊断、崩盘保护比较、对称多空测试等。该目录不属于主线生产入口。

## 输出文件总览

`output/` 目录按阶段保存中间结果和最终结果。

```text
data_audit_report.txt                 数据审计摘要
market_index.parquet                  成交额加权市场指数
universe_mask.parquet                 动态票池 mask
factor_matrix.parquet                 长表因子矩阵
factors_cache/                        单因子缓存目录
factor_research/factor_research_report.csv
factor_research/ic_decay_analysis.csv
selected_features.json                最终入模因子
selected_features_validated.csv       最终入模因子 CSV
selected_feature_correlation.csv      入选因子相关性矩阵
rolling_predictions.parquet           滚动训练样本外预测
fold_overfit_diagnostics.csv          fold 级过拟合诊断
feature_importance.csv                特征重要性
predictions.parquet                   回测预测宽表
positions.parquet                     合并持仓
long_positions.parquet                多头持仓
short_positions.parquet               空头持仓
trades.csv                            交易记录
metrics.csv                           回测指标
equity_curve.csv                      净值曲线
*.png                                 可视化图表
```

## 依赖环境

代码使用 Python 生态中的数据分析、机器学习和绘图库。核心依赖包括：

```text
pandas
numpy
matplotlib
scipy
lightgbm
catboost
pyarrow 或 fastparquet
```

`attribution.py`、`visualizer.py` 和部分诊断脚本还依赖 matplotlib 图表环境。parquet 文件读取需要安装 `pyarrow` 或 `fastparquet`。

## 研究边界

本模块是研究与回测系统，不是实盘交易执行系统。适用边界包括：

- 输入数据为小时级 K 线面板，未包含盘口深度、逐笔成交和真实撮合队列。
- 回测成本采用手续费与滑点参数化方式，未建模订单簿冲击成本。
- 标签是未来相对收益 rank，不是未来价格或绝对收益金额。
- 因子筛选和模型训练依赖历史样本，市场结构切换会影响稳定性。
- BTC/ETH overlay 是风险调节层，不是独立择时策略。
- 崩盘保护基于 BTC 短周期跌幅规则，无法覆盖全部极端行情形态。

## 维护口径

模块维护时应保持以下口径一致：

1. 数据清洗逻辑集中在 `data_engine.py`，不要在后续阶段重复局部清洗。
2. 因子定义集中在 `factor_engine.py::_get_factor_definitions()`。
3. 因子选择阈值集中在 `config.py` 和 `factor_selector.py`。
4. 标签构建集中在 `model_trainer.py::build_labels()`。
5. 组合生成和交易成本集中在 `backtester.py`。
6. 新增输出文件应写入 `OUTPUT_DIR`，保持结果可复现。
7. 修改预测窗口时，`FORWARD_HOURS` 与 `ROLLING_EMBARGO` 需要保持时间隔离关系。
8. 修改调仓频率时，因子研究采样、持仓生成和换手成本分析需要同步检查。
