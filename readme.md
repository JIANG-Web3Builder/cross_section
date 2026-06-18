# Cross Section

## 项目定位

本项目是一个面向加密资产横截面研究的量化策略工程，核心目标是基于 Binance 多币种小时级行情数据，构建动态交易标的池、计算多类横截面因子、完成因子有效性检验、训练滚动机器学习模型，并在样本外区间生成多空组合回测结果。

项目主体包含四类研究模块：

1. `cross_section/`：当前主线版本，采用横截面多因子模型、LightGBM 与 CatBoost 集成学习、动态票池、BTC/ETH overlay、交易成本扣除和回测归因。
2. `v2_多空山寨70_30/`：较完整的 V2 多空山寨币策略版本，保留数据审计、因子验证、模型训练、回测和可视化流程。
3. `binance_allcoin_data/`：Binance 全币种历史行情下载、清洗与格式转换模块，用于形成策略所需的 parquet 面板数据。
4. `套利BTC_ETH/`：BTC/ETH 配对统计套利研究模块，覆盖协整检验、动态 hedge ratio、spread、z-score 信号、仓位管理和绩效分析。

## 研究框架

项目采用“数据工程 - 因子工程 - 因子研究 - 模型训练 - 策略回测 - 可视化归因”的流水线结构。

```text
原始行情数据
  -> 数据审计与清洗
  -> 动态交易标的池
  -> 市场指数构建
  -> 横截面因子计算
  -> 因子标准化与正交化
  -> Rank IC / ICIR / 单调性 / 稳定性检验
  -> 因子筛选与相关性去重
  -> 滚动窗口模型训练
  -> 样本外预测
  -> Top N / Bottom N 多空组合
  -> BTC/ETH overlay 与崩盘保护
  -> 扣除交易成本后的回测与归因
```

## 目录结构

```text
cross_section/
  binance_allcoin_data/
    binance_data_fetcher.py      Binance 合约数据抓取工具
    download_all_data.py         批量下载全币种历史数据
    clean_all_data.py            数据清洗、重采样与 parquet 转换
    数据清洗与转化.md             数据清洗口径说明

  cross_section/
    config.py                    主线策略参数、路径、模型和风控配置
    main.py                      主线策略入口
    data_engine.py               数据加载、审计、清洗、票池与市场指数
    factor_engine.py             横截面因子计算、缓存、正交化与标准化
    factor_research.py           因子 IC、ICIR、衰减、单调性与稳定性检验
    factor_selector.py           因子筛选、ICIR 排序与相关性去重
    model_trainer.py             LightGBM/CatBoost 滚动训练与预测
    backtester.py                持仓生成、overlay、交易成本、绩效指标
    attribution.py               策略收益、成本、敞口、持仓归因
    visualizer.py                净值、回撤、因子、持仓、收益分布图表
    rerun_backtest.py            基于既有结果重新运行回测
    analyze_oct2025_drawdown.py  特定回撤区间诊断脚本
    output/                      回测输出、图表与中间结果

  v2_多空山寨70_30/
    main.py                      V2 策略入口
    data_engineering.py          V2 数据工程
    data_auditor.py              V2 数据质量审计
    factor_engineering.py        V2 因子工程
    factor_research.py           V2 因子研究
    factor_validation.py         V2 因子验证
    model_training.py            V2 模型训练
    strategy_backtest.py         V2 回测引擎
    visualization.py             V2 可视化
    README.md                    V2 说明文档

  套利BTC_ETH/
    config.py                    配对套利参数
    data_loader.py               BTC/ETH 数据加载
    cointegration.py             协整检验
    hedge_ratio.py               hedge ratio 与 spread 分析
    signal_generator.py          z-score 信号与仓位生成
    backtester.py                配对套利回测
    performance_analytics.py     绩效分析
    main.py                      配对套利入口
```

## 数据口径

主线策略读取 `data/cross_section/` 下的 parquet 文件，文件名由 `cross_section/config.py` 中的 `DATA_FILES` 统一管理：

```text
cross_section_open.parquet
cross_section_high.parquet
cross_section_low.parquet
cross_section_close.parquet
cross_section_volume.parquet
cross_section_quote_volume.parquet
cross_section_returns.parquet
```

数据以时间为索引、交易对为列，策略主要围绕小时级 OHLCV 面板计算。`DataEngine` 不直接信任预处理后的收益率，而是在清洗后的 close 上重新计算收益率，并区分因子计算使用的清洗收益率与回测结算使用的原始收益率。

核心数据处理口径包括：

- 排除稳定币与已知问题交易对，例如 `USDCUSDT`、`USTCUSDT`、`LUNAUSDT`、`FTTUSDT`、`DAIUSDT`、`TUSDUSDT`、`BUSDUSDT`。
- 检查缺失率、极端收益、死币、低覆盖率币种。
- 对价格字段执行有限前向填充，对成交量缺失填 0。
- 单小时收益率设置硬上限 `±15%`。
- 横截面收益率进行 `±3.5σ` 缩尾。
- 对退市或提前消失的币种设置 `72` 小时退市缓冲，降低存活偏差。
- 按 24 小时滚动成交额排名构建动态票池，剔除成交额 Top 10，保留 Rank 11-100 的中等市值流动性资产。

## 动态票池

主线策略的交易标的池由 `DataEngine.build_universe()` 生成。票池同时满足以下条件：

- 24 小时滚动成交额排名处于 `11-100`。
- 小时成交额不低于 `50,000 USDT`。
- 历史数据长度不低于 `720` 小时。
- 数据覆盖率不低于 `85%`。
- 不处于退市缓冲区。
- 当前时间点有有效价格数据。

BTC 在票池中被保留用于市场指数、风险刻画和 overlay，不作为普通山寨币参与横截面选币排序。

## 市场指数

主线版本使用成交额加权市场指数替代简单中位数指数。该设计用于降低大量低流动性或异常资产对基准收益的污染。市场指数收益用于：

- 构建去市场收益后的 forward alpha label。
- 计算策略相对市场的 excess return 和 information ratio。
- 作为可视化和归因分析中的基准序列。

## 因子体系

`FactorEngine` 负责生成横截面因子，并支持 parquet 缓存。因子类型包括：

1. WorldQuant Alpha 101 精选因子：如 `alpha001`、`alpha006`、`alpha009`、`alpha012`、`alpha040`、`alpha060`、`alpha101`。
2. 动量与趋势因子：如 `roc_6`、`momentum_72`、`slope_24`、`rsi_24`、`macd_hist`。
3. 波动率因子：如 `atr_14`、`natr_14`、`stddev_24`、`ulcer_index`、`parkinson_vol`。
4. 量价与流动性因子：如 `mfi_24`、`vol_ratio_24`、`pvt_norm`、`liquidity_ratio`、`davids_ratio`。
5. 统计分布因子：如 `skewness_72`、`kurtosis_72`、`zscore_close`、`ts_argmax_24`、`corr_pv_24`。
6. K 线结构因子：如 `shadow_upper`、`shadow_lower`、`body_size`、`gap`、`log_return`。
7. 扩展强弱因子：如 `williams_r_72`、`stoch_k_72`、`realized_vol_72`、`max_dd_168`、`up_down_ratio_72`。

因子处理流程包括：

- 对每个因子按时间截面进行 z-score 标准化。
- 标准化后限制在 `[-3, 3]`。
- 使用动态票池 mask，避免使用不在可交易宇宙内的资产。
- 支持对 BTC 收益进行残差正交化，降低模型学习市场方向暴露的概率。
- 生成长表结构 `factor_matrix.parquet`，索引为 `(timestamp, symbol)`。

## 因子研究与筛选

`FactorResearcher` 执行因子层面的有效性检验。核心指标包括：

- Pearson IC：因子值与未来 alpha 收益的线性相关。
- Rank IC：因子排序与未来收益排序的相关性。
- ICIR：Rank IC 均值与波动的比值。
- IC Decay：在 `6h`、`12h`、`24h`、`48h` 不同预测周期下检验信号衰减。
- 分组单调性：检验因子分位数组合收益是否呈现稳定排序。
- 滚动稳定性：检验 IC 方向在年份、滚动窗口和样本前后段是否一致。
- 因子自相关与换手率：用于判断因子持仓周期和换仓频率的匹配程度。

`FactorSelector` 对研究结果执行筛选，主要规则包括：

- `|Rank IC| >= 0.005`。
- `|ICIR| >= 0.10`。
- 按 `research_score` 排序。
- 对候选因子做相关性去重，相关性阈值为 `0.70`。
- 最多保留 `20` 个因子，最少保留 `8` 个因子。

当前主线输出中的 `selected_features.json` 包含以下特征：

```text
rsi_24, momentum_72, mfi_24, davids_ratio, alpha012,
alpha040, slope_24, ulcer_index, alpha009, ts_argmax_24,
corr_pv_12, up_down_ratio_72, slope_12, alpha060, alpha053,
atr_14, roc_6, corr_pv_24, alpha101, skewness_72
```

## 标签构建

`ModelTrainer.build_labels()` 使用未来 `12` 小时收益构建监督学习标签。计算逻辑如下：

1. 计算每个币种未来 `FORWARD_HOURS=12` 小时累计收益。
2. 计算市场指数未来同周期收益。
3. 用个币未来收益减去市场指数未来收益，得到 forward alpha。
4. 在每个时间截面对 forward alpha 做 percentile rank。
5. 仅保留动态票池内资产的标签。

该标签不是直接交易收益，而是横截面相对强弱排序目标。

## 模型训练

主线模型采用 LightGBM 与 CatBoost 的滚动集成训练。训练参数由 `config.py` 管理。

核心训练口径：

- 训练窗口：`4320` 小时，约 6 个月。
- 测试窗口：`720` 小时，约 1 个月。
- Embargo：`24` 小时，即 `2 x FORWARD_HOURS`，用于降低标签重叠带来的未来信息泄露。
- 模型：LightGBM Regressor 与 CatBoost Regressor。
- 融合方式：分别对两个模型的预测值按时间截面做 z-score，再按 `0.5 / 0.5` 权重融合。
- 特征重要性：取最后一个 fold 中 LightGBM 与 CatBoost 的归一化重要性均值。

训练输出包括：

```text
rolling_predictions.parquet
selected_features_validated.csv
feature_importance.csv
fold_overfit_diagnostics.csv
loss_curves.png
```

## 策略组合与回测

`StrategyAssembler` 将模型预测值转化为交易组合。核心逻辑如下：

1. 仅在滚动训练得到的样本外测试时间上生成预测与回测。
2. 预测结果转为宽表，按每个时间截面对币种排序。
3. 剔除 BTC/ETH overlay 标的，不让其参与普通山寨币排名。
4. 选择预测分数最高的 Top 10 作为多头。
5. 选择预测分数最低的 Bottom 10 作为空头。
6. 核心山寨币多空组合保持对称：多头总权重 `+1.0`，空头总权重 `-1.0`。
7. 每 `96` 小时调仓一次。
8. 使用持仓 buffer 机制保留仍处于较优分位的原持仓，降低换手。
9. 根据 BTC 日线 RSI 控制 BTC/ETH 小仓位 overlay。
10. 当 BTC 在 `12` 小时内跌幅超过 `5%` 时触发 crash protection，缩减山寨币空头并强制配置 BTC/ETH 多头 overlay。

交易成本口径：

- Maker fee：`0.02%`。
- Taker fee：`0.05%`。
- Slippage：`0.10%`。
- 回测扣除 `TAKER_FEE + SLIPPAGE`。
- 仅在新开仓、平仓或方向切换时计入真实交易成本。

绩效指标包括：

- total return
- annualized return
- annualized volatility
- Sharpe ratio
- max drawdown
- Calmar ratio
- win rate
- profit/loss ratio
- average turnover
- index return
- excess return
- information ratio
- long return
- short return
- portfolio beta

## BTC/ETH 配对套利模块

`套利BTC_ETH/` 是独立统计套利研究分支，研究对象为 BTC 与 ETH 的长期均衡关系。该模块采用以下框架：

- 对 BTC/ETH 价格做 log-price 转换。
- 使用 Engle-Granger 两步法进行协整检验。
- 对残差做 ADF 检验，确认 spread 是否平稳。
- 使用滚动 OLS 或 Kalman Filter 估计动态 hedge ratio。
- 计算 spread 与 z-score。
- 当 z-score 超过阈值时开仓，回归均值时平仓，极端偏离时止损。
- 回测中统计资金曲线、交易记录、收益分布、回撤与绩效指标。

该模块与横截面策略主线相互独立，主要用于展示另一类市场中性策略研究路径。

## 运行方式

进入主线目录：

```bash
cd cross_section
```

运行完整策略流水线：

```bash
python main.py
```

重新计算所有因子，不使用缓存：

```bash
python main.py --no-cache
```

仅基于已有结果重新生成图表：

```bash
python main.py --visualize
```

运行 BTC/ETH 配对套利模块：

```bash
cd 套利BTC_ETH
python main.py
```

## 环境依赖

项目未在根目录提供统一的 `requirements.txt`。根据代码导入，核心依赖包括：

```text
pandas
numpy
matplotlib
scipy
lightgbm
catboost
pyarrow 或 fastparquet
requests
statsmodels
pykalman
seaborn
```

运行前需确保 `data/cross_section/` 下存在主线策略配置中声明的 parquet 文件。

## 输出结果

主线策略输出位于 `cross_section/output/`。常见产物包括：

```text
data_audit_report.txt              数据审计报告
market_index.parquet               成交额加权市场指数
universe_mask.parquet              动态票池 mask
factor_matrix.parquet              因子矩阵
factor_research/                   因子研究报告目录
selected_features.json             最终入模因子
selected_features_validated.csv    最终入模因子 CSV
selected_feature_correlation.csv   入选因子相关性矩阵
rolling_predictions.parquet        滚动样本外预测
feature_importance.csv             特征重要性
predictions.parquet                回测使用的预测矩阵
positions.parquet                  合并持仓
long_positions.parquet             多头持仓
short_positions.parquet            空头持仓
trades.csv                         交易记录
metrics.csv                        回测绩效指标
equity_curve.csv                   净值曲线
equity_curve.png                   净值、回撤、滚动夏普图
monthly_returns.png                月度收益图
feature_importance.png             因子重要性图
position_analysis.png              持仓分析图
ic_decay.png                       IC 衰减图
attribution_analysis.png           收益归因图
return_distribution.png            收益分布图
```

## 适用边界

本项目属于研究与回测工程，不等同于实盘交易系统。适用边界包括：

- 数据依赖小时级历史 K 线，无法刻画盘口深度、撮合延迟和真实成交队列。
- 回测交易成本采用费率与滑点参数化方式，未模拟订单簿冲击成本。
- 模型标签是未来相对收益排序，不是直接收益金额预测。
- 因子检验以历史样本为基础，结果依赖样本区间、币种覆盖、交易所数据质量和市场状态。
- BTC/ETH overlay 使用日线 RSI regime，不构成独立择时策略。
- 配对套利模块使用协整与均值回归假设，当资产关系发生结构性变化时，信号失效风险上升。

## 扩展方向

系统扩展方向包括：

- 增加统一依赖文件与环境锁定文件。
- 增加端到端测试与关键函数单元测试。
- 将数据下载、清洗、研究、训练、回测拆分为可配置命令行任务。
- 引入资金费率、盘口深度、成交量冲击和多交易所数据。
- 将因子研究报告自动化输出为 HTML 或 PDF。
- 增加参数扫描与 walk-forward 稳健性报告。
- 建立实盘前的 paper trading 层，分离研究信号与交易执行。
