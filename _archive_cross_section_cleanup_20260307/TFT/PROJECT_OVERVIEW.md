# TFT-LGBM 加密货币量化策略项目详解 (Project Documentation)

## 1. 项目架构与核心理念

本项目构建了一个 **混合人工智能策略 (Hybrid AI Strategy)**，旨在结合 **Temporal Fusion Transformer (TFT)** 的宏观择时能力与 **LightGBM (LGBM)** 的截面选币能力（LGBM部分待集成）。

### 核心逻辑
1.  **宏观感知 (Macro Awareness)**: 使用 TFT 深度学习模型，输入过去 **14天 (336小时)** 的多维数据，识别当前的市场体制 (Regime)。
2.  **体制分类 (Regime Classification)**: 将市场划分为 **强势多头 (Strong Bull)**、**强势空头 (Strong Bear)**、**震荡 (Dead Calm)** 或 **噪音 (Noisy)**。
3.  **动态风控 (Dynamic Risk)**: 根据市场体制动态调整仓位杠杆，并在高波动或不利行情下自动降仓或空仓。

---

## 2. 文件结构详解 (File Structure)

| 文件名 | 功能描述 | 关键逻辑 |
| :--- | :--- | :--- |
| **`config_tft.py`** | **全局配置中心** | 定义了所有模型参数、特征列表、训练窗口、风控阈值。是项目的“大脑”。 |
| **`calculate_factors.py`** | **因子计算** | 计算原始行情数据的技术指标。**特点**：移除了RSI/MACD等零售指标，专注于 **动量 (Momentum)** 和 **波动率结构 (Volatility Structure)**。 |
| **`tft_data_processor.py`** | **数据工程管道** | 将 BTC 和 ETH 的因子数据与宏观数据合并，进行清洗、对齐、标准化，并生成 **TFT训练所需的宽表 (Parquet)**。支持多币种堆叠 (Stacking)。 |
| **`train_tft.py`** | **基座模型训练** | 使用 **2021-2023年** 的历史数据训练一个通用的 TFT 基座模型。学习市场通用的特征表达。 |
| **`train_tft_lgbm_walkforward.py`** | **滚动微调与预测** | **核心实战脚本**。模拟 **2024-2026年** 的每日交易。每天使用过去 **6个月** 数据微调模型，预测未来 **24小时** 的概率。 |
| **`strategy_engine.py`** | **策略执行引擎** | 将模型的预测概率 ($P_{long}, P_{short}, P_{neutral}$) 转化为具体的 **交易信号** 和 **目标仓位**。 |
| **`backtest_tft_lgbm.py`** | **回测系统** | 基于事件驱动的回测，计算资金曲线、最大回撤、夏普比率等绩效指标。 |

---

## 3. 数据工程与特征体系 (Data Engineering)

**处理脚本**: `tft_data_processor.py`

本项目采用 **多币种堆叠 (Multi-Symbol Stacking)** 结构，将 BTC 和 ETH 的数据在时间轴上堆叠，同时共享宏观特征。

### 3.1 特征输入 (Features)
我们摒弃了传统的零售技术指标，采用了更科学的统计特征：

*   **资产特有特征 (Asset-Specific) - 通用命名**:
    *   **动量类**: `log_return_1h`, `log_return_24h` (短期), `log_return_168h` (中期周线动量)。
    *   **波动率类**: `volatility_24h` (短期波动), `volatility_168h` (中期波动)。
    *   **衍生品市场特征 (Derivative Market)**:
        *   `funding_rate`: 资金费率 (多空情绪指标)。
        *   `open_interest_value`: 持仓量 (市场热度/资金博弈)。
        *   `top_trader_ls_ratio`: 大户多空比 (聪明钱流向)。
        *   `taker_buy_sell_ratio`: 主动买卖比 (即时盘口压力)。
    *   **衍生特征**:
        *   `volatility_term_structure`: $\frac{Vol_{24h}}{Vol_{168h}}$ (衡量市场是否处于恐慌加速期)。
        *   `volume_change_ratio`: 成交量异动。
*   **宏观共享特征 (Shared Macro) - 广播给所有币种**:
    *   `btc_dominance_change`: 比特币市占率变化 (判断资金是在流入大饼还是山寨)。
    *   `spx_correlation_rolling_7d`: 与美股(S&P500)的7天滚动相关性 (判断宏观脱钩/耦合)。
    *   `eth_btc_ratio_change`: ETH/BTC 汇率变化 (风险偏好 Risk-On/Off 指标)。

### 3.2 预测目标 (Labeling)
**方法**: **三重势垒法 (Triple Barrier Method)**
不再预测具体的收盘价，而是预测 **未来 24小时内** 价格最先触碰哪条线：
*   **上轨 (Upper Barrier)**: 当前价格 $\times (1 + 0.5 \times Volatility)$
*   **下轨 (Lower Barrier)**: 当前价格 $\times (1 - 0.5 \times Volatility)$
*   **类别定义**:
    *   **Class 0 (Hold/Neutral)**: 24小时内未触碰任何轨（震荡市）。
    *   **Class 1 (Long)**: 先触碰上轨。
    *   **Class 2 (Short)**: 先触碰下轨。

---

## 4. 模型配置与训练逻辑 (Training Pipeline)

**配置文件**: `config_tft.py`

### 4.1 TFT 模型参数
*   **回看窗口 (Lookback)**: **336 小时 (14天)**。模型能“看到”过去两周的完整走势。
*   **预测窗口 (Forecast)**: **24 小时 (1天)**。每日进行一次决策。
*   **网络结构**: 4层 LSTM，隐藏层大小 96，Dropout 0.3 (强正则化防止过拟合)。
*   **损失函数**: **CrossEntropy** (多分类交叉熵)。

### 4.2 训练流程 (两阶段法)

#### **阶段一：基座训练 (Base Training)**
*   **脚本**: `train_tft.py`
*   **时间段**: 2021-01-01 至 2024-01-01 (3年)。
*   **目的**: 让模型学习加密货币市场的通用规律和特征表达 (Encoder)。

#### **阶段二：滚动微调 (Walk-Forward Fine-tuning)**
*   **脚本**: `train_tft_lgbm_walkforward.py`
*   **时间段**: 2024-01-01 至 2026-01-01。
*   **逻辑**: 模拟真实交易环境，**严禁未来函数**。
    1.  **加载** 2023年底的基座模型。
    2.  **每日更新**:
        *   截取过去 **4320小时 (6个月)** 的数据作为微调集。
        *   使用极低的学习率 (`1e-4`) 对模型进行 **3个 Epoch** 的快速微调。
        *   这使得模型能适应最近的市场风格（如“AI热潮”或“加息周期”），同时保留长期记忆。
    3.  **每日预测**: 生成未来 24小时的看多/看空概率。

---

## 5. 策略逻辑 (Strategy Logic)

**脚本**: `strategy_engine.py` (集成在 walkforward 流程中)

模型输出的是三个概率：$P_{hold}, P_{long}, P_{short}$。策略层将其转化为交易指令。

### 5.1 信号评分 (Score Calculation)
我们定义一个 **方向性分数 (Directional Score)**：
$$ Score = \frac{P_{long}}{P_{long} + P_{short}} $$
*   $Score \approx 1.0$: 极度看多
*   $Score \approx 0.0$: 极度看空
*   $Score \approx 0.5$: 多空平衡

### 5.2 体制判定表 (Regime Matrix)

| 市场体制 (Regime) | 触发条件 | 仓位/杠杆建议 |
| :--- | :--- | :--- |
| **强势多头 (Strong Bull)** | $Score > 0.6$ 且 $P_{hold} < 0.5$ | **1.5x 做多** (根据 Score 强度动态调整) |
| **强势空头 (Strong Bear)** | $Score < 0.4$ 且 $P_{hold} < 0.5$ | **1.0x 做空** (保守做空) |
| **死寂震荡 (Dead Calm)** | $P_{hold} > 0.6$ | **0.0x 空仓** (避免磨损) |
| **噪音 (Noisy)** | 其他情况 | **0.0x 空仓** (信号不明) |

### 5.3 风险控制 (Risk Management)
1.  **波动率目标 (Vol Target)**: 设定年化波动率目标为 **50%**。如果市场实际波动率过高，自动降低仓位。
2.  **止损 (Stop Loss)**: 组合净值回撤超过 **15%** 时，强制平仓并通过冷却期 (24h) 停止交易，防止连续亏损。
3.  **杠杆限制**: 多头最大 **1.5倍**，空头最大 **1.0倍**。

---

## 6. 如何运行项目 (Execution Steps)

按顺序执行以下命令：

1.  **生成因子**:
    ```bash
    python D:\strategy\TFT\calculate_factors.py
    ```
    *(生成 BTCUSDT_factors.parquet 等)*

2.  **数据处理与合并**:
    ```bash
    python D:\strategy\TFT\tft_data_processor.py
    ```
    *(生成 D:\strategy\TFT\final_df\tft_training_dataset.parquet)*

3.  **训练基座模型**:
    ```bash
    python D:\strategy\TFT\train_tft.py
    ```
    *(耗时较长，生成 models/tft_base.pth)*

4.  **滚动微调与信号生成**:
    ```bash
    python D:\strategy\TFT\train_tft_lgbm_walkforward.py
    ```
    *(模拟每日实盘，生成 output/tft_signals.parquet)*

5.  **策略回测**:
    ```bash
    python D:\strategy\TFT\backtest_tft_lgbm.py
    ```
    *(输出回测报告和资金曲线)*
