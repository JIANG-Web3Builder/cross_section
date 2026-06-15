
---

### 第一部分：特征工程 (Inputs) —— 垃圾进，垃圾出

TFT 的强大在于处理多模态数据，但如果不经清洗和变换直接扔原始数据，它学不到任何东西。

#### 1. 必须构建的输入清单

你需要将输入分为三类，这直接对应 TFT 的配置参数：

**A. 静态变量 (Static Covariates)**
*   **Symbol**: 'BTC'，'ETH','SOL','BNB'

**B. 已知未来变量 (Known Future Inputs)**
*   **时间特征**: `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos` (必须用三角函数编码，不要用 0-23 整数)。
*   **美股日历**: `is_us_market_open` (0/1)。*补充：加入 `is_us_market_close_impact`，即美股收盘前半小时，波动率往往剧烈。*
*   **减半周期**: `log_days_since_halving`。

**C. 过去观测变量 (Past Observed Inputs) —— 最关键部分**
不要直接把 OHLCV 扔进去，必须进行**平稳化 (Stationarity)** 处理。TFT 甚至神经网络都很难处理非平稳的价格序列。

1.  **目标标的 (BTC/ETH/SOL/BNB) 特征**:
    *   **Log Returns**: `log(close_t / close_t-1)`。
    *   **Realized Volatility**: 过去 24h 和 168h 的滚动波动率。
    *   **Volume Change**: `log(vol_t / mean_vol_24h)`。
    *   **Relative Strength**: `RSI` (14, 168)。`Perp_Price - Spot_Price`。如果没有现货数据，用 `Funding Rate` 的 3日滚动均值代替。*这是判断牛熊情绪的最核心指标。*

2.  **宏观与关联资产特征**:
    *   **QQQ/SPX**: 不要放价格，放 **滚动相关性 (Rolling Correlation)**。
    *   **BTC Dominance**: `Change_Rate_24h` 和 `RSI`。如果 BTC.D 飙升且 BTC 跌，是山寨币的地狱模式。

3.  **市场微观结构 (如果数据允许)**:
    *   **Liqudity Proxy**: `Volume / High-Low Range`。衡量流动性充沛程度。

---

### 第二部分：TFT 模型架构与训练逻辑 (The Engine)

这是该方案的核心。不要用默认参数。

#### 1. 时间窗口设计 (Window Engineering)
*   **Max Encoder Length (回看窗口)**: **168 (1周)** 或 **336 (2周)**。
    *   *理由*: Crypto 是 7x24 小时市场，包含明显的周度效应（周末流动性枯竭）。必须让模型看到至少一个完整的周度周期。
*   **Max Prediction Length (预测窗口)**: **24 (1天)** 或 **36 (1.5天)**。
    *   *理由*: 配合你 LGBM 的标签周期。预测太远（如7天）准确率会指数级下降，噪音太大。

#### 2. 预测目标 (Target) —— 别预测价格！
顶级机构绝不直接预测价格（Price Prediction is a fool's errand）。你应该训练 TFT 预测以下**两个目标之一**（或多任务学习）：

*   **目标 A：未来累积收益率的分位数 (Quantiles of Cumulative Return)**
    *   预测未来 24H 的累积收益。
    *   **Loss Function**: `QuantileLoss(quantiles=[0.1, 0.5, 0.9])`。
    *   *输出*: P10 (悲观底), P50 (中位趋势), P90 (乐观顶)。
    *   *P90 - P10* = **预测的不确定性 (Implied Uncertainty)**。

*   **目标 B：波动率/风险 (Realized Volatility)**
    *   预测未来 24H 的波动率。
    *   *用途*: 直接作为风控分母。

**建议**: 先做 **目标 A**，因为它能同时提供方向（P50）和风险（P90-P10）。

#### 3. 训练策略：严格的 Walk-Forward (滚动扩充)

你手里有 2021-2025 的 BTC 数据。
*   **Pre-training (预训练)**: 2021.01 - 2023.12。
    *   Epochs: 50-100 (Early Stopping)。
    *   Learning Rate: 0.01 - 0.001。
    *   Batch Size: 64 or 128。
*   **Incremental Updating (增量更新)**:
    *   进入 2024 年后，模拟真实时间流逝。
    *   每过 **1个月** (或者1个季度)，将这期间产生的新 BTC 数据加入训练集。
    *   **Fine-tune**: 加载上个月的模型权重，在新数据上训练 1-3 个 Epochs，学习率调低 (LR/10)。
    *   *目的*: 适应 2024 年 ETF 通过后的市场结构性变化。

---

### 第三部分：模型输出与集成 (Integration)

训练好 TFT 后，它的输出是什么？怎么喂给 LGBM？

#### 1. TFT 的输出转化 (Feature Extraction)
对于 2024.01 - 2025.12 的每一个小时，TFT 会输出：
*   `pred_p50`: 预测未来 24H BTC 的涨跌幅。
*   `pred_p10`: 预测的最差情况。
*   `pred_p90`: 预测的最好情况。

你需要衍生出三个**宏观因子 (Macro Factors)**，并将其**广播 (Broadcast)** 给 LGBM 里的每一个币种：

1.  **Macro_Trend (宏观趋势)**: `pred_p50`。
    *   *逻辑*: 如果 BTC 预期大涨，LGBM 应该敢于给高 Beta 的山寨币更高分。
2.  **Macro_Risk (宏观风险)**: `pred_p90 - pred_p10`。
    *   *逻辑*: 这是一个预判性的波动率指标。如果不确定性极高，说明变盘在即，LGBM 应降低对所有预测的置信度。
3.  **Macro_Tail_Prob (尾部风险)**: `abs(pred_p10) / (pred_p90 - pred_p10)` (归一化后的下行偏度)。
    *   *逻辑*: 如果 P10 极深，说明左侧肥尾风险大。

#### 2. 配合 LGBM 训练 (Training LGBM)
在 `factor_engineering.py` 中：
*   将上述 3 个特征列拼接到 `factor_matrix` 中。
*   所有币种在同一时间共享这 3 个值。
*   **注意**: 必须确保 TFT 的预测是 Out-of-Sample 的（即用截止到 T 时刻的数据预测 T+24，然后这个预测值作为 T 时刻的特征）。

LGBM 会自动学习到如下规则：
> *Interaction: If `Macro_Risk` is High AND `Coin_Momentum` is High -> Reduce Weight (防止假突破).*

---

### 第四部分：交易执行与风控 (Execution & Risk)

除了作为 LGBM 的特征，TFT 的输出必须直接干预交易执行。这是**双重保险**。

在 `strategy_backtest.py` 中，加入**“熔断机制” (Circuit Breaker)**：

```python
# 伪代码：基于 TFT 的动态风控

def calculate_position_size(lgbm_signals, tft_output, base_leverage=1.0):
    
    # 1. 提取 TFT 信号
    expected_market_return = tft_output['p50']
    market_uncertainty = tft_output['p90'] - tft_output['p10']
    
    # 2. 状态定义 (Regime Definition)
    
    # 状态 A: 极度危险 (High Uncertainty + Negative Trend)
    # 例如：预测波动率 > 历史80分位 且 预期下跌
    if market_uncertainty > quantile_80_vol and expected_market_return < -0.02:
        return 0.0  # 空仓/熔断
        
    # 状态 B: 确定性上涨 (Low Uncertainty + Positive Trend)
    elif market_uncertainty < quantile_40_vol and expected_market_return > 0.01:
        return base_leverage * 1.5  # 激进模式
        
    # 状态 C: 震荡/噪音 (High Uncertainty + Flat Trend)
    elif market_uncertainty > quantile_80_vol:
        return base_leverage * 0.5  # 防守模式
        
    else:
        return base_leverage

    # 3. 截面多空对冲
    # 即使在激进模式下，依然保持多空对冲，只是总敞口放大
```

在TFT（Temporal Fusion Transformer）的训练架构配置中，针对你的加密货币（Crypto）多因子预测任务，有几个**至关重要的架构细节**决定了模型的成败。

基于你之前提供的 Alpha 101 因子和多币种数据，以下是作为顶级量化工程师的**架构配置检查清单**。

---

### 一、 输入层的变量映射 (Input Mapping) —— TFT 的灵魂

TFT 最大的优势是它明确区分了“我们知道什么”和“我们不知道什么”。不要把所有特征一股脑扔进去，必须严格分类。

#### 1. Static Categoricals (静态类别变量)
*   **配置**: `group_ids=["symbol"]`, `static_categoricals=["symbol", "asset_type"]`
*   **架构原理**: TFT 会为每个 `symbol` 学习一个 **Entity Embedding（实体嵌入向量）**。
*   **作用**: 这让模型理解“SOL”和“BTC”的个性差异。比如模型学到 SOL 的 Embedding 向量在“高波动”维度上值很大，它就会在预测 SOL 时自动放宽置信区间。

#### 2. Time Varying Known Reals (已知时变变量)
*   **配置**: `["time_idx", "day_of_week", "hour_of_day", "month"]`
*   **关键点**: **这里严禁放入任何未来不可知的数据。**
*   **宏观数据的特殊处理**: 如果你认为宏观数据（如标普500相关性）变化很慢，或者你有宏观的预测值，可以放这里；**但最严谨的做法是放入 Observed（未知）**，防止未来函数。
*   **作用**: 帮助 Decoder（解码器）在预测未来时，利用日历效应（比如周末成交量低）来修正预测。

#### 3. Time Varying Unknown Reals (观测时变变量)
*   **配置**: 所有的 Alpha 因子、技术指标、Log Returns、成交量。
    *   `["log_ret", "alpha001", "rsi_14", "network_activity_z", ...]`
*   **架构原理**: 这些数据只会进入 **Encoder（编码器）**，无法进入 Decoder。
*   **注意**: 你的因子列表很长（几十个），TFT 的 **Variable Selection Network (VSN)** 会自动计算每个因子的权重。你不需要手动剔除弱因子，但要确保没有强相关且无意义的噪音。

---

### 二、 时间窗口设定 (Time Context) —— 视野决定成败

TFT 处理的是序列，你需要设定“看多久历史”和“预测多久未来”。

#### 1. max_encoder_length (回看窗口)
*   **建议值**:
    *   **小时线 (Hourly)**: `168` (1周) 或 `336` (2周)。
    *   **日线 (Daily)**: `60` (2个月) 或 `90` (1季度)。
*   **架构思考**: 窗口必须长到足以包含一个完整的市场形态（如一次完整的震荡或趋势启动）。如果太短（比如只看3天），模型看不出 Alpha 因子（如 `roc_72`）的长期衰减效果。

#### 2. max_prediction_length (预测步长)
*   **建议值**:
    *   **小时线**: `24` (预测未来24小时)。
    *   **日线**: `1` 到 `7` (预测未来1天到1周)。
*   **核心警告**: **不要贪心**。Crypto 市场的噪声随时间呈指数级增长。TFT 预测一步（One-step ahead）最准，预测越远方差越大。如果你是做高频策略，设为 `1` 就够了。

---

### 三、 模型内部超参数 (Model Hyperparameters) —— 防止过拟合

鉴于你的数据（2023-2025有效数据）对于深度学习来说属于**“小样本”**（Small Data），防止过拟合是架构调整的核心。

#### 1. hidden_size (隐藏层维度)
*   **建议**: `64` 或 `128`。
*   **避坑**: 不要用默认的 `512` 或更大。你的数据量撑不起这么大的参数空间，太大会导致模型死记硬背（Memorization），导致 Validation Loss 很低但回测亏钱。

#### 2. dropout (丢弃率)
*   **建议**: `0.3` 到 `0.5` (非常激进)。
*   **原因**: 金融信噪比极低。高 Dropout 迫使网络不依赖单一特征（比如只看 RSI），而是学习多因子的鲁棒组合。

#### 3. attention_head_size (注意力头数)
*   **建议**: `4`。
*   **原因**: 能够捕捉不同模式（一个头关注短期突变，一个头关注长期趋势，一个头关注宏观联动）。

---

### 四、 损失函数 (Loss Function) —— 也是架构的一部分

TFT 的标准输出不是一个数字，而是一个**分布**。

*   **配置**: `QuantileLoss(quantiles=[0.1, 0.5, 0.9])`
*   **架构意义**:
    *   `0.5` (中位数): 你的点预测值（Point Forecast），用来做方向判断。
    *   `0.1` 和 `0.9`: 预测区间（Prediction Interval）。
*   **实战用法**: 当 `0.9 - 0.1` 的差值（预测带宽）突然变大时，说明**模型自己也慌了**（不确定性增加）。这时候你可以选择**空仓观望**。这是一个天然的风险控制信号。

---

### 五、 变量选择网络 (Variable Selection Network, VSN) 的解释性

这是 TFT 最迷人的架构特性。训练完后，**一定要调用解释性分析接口**。

1.  **Encoder Variable Importance**: 查看哪些因子在“回顾历史”时最重要。
    *   如果发现 `time_idx` 权重最高，说明模型没学到逻辑，只学到了线性趋势（失败）。
    *   如果发现你的 `alpha006` 或 `network_activity` 权重很高，说明因子有效。
2.  **Decoder Variable Importance**: 查看哪些已知变量（如星期几）对预测未来最重要。

### 六、 总结：你的 `TimeSeriesDataSet` 代码配置模板

这是直接落地到代码的架构蓝图：

```python
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# 关键配置
max_encoder_length = 168  # 看过去一周 (小时线)
max_prediction_length = 24 # 预测未来24小时
training_cutoff = data["time_idx"].max() - max_prediction_length

training_dataset = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="target",                # 你的预测目标 (如未来收益率)
    group_ids=["symbol"],           # 多币种训练的核心
    min_encoder_length=max_encoder_length // 2, # 允许变长输入，增加鲁棒性
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    
    # 1. 静态变量 (让模型认识不同的币)
    static_categoricals=["symbol"],
    
    # 2. 已知变量 (日历)
    time_varying_known_reals=["time_idx", "day_of_week", "hour_of_day"],
    
    # 3. 观测变量 (你的Alpha因子大军)
    time_varying_unknown_reals=[
        "log_ret", "volatility", 
        "rsi_14", "macd_hist", "alpha101_proxy", 
        "network_activity_z", "spx_corr" # 注意：如果spx_corr是滚动的，放这里比较安全
    ],
    
    # 4. 目标变换 (如果是预测收益率，通常不需要Log变换，因为已经是Log Return了)
    target_normalizer=None,  # 或者 GroupNormalizer(groups=["symbol"])
    
    # 5. 特征归一化 (重要！让所有因子处于同一量级)
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# 模型初始化架构
tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=0.03,
    hidden_size=64,          # 控制模型大小
    attention_head_size=4,
    dropout=0.4,             # 强正则化
    hidden_continuous_size=16, # 连续变量的Embedding大小
    loss=QuantileLoss(),
    optimizer="Ranger",      # 推荐优化器
    reduce_on_plateau_patience=4,
)
```

**最后一句叮嘱**：TFT 架构中最容易被忽视的是 **`static_categoricals=["symbol"]`**。有了这个，模型才能在训练 BTC 时借鉴 ETH 的规律，在训练 SOL 时借鉴 BNB 的规律（Transfer Learning），这正是你用多币种合并训练的最大红利。


把D:\strategy\TFT\factor里面的parquet合并成一个大表，按照D:\strategy\TFT\TFT_LGBM_Architecture_v2.md里面的架构和步骤来构造和训练TFT