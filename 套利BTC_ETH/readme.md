这是 **Two Sigma / WorldQuant** 内部标准的 **Pairs Trading (Statistical Arbitrage)** 策略开发文档。

我们在处理 ETH/BTC 这种“强相关但非固定挂钩”的资产对时，绝不能只看 K 线图画几条线。我们需要严谨的**计量经济学（Econometrics）**框架。

以下是开发该策略的 **5 个标准阶段**：

---

### 第一阶段：数学定义与假设检验 (Foundations)

在写代码之前，必须先通过数学测试。**不要假设**它们是均值回归的，要**证明**它。

#### 1. 对数价格转换 (Log-Prices)
金融资产价格分布通常是偏态的。为了让数据更符合正态分布假设，必须使用对数价格。
$$ p_t^{ETH} = \ln(Price_t^{ETH}) $$
$$ p_t^{BTC} = \ln(Price_t^{BTC}) $$

#### 2. 协整性检验 (Cointegration Test)
**相关性 (Correlation) $\neq$ 协整性 (Cointegration)。**
*   **相关性**：两个醉汉走在路上，大概方向一致。
*   **协整性**：一个醉汉牵着一条狗。虽然他们各自乱晃（非平稳），但他们之间的距离（残差）是受限的（平稳的）。

**执行步骤**：
使用 **Engle-Granger Two-Step Method**：
1.  **OLS 回归**：$p_t^{ETH} = \alpha + \beta \cdot p_t^{BTC} + \epsilon_t$
2.  **ADF 检验 (Augmented Dickey-Fuller)**：对残差 $\epsilon_t$ 进行单位根检验。
    *   如果 p-value < 0.05，拒绝原假设（存在单位根），证明残差是平稳的。
    *   **结论**：如果平稳，说明 ETH 和 BTC 之间存在长期均衡关系，策略可行。**否则，禁止交易。**

---

### 第二阶段：动态对冲比率 (Dynamic Hedge Ratio)

这是业余选手和专业量化的分水岭。
**业余选手**：固定 1 ETH 对冲 14 BTC（假设价格比是 14）。
**专业量化**：计算 Beta ($\beta$)。ETH 的波动率通常是 BTC 的 1.2 倍左右。如果你做 1:1 金额对冲，你实际上是在**净做多波动率**。

#### 方案 A：滚动 OLS (Rolling OLS) - *MVP版本*
使用过去 N 天（例如 30天/720小时）的数据，每天重新计算 $\beta$。
$$ \text{Spread}_t = p_t^{ETH} - \beta_{rolling} \cdot p_t^{BTC} $$

#### 方案 B：卡尔曼滤波 (Kalman Filter) - *机构版本*
市场结构在变（如 ETF 通过、以太坊升级）。OLS 有滞后性。
卡尔曼滤波将 $\beta$ 视为一个随时间变化的“隐藏状态”，实时更新。它对突变的反应比 OLS 快得多。

---

### 第三阶段：信号生成系统 (Signal Generation)

构建具体的交易信号。我们交易的对象是 **Z-Score（标准分）**。

#### 1. 计算 Spread (残差)
$$ \epsilon_t = p_t^{ETH} - (\alpha_t + \beta_t \cdot p_t^{BTC}) $$

#### 2. 计算 Z-Score
$$ Z_t = \frac{\epsilon_t - \mu_{spread}}{\sigma_{spread}} $$
*   $\mu$ (Mean) 和 $\sigma$ (Std) 必须基于滚动窗口（如 24小时 或 72小时）计算，以适应波动率的变化。

#### 3. 计算半衰期 (Half-Life)
这是**最关键**的指标。对 Spread 进行 Ornstein-Uhlenbeck (OU) 过程拟合。
$$ d\epsilon_t = \theta (\mu - \epsilon_t)dt + \sigma dW_t $$
$$ \text{Half-Life} = \frac{\ln(2)}{\theta} $$
*   **含义**：价格回归均值预计需要多久？
*   **应用**：如果 Half-Life = 4小时，适合高频；如果 = 10天，不适合资金费率高的合约市场。**如果 Half-Life 突然变长，说明协整关系正在减弱，应停止交易。**

---

### 第四阶段：交易规则与执行 (Execution Logic)

设定严格的开平仓阈值。

| 信号区间 | 行为 (Action) | 逻辑 |
| :--- | :--- | :--- |
| **$Z > +2.0$** | **Short Spread** | ETH 相比 BTC 太贵。**做空 ETH + 做多 $\beta$份 BTC**。 |
| **$Z < -2.0$** | **Long Spread** | ETH 相比 BTC 太便宜。**做多 ETH + 做空 $\beta$份 BTC**。 |
| **$Z$ 回归到 0** | **Close Position** | 均值回归完成，止盈离场。不要贪婪。 |
| **$Z > +4.0$** | **Stop Loss** | **硬止损**。如果你亏到 4个标准差，说明这不是震荡，而是**结构性崩塌**（脱钩），认赔出局。 |

---

### 第五阶段：Python 代码实现原型 (Prototype)

这是一个可直接用于回测的逻辑核心：

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class StatArbStrategy:
    def __init__(self, lookback_window=720, z_entry=2.0, z_exit=0.0, z_stop=4.0):
        self.lookback = lookback_window # 30天 (按小时线)
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        
    def calculate_metrics(self, df):
        """
        df 需要包含: 'eth_close', 'btc_close' (原始价格)
        """
        # 1. 对数转换
        y = np.log(df['eth_close'])
        x = np.log(df['btc_close'])
        
        # 2. 滚动计算 Beta 和 Spread
        # 注意：这里为了效率使用了简化版滚动OLS，实盘建议用 Kalman Filter
        rolling_beta = []
        rolling_spread = []
        
        for t in range(self.lookback, len(df)):
            # 获取窗口数据
            y_window = y.iloc[t-self.lookback : t]
            x_window = x.iloc[t-self.lookback : t]
            
            # 添加截距项
            x_window_const = sm.add_constant(x_window)
            
            # OLS 回归
            model = sm.OLS(y_window, x_window_const).fit()
            beta = model.params[1]
            alpha = model.params[0]
            
            # 计算当下的 Spread (Out-of-Sample)
            # Spread = Actual_ETH - Predicted_ETH
            current_spread = y.iloc[t] - (alpha + beta * x.iloc[t])
            
            rolling_beta.append(beta)
            rolling_spread.append(current_spread)
            
        # 对齐索引
        calc_index = df.index[self.lookback:]
        series_spread = pd.Series(rolling_spread, index=calc_index)
        series_beta = pd.Series(rolling_beta, index=calc_index)
        
        # 3. 计算 Z-Score
        # Spread 本身也需要标准化
        spread_mean = series_spread.rolling(window=24).mean() # 24h 均值
        spread_std = series_spread.rolling(window=24).std()
        z_score = (series_spread - spread_mean) / spread_std
        
        return z_score, series_beta

    def generate_signals(self, z_score):
        """
        生成持仓信号: 1 (Long Spread), -1 (Short Spread), 0 (Flat)
        """
        signals = pd.Series(index=z_score.index, data=0)
        position = 0 # 0: Empty, 1: Long, -1: Short
        
        for t in z_score.index:
            z = z_score.loc[t]
            
            # 止损逻辑 (Stop Loss)
            if abs(z) > self.z_stop:
                position = 0 # 强制平仓
            
            # 开仓逻辑 (Entry)
            elif position == 0:
                if z > self.z_entry:
                    position = -1 # Short Spread (Short ETH, Long BTC)
                elif z < -self.z_entry:
                    position = 1  # Long Spread (Long ETH, Short BTC)
            
            # 平仓逻辑 (Exit)
            elif position == -1:
                if z <= self.z_exit:
                    position = 0
            elif position == 1:
                if z >= -self.z_exit: # 注意这里是对称的
                    position = 0
                    
            signals.loc[t] = position
            
        return signals
```

---

### 第六阶段：实盘陷阱与风控 (Critical Risks)

在回测这些代码之前，必须考虑以下 **"Alpha Killers"**：

1.  **资金费率陷阱 (Funding Drag)**：
    *   假设模型让你 **Short Spread** (做空 ETH + 做多 BTC)。
    *   如果此时 ETH 的 Funding Rate 是 +0.05%，而 BTC 是 +0.01%。
    *   你做空 ETH 能收到钱，做多 BTC 要付钱。这是有利的（Positive Carry）。
    *   **反之**，如果方向做反了，你可能每天要支付巨大的费率差。**必须将 Funding Rate 作为成本加入回测。**

2.  **执行滑点 (Execution Risk)**：
    *   这是一个双腿策略（2 Legs）。如果你先市价买入 ETH，还没来得及做空 BTC，BTC 突然暴跌，你就有了裸头寸风险。
    *   **实盘要求**：必须使用算法交易（Algo Execution），同时并发下单，或者根据流动性先成交流动性差的一腿（Taker），再成交流动性好的一腿（Maker）。

3.  **Beta 漂移**：
    *   如果 $\beta$ 从 1.2 突然变成 0.8（例如 ETH 独立行情），你的对冲比例就错了。这就是为什么推荐使用 **Kalman Filter** 的原因。

### 行动建议
1.  **数据清洗**：下载 Binance 过去 3 年 BTC 和 ETH 的**每分钟**或**每小时**数据。
2.  **ADF 检验**：先跑一遍全量数据的 ADF 检验，确认大周期上的确存在协整关系。
3.  **简单回测**：把上面的代码跑通，画出资金曲线。
4.  **费率叠加**：手动扣除每天 3 次的资金费率，看看曲线是否还向上。如果加上费率后不赚钱，**放弃该策略**。