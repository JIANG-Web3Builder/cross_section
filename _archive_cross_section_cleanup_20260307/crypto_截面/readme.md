策略类型：中频截面多因子策略 (Mid-Frequency Cross-Sectional Multi-Factor)
预测目标：未来 12~24 小时 的收益排名（Ranking）。
核心模型：LightGBM (GBDT)。
风控逻辑：市场波动率择时 + 动态分散持仓。

第一阶段：数据工程 (Data Engineering) —— 决定上限
既然数据是加密的，你无法知道哪个是BTC。你需要自己构建“大盘指数”。
步骤 1：构建动态票池 (Dynamic Universe)
Crypto市场充斥着僵尸币。
操作：计算每个 Asset_ID 过去 24小时的成交金额
筛选：每个时间点，只保留成交金额排名前 50 的 ID。
步骤 2：构建“合成大盘指数” (Synthetic Market Index)
操作：计算这 Top 50 资产的平均收益率（Equal Weighted Index）。
用途：这个“合成指数”将代替 BTC，用于后续的择时判断和去 Beta。
Survivorship Bias处理：确保回测时只用当时可交易的币种
步骤 3：数据对齐与去极值
对齐：使用 Pandas 的 MultiIndex (Time, Asset_ID) 格式。
去极值：Crypto 经常有插针（瞬间涨跌 50%）。对 OHLCV 数据进行 Winsorization（缩尾处理），比如超过 3倍标准差的值强行拉回。

第二阶段：因子工厂 (Factor Engineering) —— 挖掘规律
由于不知道具体币种，我们只用量价因子。注意：1小时数据的参数要放大。
步骤 1：构建 4 大类因子 (共约 30-50 个)
利用 pandas-ta 批量生成，重点关注中周期：
动量类 (Trend)：
RSI_24 (过去1天), RSI_168 (过去1周)。
Slope (线性回归斜率，衡量趋势强度)。
反转类 (Reversal)：
Williams %R (24周期)。
Bias (乖离率)：Close / MA(24) - 1。
波动率 (Volatility)：
ATR_24 / Close (标准化波动率)。
StdDev_24。
量价关系 (Volume-Price)：
PVT (Price Volume Trend)。
VWAP 偏离度：Close / VWAP - 1。
步骤 2：截面标准化 (Cross-Sectional Normalization) —— 最关键一步
因为不同币种价格差异巨大（有的价格 0.0001，有的 50000）。
操作：对每一个时间切片（Timestamp），对所有 Top 50 的币种的因子值做 Z-Score。
结果：所有因子变成了以 0 为中心，标准差为 1 的分布。这样模型才能在横截面上比较“谁更强”。

第三阶段：模型训练 (Modeling) —— 预测排序
步骤 1：标签构建 (Label Generation)
原始目标：计算未来 24 小时收益率。
去大盘 Beta：用未来 24 小时收益率-IndexRate（用第一阶段做的合成指数）。
排序 (Ranking)：将 AlphaRet转换为 0 到 1 之间的 Rank 值。这是我们训练的 Target。
步骤 2：训练 LightGBM
切分数据：前 70% 训练，后 30% 验证。严禁 shuffle（打乱顺序），必须按时间轴切分。
参数：
objective: 'regression' (或者 'lambdarank' 进阶版)。
metric: 'l2' (MSE)。
max_depth: 4~6 (防止过拟合)。
learning_rate: 0.02。
训练：输入因子，预测 Rank。
步骤 3：特征筛选
训练完看 feature_importance。
剔除重要性极低（< 1%）的垃圾因子。
保留前 20-35 个因子重新训练。
相关性分析：因子间相关矩阵，剔除高度相关因子
合成方法：等权、IC加权、IC_IR加权
正交化：Gram-Schmidt正交，避免因子共线性

第四阶段：策略组装与择时 (Strategy Assembly)
步骤 1：择时模块 (Timing Filter)
利用第一阶段构建的“合成大盘指数”。
规则：
计算合成指数的 MA(168)。
计算合成指数的 Volatility(24)。
空仓信号：如果 (指数 < MA168) 或者 (波动率 > 历史 90% 分位数)，则判定为高危市场，仓位 = 0。
步骤 2：持仓逻辑
在非空仓状态下：
选取 LightGBM 预测 Rank 最高的前 5 名。
权重：因子值加权
换仓频率：建议 每 4 小时重新计算一次排名。不要每小时换仓，手续费会炸。

注意加入手续费（Maker/Taker）、滑点。然后不要使用未来函数或者未来数据