# 初级量化分析师模拟面试稿

这份材料按技术主管面试的节奏组织，目标不是背题，而是训练你把 JD 里的业务问题转成数据分析、检测逻辑、风控指标和自动化流程。

岗位关键词：

- 业务领域：风险管理、反洗钱监控、合规监测、市场推广活动分析。
- 数据对象：交易、订单、交易流水、市场数据、营销活动、客户行为、入金行为。
- 技术能力：Python、SQL、结构化数据清洗、特征构建、统计建模、机器学习、数据可视化、自动化报告。
- 工作方式：把业务需求转成清晰的分析逻辑、模型规则或自动化流程，并能向技术和非技术团队解释发现。

面试官默认判断标准：

- 你是否能先澄清业务目标，再定义指标和数据口径。
- 你是否知道误报率、漏报率、样本偏差、数据泄露、时间窗口这些问题。
- 你是否能写出可落地的 SQL 和 Python 分析思路，而不是只说模型名。
- 你是否能把分析结果讲成业务方能采取行动的结论。

## 0. 开场定位

### 问题 0.1：请你用 2 分钟介绍自己，重点说一个和数据分析、风控、交易或金融相关的项目。

追问：

- 这个项目的数据来源是什么？
- 你具体负责哪一部分？
- 最后产出了什么结果，是报告、模型、规则、看板还是自动化脚本？
- 如果让你重做，你会优先改哪一部分？

考察点：

- 是否能把经历讲成“业务问题 -> 数据 -> 方法 -> 结果 -> 反思”。
- 是否能区分自己做的事和团队做的事。
- 是否有复盘意识，而不是只描述工具。

优秀回答要点：

- 一开始说清业务目标，例如识别异常交易、提升活动转化、降低误报率。
- 说明核心字段和数据粒度，例如账户、订单、交易流水、时间戳、金额、渠道、活动 ID。
- 讲清方法，例如规则筛选、分组统计、异常检测、分类模型、可视化看板。
- 说出结果的可操作性，例如输出高风险账户清单、优化阈值、减少人工复核量。

危险回答信号：

- 只说“我用了 pandas、sklearn、画图”，但说不清业务目的。
- 只讲模型准确率，不讲误报、漏报和业务成本。
- 把课程作业包装成真实业务项目，但经不起数据口径追问。

## 1. 第一轮：基础筛选

这一轮主要确认你是否具备 JD 要求的数学、统计、Python、SQL 和金融业务基础。

### 问题 1.1：如果我要你分析一批交易流水，你会先做哪些数据质量检查？

追问：

- 交易时间字段可能有哪些坑？
- 金额字段异常你怎么处理？
- 重复订单和撤单、冲正、失败交易怎么区分？
- 如果客户 ID 缺失，你会怎么判断影响范围？

考察点：

- 是否具备结构化数据清洗意识。
- 是否理解金融交易数据不能随便删除异常值。
- 是否会先看口径，再谈建模。

优秀回答要点：

- 先确认字段含义和粒度：一行是一笔订单、一笔成交、一条流水，还是一次状态变更。
- 检查主键、时间戳、账户 ID、订单 ID、交易方向、币种、金额、手续费、状态字段。
- 区分数据错误和真实异常：极大金额可能是高风险交易，也可能是币种单位错误。
- 对撤单、失败、退款、冲正建立统一状态口径，避免重复计算交易量。
- 输出数据质量报告，包括缺失率、重复率、异常值分布、时间覆盖和字段可用性。

### 问题 1.2：请解释均值、中位数、方差、标准差、分位数在风控分析里的作用。

追问：

- 为什么异常金额监控不能只用均值加三倍标准差？
- 如果金额分布极度右偏，你会用什么统计量？
- 什么时候用 z-score，什么时候用分位数阈值？

考察点：

- 是否理解金融数据常见的厚尾和偏态。
- 是否能把基础统计量用于真实异常检测。

优秀回答要点：

- 均值和标准差适合近似对称、稳定的分布，但交易金额常见厚尾。
- 中位数、MAD、P95、P99 对极端值更稳健。
- z-score 适合标准化后比较异常程度，分位数适合制定业务阈值。
- 风控阈值不能只看统计显著，还要看人工复核能力和业务损失。

### 问题 1.3：你如何判断一个风控规则是否有效？

追问：

- 没有明确标签时怎么评估？
- 误报率和漏报率哪个更重要？
- 规则上线前你会做哪些回测？

考察点：

- 是否理解检测逻辑的评价不等于普通分类模型评价。
- 是否能结合业务成本做判断。

优秀回答要点：

- 有标签时看 precision、recall、F1、PR-AUC，并按风险等级分层。
- 没标签时看命中样本的人工复核结果、历史可疑账户重合度、行为聚类差异和案例抽样。
- 回测要按时间切分，不能用未来信息调阈值。
- 评估规则覆盖率、误报率、漏报风险、人工处理量和新增价值。

### 问题 1.4：写 SQL 时，WHERE 和 HAVING 有什么区别？在异常账户筛选里怎么用？

追问：

- 如何筛选 7 天内交易次数超过 50 次且总金额超过 10 万的账户？
- 如果还要排除失败交易，条件放在哪里？
- 如果要取每个账户最近一笔交易，你会怎么写？

考察点：

- 是否掌握筛选、聚合、窗口函数。
- 是否能把业务筛选条件转成 SQL 逻辑。

优秀回答要点：

```sql
SELECT
    account_id,
    COUNT(*) AS trade_count,
    SUM(amount) AS total_amount
FROM trades
WHERE trade_status = 'success'
  AND trade_time >= CURRENT_DATE - INTERVAL '7 day'
GROUP BY account_id
HAVING COUNT(*) > 50
   AND SUM(amount) > 100000;
```

说明：

- `WHERE` 在聚合前过滤明细行，例如交易状态、时间范围。
- `HAVING` 在聚合后过滤账户级指标，例如次数、总金额。
- 最近一笔交易通常用 `ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY trade_time DESC)`。

### 问题 1.5：Python 中 pandas 的 groupby、merge、rolling 分别适合解决什么问题？

追问：

- groupby 后如何同时算交易次数、总金额、最大金额？
- merge 时怎么避免一对多导致行数膨胀？
- rolling 计算 7 日交易金额时要注意什么？

考察点：

- 是否能用 Python 处理结构化交易和客户行为数据。
- 是否知道常见数据处理错误。

优秀回答要点：

```python
account_stats = (
    trades.groupby("account_id")
    .agg(
        trade_count=("order_id", "nunique"),
        total_amount=("amount", "sum"),
        max_amount=("amount", "max"),
    )
    .reset_index()
)
```

- `groupby` 用于账户、活动、渠道、日期等维度聚合。
- `merge` 用于拼接客户画像、账户风险等级、活动信息，要检查连接键唯一性。
- `rolling` 用于时间窗口特征，必须按账户和时间排序，避免跨账户滚动。

## 2. 第二轮：技术主管深挖

这一轮会更接近真实工作场景，重点看你是否能把模糊需求拆成可执行方案。

### 问题 2.1：现在业务方说“最近可疑交易变多了”，你会怎么开始分析？

追问：

- 你会先要哪些数据？
- 如何判断是真的变多，还是检测规则变严了？
- 如何定位是哪类账户、渠道或交易类型贡献了增长？

考察点：

- 是否先定义指标口径，而不是直接建模。
- 是否会做分解分析和归因。

优秀回答要点：

- 先定义“可疑交易”的口径：规则命中、人工确认、合规上报，还是模型高分。
- 拉取时间序列：每日交易量、可疑交易数、可疑率、命中规则分布、人工确认率。
- 拆分维度：账户类型、注册时间、渠道、币种、交易方向、金额分层、国家地区、设备。
- 对比规则变更、活动上线、市场波动和数据延迟，排除口径变化。
- 输出结论时区分现象、原因和建议，例如提高某规则阈值、增加某渠道复核、补充新特征。

### 问题 2.2：请设计一个异常交易检测逻辑，用于识别短时间内频繁小额交易后突然大额转出的账户。

追问：

- 你会定义哪些特征？
- 阈值怎么定？
- 如何降低误报？
- 如果要做成自动化监控，你会怎么安排流程？

考察点：

- 是否能把业务模式转成特征和规则。
- 是否理解规则监控需要可解释性和可维护性。

优秀回答要点：

- 账户级时间窗口特征：近 1 小时、24 小时、7 天的交易次数、入金金额、出金金额、最大单笔金额。
- 行为变化特征：当前出金金额相对历史中位数、P95、账户余额的比例。
- 模式特征：小额入金次数、不同付款来源数量、入金后出金间隔、是否首次大额出金。
- 阈值可以从历史分位数、人工复核能力、已知可疑样本中校准。
- 降误报方式：按账户生命周期、客户等级、历史交易习惯、活动参与状态分层设阈值。
- 自动化流程：每日或实时生成特征，执行规则，输出风险分、命中原因、证据字段和复核优先级。

### 问题 2.3：如果你要做一个账户风险评分模型，你会如何设计？

追问：

- 标签从哪里来？
- 没有标签怎么办？
- 你会选择逻辑回归、随机森林还是 XGBoost？
- 如何解释模型结果给合规团队？

考察点：

- 是否理解模型不是第一步，标签、特征、评估和解释同样重要。
- 是否能在金融风控环境中考虑可解释性。

优秀回答要点：

- 明确预测目标：未来一段时间是否被确认为可疑账户、是否发生拒付、是否触发严重合规事件。
- 标签来源：人工复核结果、合规上报记录、历史封禁、黑名单命中。
- 特征来源：账户画像、交易频率、金额分布、设备/IP、入出金路径、活动行为、历史规则命中。
- 初版可以先用逻辑回归或树模型，并保留规则基线做对照。
- 评估用时间外样本，重点看高分段 precision、召回覆盖、稳定性和人工复核量。
- 解释输出包括 top features、分箱风险率、单账户命中原因，而不是只给分数。

### 问题 2.4：市场推广活动上线后，你如何评估活动效果？

追问：

- 只看入金总额可以吗？
- 怎么区分自然增长和活动带来的增长？
- 如果活动吸引了高风险客户，你会怎么发现？

考察点：

- 是否能结合运营分析和风险分析。
- 是否理解转化率、留存、ROI 和风险质量的平衡。

优秀回答要点：

- 建立活动漏斗：曝光、点击、注册、KYC、首入金、首交易、复交易、留存。
- 指标包括转化率、入金金额、交易量、手续费收入、活动成本、ROI、客户留存。
- 用对照组、历史同期、渠道分层或差分方法估计增量效果。
- 同时看风险指标：异常交易率、AML 命中率、拒付率、短期大额出金率。
- 输出建议时不能只说活动有效，要说明哪类客户、哪个渠道、哪段生命周期表现最好。

### 问题 2.5：你接手一个误报率很高的检测规则，会如何优化？

追问：

- 你会先看规则逻辑还是先调阈值？
- 如何判断误报集中在哪些人群？
- 优化后如何避免漏报增加？

考察点：

- 是否遵守“先找根因，再改规则”的分析习惯。
- 是否能比较多种优化方案。

优秀回答要点：

- 先复盘规则目标、命中样本、人工复核结果和历史变更。
- 将命中样本按账户类型、交易金额、渠道、地区、活动、生命周期分层。
- 找出误报集中模式，例如新用户活动期、专业交易账户、高净值客户。
- 优化方式至少比较两种：分层阈值、增加排除条件、增加组合特征、改成评分排序。
- 用历史回测比较命中量、确认率、漏报风险和人工处理量。
- 上线后监控规则命中率和确认率，避免一次性调参后无人维护。

### 问题 2.6：如果让你把检测逻辑做成自动化报告，你会设计哪些内容？

追问：

- 报告给技术团队和合规团队有什么不同？
- 每日监控报告应该有什么固定字段？
- 什么情况需要报警，而不是只写进报表？

考察点：

- 是否能把分析结果产品化。
- 是否能区分监控、报告和报警。

优秀回答要点：

- 技术团队需要数据质量、任务状态、规则命中明细、异常日志和延迟。
- 合规团队需要高风险账户清单、命中原因、证据字段、趋势和处理建议。
- 固定指标包括交易量、可疑交易数、可疑率、规则命中分布、确认率、误报率、高风险账户数。
- 报警条件包括指标突增、关键任务失败、数据延迟、严重风险规则命中、单账户异常金额过大。
- 自动化报告应保留可追溯明细，方便人工复核。

## 3. 第三轮：业务场景 Case

这一轮要求你现场组织分析方案。回答时不要急着给模型，先讲目标、口径、数据、方法、验证、输出。

### Case 3.1：异常账户活动

面试官：

我们发现某个营销活动带来的新用户交易量很高，但合规团队怀疑里面有一批账户在刷活动奖励。你会怎么分析？

建议回答结构：

1. 明确业务目标：识别是否存在刷奖励账户，评估活动质量和风险。
2. 定义数据范围：活动曝光、注册、KYC、入金、交易、奖励发放、出金、设备/IP、账户关系。
3. 构造特征：
   - 注册到入金时间。
   - 入金到交易时间。
   - 交易金额是否接近奖励门槛。
   - 交易后快速出金比例。
   - 多账户共享设备、IP、银行卡或推荐人。
   - 交易品种和交易方向是否高度一致。
4. 识别方法：
   - 规则筛选高风险账户。
   - 聚类发现行为相似账户。
   - 网络关系图识别团伙特征。
   - 和非活动用户、历史活动用户做对照。
5. 输出：
   - 高风险账户清单。
   - 风险模式总结。
   - 活动规则优化建议。
   - 是否暂停奖励或增加复核。

技术主管追问：

- 如果刷子故意分散金额，怎么发现？
- 如果没有设备/IP 数据，如何替代？
- 如何避免误伤真实高活跃用户？

优秀补充：

- 用行为序列和时间间隔识别相似模式，而不是只看金额。
- 缺设备数据时用推荐关系、资金路径、交易品种、时间同步性替代。
- 对高价值老客户、新客户、活动客户分别建基线，避免同一个阈值打所有人。

### Case 3.2：AML 可疑交易监控

面试官：

请你设计一个反洗钱监控逻辑，识别“资金快速进出且交易目的不清晰”的客户。

建议回答结构：

1. 风险假设：账户将平台作为资金过桥，入金后短时间出金，实际交易行为很弱。
2. 核心指标：
   - 入金后 24 小时、72 小时出金比例。
   - 入金到出金时间间隔。
   - 交易额与入金额比例。
   - 交易手续费与资金流转金额比例。
   - 资金来源和去向是否频繁变化。
3. 规则逻辑：
   - 短时间大额入金后出金。
   - 几乎无有效交易或交易高度形式化。
   - 多账户存在相同收款或付款信息。
4. 验证方式：
   - 回看历史已确认 AML 案例。
   - 抽样人工复核。
   - 监控误报集中人群。
5. 交付物：
   - 每日高风险账户列表。
   - 单账户证据摘要。
   - 规则命中原因和建议动作。

技术主管追问：

- 如何区分正常客户撤资和资金过桥？
- 如果市场剧烈波动导致大量用户出金，规则会不会误报？
- 你会如何向合规人员解释这个规则？

优秀补充：

- 加入账户历史行为、市场波动、客户生命周期和交易目的强度。
- 高风险不是单一动作，而是多个弱信号组合。
- 对合规团队输出可解释证据，而不是只输出风险分。

### Case 3.3：规则优化与模型替代

面试官：

目前公司有 20 条人工规则，每天命中 5000 个账户，但合规团队只能复核 500 个。你会怎么优化这个系统？

建议回答结构：

1. 先做规则画像：每条规则命中量、确认率、严重等级、重复命中、历史趋势。
2. 去重和排序：同一账户多规则命中时合并证据，按风险优先级排序。
3. 分层策略：
   - 高严重规则直接进入复核。
   - 中等风险用评分排序。
   - 低确认率规则进入观察或重写。
4. 模型方案：
   - 用历史复核结果训练账户风险评分。
   - 模型作为排序层，而不是立刻替代所有规则。
   - 保留规则命中原因作为解释字段。
5. 验证方式：
   - 比较 top 500 的确认率。
   - 回测是否漏掉历史严重案例。
   - 监控不同人群的稳定性。

技术主管追问：

- 如果历史复核结果本身有偏差，怎么办？
- 如何处理新型风险模式？
- 模型上线后如何监控？

优秀补充：

- 历史标签会受人工规则和复核偏好影响，需要抽样探索未命中人群。
- 规则负责强解释和已知模式，模型负责排序和弱信号组合。
- 上线后监控分数分布、top 档确认率、规则贡献、数据漂移和人工反馈。

## 4. Python 实战追问

### 问题 4.1：给你一张交易表，你如何生成账户级特征？

表字段：

```text
account_id, order_id, trade_time, side, symbol, amount, fee, status
```

你应该能说出的特征：

- 近 1 天、7 天、30 天交易次数。
- 总交易金额、平均交易金额、最大单笔交易金额。
- 买卖方向比例。
- 交易品种数量。
- 手续费率。
- 成功交易率。
- 交易时间集中度。
- 当前行为相对历史中位数的偏离。

参考代码思路：

```python
successful = trades[trades["status"] == "success"].copy()
successful["trade_time"] = pd.to_datetime(successful["trade_time"])

features = (
    successful.groupby("account_id")
    .agg(
        trade_count=("order_id", "nunique"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        symbol_count=("symbol", "nunique"),
        total_fee=("fee", "sum"),
    )
    .reset_index()
)

features["fee_rate"] = features["total_fee"] / features["total_amount"]
```

追问：

- 如果 `total_amount` 为 0 怎么办？
- 如果同一订单有多条成交记录，`order_id` 能不能直接 count？
- 如何保证 rolling 特征不使用未来数据？

优秀回答要点：

- 明确一行数据的粒度，必要时先聚合到订单级。
- 对除零和缺失值做有业务含义的处理，而不是静默填 0。
- 时间窗口特征必须按账户和时间排序，只使用当前时点之前的数据。

### 问题 4.2：如果模型训练集准确率很高，线上效果很差，你会排查什么？

考察点：

- 是否理解过拟合、数据泄露、样本偏差和分布漂移。

优秀回答要点：

- 检查是否使用了未来字段，例如人工审核结果、交易完成后的状态。
- 检查训练集和线上样本是否来自同一分布。
- 检查标签是否延迟确认，线上时点是否不可见。
- 检查特征加工逻辑是否训练和线上一致。
- 使用时间切分验证，而不是随机切分。
- 监控线上分数分布、命中率、确认率和关键特征漂移。

## 5. SQL 实战追问

### 问题 5.1：找出每个账户最近 7 天交易金额超过自身过去 30 天日均 3 倍的账户。

回答时不用强行写完所有 SQL，但要讲清楚逻辑：

1. 先按账户和日期聚合每日交易金额。
2. 用窗口函数计算过去 30 天日均金额。
3. 计算最近 7 天累计金额。
4. 筛选最近 7 天金额超过历史基线 3 倍的账户。

参考 SQL 结构：

```sql
WITH daily_amount AS (
    SELECT
        account_id,
        CAST(trade_time AS DATE) AS trade_date,
        SUM(amount) AS daily_amount
    FROM trades
    WHERE trade_status = 'success'
    GROUP BY account_id, CAST(trade_time AS DATE)
),
baseline AS (
    SELECT
        account_id,
        trade_date,
        daily_amount,
        AVG(daily_amount) OVER (
            PARTITION BY account_id
            ORDER BY trade_date
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS avg_30d_before
    FROM daily_amount
),
recent AS (
    SELECT
        account_id,
        SUM(daily_amount) AS amount_7d,
        AVG(avg_30d_before) AS baseline_30d
    FROM baseline
    WHERE trade_date >= CURRENT_DATE - INTERVAL '7 day'
    GROUP BY account_id
)
SELECT *
FROM recent
WHERE amount_7d > baseline_30d * 7 * 3;
```

追问：

- 为什么窗口里用 `1 PRECEDING`？
- 如果账户历史不足 30 天怎么办？
- 如果某账户过去 30 天几乎没有交易，阈值会不会过低？

优秀回答要点：

- `1 PRECEDING` 是为了避免把当天交易放入历史基线。
- 历史不足时要单独标记新账户，不要和成熟账户使用同一阈值。
- 对低基线账户可以加最小金额门槛，避免小数值放大导致误报。

## 6. 沟通表达题

### 问题 6.1：你发现某个活动 ROI 很高，但风险账户比例也很高，你怎么向运营团队说明？

考察点：

- 是否能清晰、实用地解释技术分析结果。
- 是否能避免用技术语言压倒业务团队。

优秀回答结构：

```text
结论：活动带来了明显新增入金和交易，但高风险账户占比高于历史活动。
证据：主要风险集中在某渠道、注册后 24 小时内快速交易并出金的人群。
影响：如果不调整，短期 ROI 可能被风险损失和合规成本抵消。
建议：保留低风险渠道投放，对高风险渠道增加奖励延迟、KYC 复核或交易门槛。
下一步：上线监控看板，连续观察确认率、出金率和留存。
```

危险回答信号：

- 只说“模型显示这个活动风险高”，不解释原因。
- 只给技术指标，不给业务动作。
- 不区分渠道、人群和时间阶段。

## 7. 反向提问

面试最后你可以问这些问题，显示你理解岗位：

- 目前风控和合规监测主要依赖规则、模型，还是人工复核？
- 交易、订单、流水和客户行为数据是否已经有统一数据仓库？
- 现在最核心的痛点是误报率高、漏报严重、报表自动化不足，还是业务活动评估不清？
- 分析师产出的结果通常如何进入业务流程，是看板、规则配置、模型接口，还是人工报告？
- 这个岗位第一阶段最希望解决的具体业务问题是什么？

不要问得太空，例如“公司发展怎么样”。要围绕数据、流程、风控和业务落地提问。

## 8. 面试前自测清单

### 必须能讲清楚

- 一个完整项目：业务问题、数据、方法、结果、复盘。
- 风控规则如何设计、验证、上线和监控。
- 误报率高时如何定位根因。
- 活动分析如何同时看增长和风险。
- AML 场景下如何构造可解释的监控逻辑。
- SQL 中筛选、聚合、JOIN、窗口函数的常用写法。
- pandas 中 groupby、merge、rolling 的常见错误。

### 必须避免

- 一上来就说“我会用机器学习模型”，但没有业务口径。
- 只讲准确率，不讲 precision、recall、误报、漏报和人工复核成本。
- 把异常值当作脏数据直接删除。
- 使用随机切分验证时间序列风控模型。
- 讲不清字段在业务时点是否可见。
- 给非技术团队解释时只说模型术语。

### 练习方式

每道题按下面顺序回答：

```text
目标是什么？
数据有哪些？
口径怎么定？
先做哪些检查？
用什么规则或模型？
怎么验证？
怎么输出给业务？
上线后怎么监控？
```

如果你能在 2 到 3 分钟内把这个框架讲顺，基本能覆盖这类初级量化分析师岗位的核心面试要求。

## 9. 进阶技术主管追问题

这一部分更接近技术主管的深挖面。回答时要避免只背概念，要主动说清业务时点、数据口径、验证方法和上线风险。

### 问题 9.1：什么是数据泄露？请结合风控模型举例。

追问：

- 哪些字段最容易造成数据泄露？
- 为什么随机切分在风控场景里危险？
- 如果模型离线 AUC 很高，你如何判断是否泄露？

考察点：

- 是否理解“建模时可见”和“业务决策时可见”的区别。
- 是否能识别标签后信息、人工审核后信息、未来窗口统计等泄露来源。

优秀回答要点：

- 数据泄露是训练阶段使用了预测时点不可见的信息。
- 典型字段包括审核结果、最终订单状态、事后封禁标记、未来 7 天交易总额、活动结束后的奖励状态。
- 风控模型应按时间切分训练集、验证集、测试集，并严格定义特征截止时间。
- 排查方式包括看特征重要性、检查字段产生时间、做时间外验证、删除可疑字段重训对比。

### 问题 9.2：如果可疑账户样本只有 1%，你如何建模和评估？

追问：

- 准确率还能不能用？
- 你会怎么处理类别不平衡？
- 阈值怎么选？

考察点：

- 是否理解极度不平衡分类问题。
- 是否能把模型评估和人工复核产能联系起来。

优秀回答要点：

- 不能主要看 accuracy，因为全预测正常也可能有 99% 准确率。
- 重点看 precision、recall、PR-AUC、top K 命中率和高分段确认率。
- 可以使用 class weight、下采样、上采样、分层抽样，但要保持时间外测试集真实分布。
- 阈值不只由模型决定，还要看每天能复核多少账户、漏报成本和误报成本。

### 问题 9.3：规则和模型分别适合解决什么问题？

追问：

- 为什么合规团队通常不愿意只看一个模型分数？
- 什么情况你会坚持先做规则而不是模型？
- 规则系统如何避免越写越乱？

考察点：

- 是否理解金融风控里的可解释性、稳定性和治理要求。

优秀回答要点：

- 规则适合已知、高风险、强解释的模式，例如短时间大额出金、多账户共享实名信息。
- 模型适合组合大量弱信号，做风险排序和优先级分配。
- 初期没有可靠标签时，应先做规则、分析和人工复核闭环。
- 规则要有版本、命中原因、负责人、上线日期、回测结果和定期复盘机制。

### 问题 9.4：如何判断一个市场推广活动是否真的带来了增量？

追问：

- 活动期间注册和入金都上涨，是否能说明活动有效？
- 没有随机实验时怎么办？
- 如何处理渠道质量差异？

考察点：

- 是否理解相关性和因果增量的区别。
- 是否能设计对照组或准实验分析。

优秀回答要点：

- 不能只看活动期间总量上涨，要区分自然增长、市场行情、渠道投放和活动激励的影响。
- 优先使用 A/B test；没有实验时可用历史同期、相似渠道、未参加活动人群或差分分析做对照。
- 指标要分层看：注册、KYC、首入金、首交易、复交易、留存、风险命中率、净收入。
- 增量效果要和风险成本、奖励成本、用户质量一起评估。

### 问题 9.5：你如何设计一个交易异常监控看板？

追问：

- 首页放哪些核心指标？
- 哪些指标需要按小时看，哪些按天看？
- 如何让业务方能直接行动？

考察点：

- 是否能把分析交付成可运营的监控产品。

优秀回答要点：

- 首页指标：总交易量、可疑交易数、可疑率、高风险账户数、规则命中量、人工确认率、数据延迟。
- 分布维度：渠道、账户类型、地区、币种、交易方向、金额分层、规则类型。
- 趋势图用于发现突增，明细表用于复核，证据字段用于行动。
- 看板要支持下钻到账户、订单和命中规则，不只展示汇总图。

### 问题 9.6：如何处理模型上线后的数据漂移？

追问：

- 你会监控哪些漂移指标？
- 如果特征分布变了，但确认率没变，要不要处理？
- 如果确认率下降，你如何定位原因？

考察点：

- 是否理解模型不是训练完就结束。

优秀回答要点：

- 监控输入特征分布、缺失率、分数分布、top K 命中率、确认率、规则命中分布。
- 可用 PSI、KS、分位数变化、均值/方差变化来监控漂移。
- 分布变动不一定立即重训，但要定位是业务变化、数据质量问题还是用户结构变化。
- 确认率下降时检查数据链路、标签延迟、规则变化、渠道变化、人工复核口径变化。

## 10. Python 现场编程题

这一部分建议你真的动手写。面试中不一定要求代码完全可运行，但逻辑、边界和时间口径必须清楚。

### 编程题 10.1：生成账户近 7 天滚动交易特征

题目：

给定交易表 `trades`：

```text
account_id, order_id, trade_time, amount, status
```

请用 pandas 生成每个账户每天的以下特征：

- 当日成功交易金额。
- 过去 7 天成功交易金额，不包含当天。
- 过去 7 天成功交易次数，不包含当天。
- 当日金额是否超过过去 7 天日均金额的 3 倍。

参考答案：

```python
import pandas as pd

def build_rolling_features(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df = df[df["status"].eq("success")]
    df["trade_date"] = df["trade_time"].dt.date

    daily = (
        df.groupby(["account_id", "trade_date"])
        .agg(
            daily_amount=("amount", "sum"),
            daily_count=("order_id", "nunique"),
        )
        .reset_index()
    )

    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.sort_values(["account_id", "trade_date"])

    daily["amount_7d_before"] = (
        daily.groupby("account_id")["daily_amount"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=3).sum())
    )
    daily["count_7d_before"] = (
        daily.groupby("account_id")["daily_count"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=3).sum())
    )
    daily["avg_amount_7d_before"] = daily["amount_7d_before"] / 7
    daily["is_amount_spike"] = (
        daily["daily_amount"] > daily["avg_amount_7d_before"] * 3
    )
    return daily
```

面试官追问：

- 为什么要 `shift(1)`？
- 如果某账户中间几天没有交易，这段代码是否等价于自然日滚动？
- `min_periods=3` 的业务含义是什么？

优秀回答要点：

- `shift(1)` 避免把当天金额放入历史基线，防止泄露。
- 这段代码按交易日记录滚动，不是完整自然日滚动；如果必须按自然日，要先补齐账户日期面板。
- `min_periods` 是历史长度要求，新账户应单独处理。

### 编程题 10.2：识别“频繁小额入金后快速大额出金”的账户

题目：

给定资金流水表 `cashflows`：

```text
account_id, flow_time, flow_type, amount
```

其中 `flow_type` 为 `deposit` 或 `withdraw`。请找出过去 24 小时内小额入金次数不少于 5 次，且随后 6 小时内发生大额出金的账户。

参考答案：

```python
import pandas as pd

def detect_deposit_withdraw_pattern(
    cashflows: pd.DataFrame,
    small_deposit_limit: float = 1000,
    large_withdraw_limit: float = 10000,
) -> pd.DataFrame:
    df = cashflows.copy()
    df["flow_time"] = pd.to_datetime(df["flow_time"])
    df = df.sort_values(["account_id", "flow_time"])

    deposits = df[
        df["flow_type"].eq("deposit") & df["amount"].le(small_deposit_limit)
    ].copy()
    withdraws = df[
        df["flow_type"].eq("withdraw") & df["amount"].ge(large_withdraw_limit)
    ].copy()

    alerts = []
    for account_id, w in withdraws.groupby("account_id"):
        account_deposits = deposits[deposits["account_id"].eq(account_id)]
        for _, row in w.iterrows():
            start = row["flow_time"] - pd.Timedelta(hours=24)
            end = row["flow_time"]
            deposit_count = account_deposits[
                account_deposits["flow_time"].between(start, end, inclusive="left")
            ].shape[0]
            if deposit_count >= 5:
                alerts.append(
                    {
                        "account_id": account_id,
                        "withdraw_time": row["flow_time"],
                        "withdraw_amount": row["amount"],
                        "small_deposit_count_24h": deposit_count,
                    }
                )

    return pd.DataFrame(alerts)
```

面试官追问：

- 这段代码性能有什么问题？
- 如果数据量很大，你会怎么优化？
- 这个规则为什么可能误报？

优秀回答要点：

- 循环写法直观但大数据下性能差，可以用窗口聚合、SQL 预计算或流式特征表优化。
- 需要按账户、时间建立索引或在数据仓库侧先生成滚动特征。
- 误报可能来自真实用户拆分入金、支付限额、活动奖励门槛等，需要结合账户等级和历史行为分层。

### 编程题 10.3：计算规则命中后的 precision、recall 和人工复核量

题目：

给定规则命中结果 `alerts` 和人工复核标签 `labels`：

```text
alerts: account_id, alert_date, rule_id
labels: account_id, label_date, is_suspicious
```

请计算某条规则在指定时间段内的 precision 和复核账户数。

参考答案：

```python
def evaluate_rule(alerts: pd.DataFrame, labels: pd.DataFrame, rule_id: str) -> dict:
    rule_alerts = alerts[alerts["rule_id"].eq(rule_id)].copy()
    alerted_accounts = rule_alerts[["account_id"]].drop_duplicates()

    latest_labels = (
        labels.sort_values("label_date")
        .drop_duplicates("account_id", keep="last")
        [["account_id", "is_suspicious"]]
    )

    evaluated = alerted_accounts.merge(latest_labels, on="account_id", how="left")
    reviewed = evaluated[evaluated["is_suspicious"].notna()]

    review_count = len(reviewed)
    suspicious_count = reviewed["is_suspicious"].sum()
    precision = suspicious_count / review_count if review_count else None

    return {
        "rule_id": rule_id,
        "alerted_accounts": len(alerted_accounts),
        "reviewed_accounts": review_count,
        "confirmed_suspicious": int(suspicious_count),
        "precision": precision,
    }
```

面试官追问：

- 这段代码为什么没有算 recall？
- 标签缺失是负样本吗？
- 如果一个账户多次命中规则，应该怎么算？

优秀回答要点：

- recall 需要知道所有真实可疑账户，而不仅是被规则命中的账户。
- 标签缺失不等于正常，可能只是没有复核。
- 账户级、告警级、规则级口径要分开，否则会高估或低估效果。

### 编程题 10.4：计算活动漏斗和风险质量

题目：

给定活动用户表 `events`：

```text
user_id, channel, event_time, event_name
```

`event_name` 包括 `register`、`kyc_pass`、`deposit`、`trade`、`withdraw`、`risk_alert`。请按渠道计算注册到 KYC、KYC 到入金、入金到交易、交易后风险命中率。

参考答案思路：

```python
def campaign_funnel(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    user_events = (
        df.assign(value=1)
        .pivot_table(
            index=["channel", "user_id"],
            columns="event_name",
            values="value",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    result = (
        user_events.groupby("channel")
        .agg(
            registers=("register", "sum"),
            kyc_pass=("kyc_pass", "sum"),
            deposits=("deposit", "sum"),
            trades=("trade", "sum"),
            risk_alerts=("risk_alert", "sum"),
        )
        .reset_index()
    )

    result["kyc_rate"] = result["kyc_pass"] / result["registers"]
    result["deposit_rate"] = result["deposits"] / result["kyc_pass"]
    result["trade_rate"] = result["trades"] / result["deposits"]
    result["risk_alert_rate"] = result["risk_alerts"] / result["trades"]
    return result
```

面试官追问：

- 如果一个用户先交易后 KYC，说明什么问题？
- 只看是否发生事件够不够？
- 活动分析为什么要看风险命中率？

优秀回答要点：

- 需要检查事件顺序和业务流程是否一致，异常顺序可能是埋点或数据同步问题。
- 还要看时间间隔、金额、留存、ROI 和奖励成本。
- 高转化渠道如果风险命中率也高，可能不是优质增长。

### 编程题 10.5：用 IsolationForest 做异常检测时，你会怎么防止误用？

题目：

你有账户级特征表：

```text
account_id, trade_count_7d, amount_7d, withdraw_ratio_7d, symbol_count_7d, account_age_days
```

请说明如何训练一个无监督异常检测模型，并指出它不能直接解决什么问题。

参考代码骨架：

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

feature_cols = [
    "trade_count_7d",
    "amount_7d",
    "withdraw_ratio_7d",
    "symbol_count_7d",
    "account_age_days",
]

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(contamination=0.02, random_state=42)),
    ]
)

model.fit(train_features[feature_cols])
scores = -model.decision_function(score_features[feature_cols])
```

面试官追问：

- `contamination=0.02` 是什么意思？
- 无监督异常分数能不能等于风险概率？
- 如何验证这个模型有业务价值？

优秀回答要点：

- `contamination` 是预期异常比例，会影响阈值，不是自然真相。
- 异常不等于违规，高异常可能只是高净值或专业交易账户。
- 必须结合人工复核、历史案例、分层抽样和规则对照验证。

## 11. SQL 现场编程题

SQL 面试题重点不是背语法，而是看你能不能把业务口径拆成明细过滤、聚合、窗口和结果解释。

### SQL 题 11.1：找出 24 小时内入金后快速出金的账户

表结构：

```text
cashflows(account_id, flow_time, flow_type, amount)
```

题目：

找出任意一笔入金后 24 小时内出金金额超过该笔入金 80% 的账户。

参考答案：

```sql
WITH deposits AS (
    SELECT
        account_id,
        flow_time AS deposit_time,
        amount AS deposit_amount
    FROM cashflows
    WHERE flow_type = 'deposit'
),
withdraws AS (
    SELECT
        account_id,
        flow_time AS withdraw_time,
        amount AS withdraw_amount
    FROM cashflows
    WHERE flow_type = 'withdraw'
)
SELECT
    d.account_id,
    d.deposit_time,
    d.deposit_amount,
    MIN(w.withdraw_time) AS first_withdraw_time,
    SUM(w.withdraw_amount) AS withdraw_amount_24h
FROM deposits d
JOIN withdraws w
  ON d.account_id = w.account_id
 AND w.withdraw_time > d.deposit_time
 AND w.withdraw_time <= d.deposit_time + INTERVAL '24 hour'
GROUP BY d.account_id, d.deposit_time, d.deposit_amount
HAVING SUM(w.withdraw_amount) >= d.deposit_amount * 0.8;
```

追问：

- 如果同一笔出金被多个入金窗口重复匹配怎么办？
- 为什么这里用 `>` 而不是 `>=`？
- 这个规则会命中哪些正常客户？

优秀回答要点：

- 复杂资金归因需要 FIFO、账户余额路径或按时间窗口去重。
- 时间边界要符合业务定义，避免同一时间戳重复归因。
- 正常撤资、活动套利、支付失败回退都可能命中，需要结合交易行为判断。

### SQL 题 11.2：每个账户最近一笔成功订单

表结构：

```text
orders(account_id, order_id, order_time, status, amount)
```

参考答案：

```sql
WITH ranked AS (
    SELECT
        account_id,
        order_id,
        order_time,
        amount,
        ROW_NUMBER() OVER (
            PARTITION BY account_id
            ORDER BY order_time DESC, order_id DESC
        ) AS rn
    FROM orders
    WHERE status = 'success'
)
SELECT
    account_id,
    order_id,
    order_time,
    amount
FROM ranked
WHERE rn = 1;
```

追问：

- 如果同一账户同一时间有两笔订单，为什么要加 `order_id DESC`？
- `ROW_NUMBER`、`RANK`、`DENSE_RANK` 有什么区别？
- 最近订单按订单创建时间还是成交时间？

优秀回答要点：

- 排序必须稳定，否则每次结果可能不同。
- 时间字段选择要和业务问题一致，风控通常更关心实际成交或资金变动时间。

### SQL 题 11.3：计算规则的每日命中率和确认率

表结构：

```text
alerts(alert_id, account_id, alert_time, rule_id)
reviews(alert_id, review_time, is_suspicious)
trades(account_id, trade_time, order_id)
```

题目：

按天、按规则计算：

- 命中账户数。
- 当日有交易账户数。
- 命中率。
- 已复核告警数。
- 确认率。

参考答案：

```sql
WITH daily_alerts AS (
    SELECT
        CAST(alert_time AS DATE) AS alert_date,
        rule_id,
        COUNT(DISTINCT account_id) AS alerted_accounts,
        COUNT(DISTINCT alert_id) AS alert_count
    FROM alerts
    GROUP BY CAST(alert_time AS DATE), rule_id
),
daily_traders AS (
    SELECT
        CAST(trade_time AS DATE) AS trade_date,
        COUNT(DISTINCT account_id) AS trading_accounts
    FROM trades
    GROUP BY CAST(trade_time AS DATE)
),
daily_reviews AS (
    SELECT
        CAST(a.alert_time AS DATE) AS alert_date,
        a.rule_id,
        COUNT(DISTINCT r.alert_id) AS reviewed_alerts,
        SUM(CASE WHEN r.is_suspicious = 1 THEN 1 ELSE 0 END) AS confirmed_alerts
    FROM alerts a
    LEFT JOIN reviews r
      ON a.alert_id = r.alert_id
    GROUP BY CAST(a.alert_time AS DATE), a.rule_id
)
SELECT
    a.alert_date,
    a.rule_id,
    a.alerted_accounts,
    t.trading_accounts,
    a.alerted_accounts * 1.0 / NULLIF(t.trading_accounts, 0) AS alert_rate,
    r.reviewed_alerts,
    r.confirmed_alerts * 1.0 / NULLIF(r.reviewed_alerts, 0) AS confirmation_rate
FROM daily_alerts a
LEFT JOIN daily_traders t
  ON a.alert_date = t.trade_date
LEFT JOIN daily_reviews r
  ON a.alert_date = r.alert_date
 AND a.rule_id = r.rule_id;
```

追问：

- 命中率分母用交易账户数合理吗？
- 确认率按告警算还是按账户算？
- 未复核告警能不能当作正常？

优秀回答要点：

- 分母要和业务目标一致，也可以用活跃账户数、入金账户数或全部账户数。
- 告警级和账户级都要看，但含义不同。
- 未复核不是负样本，直接当正常会低估风险。

### SQL 题 11.4：识别共享设备的多账户团伙

表结构：

```text
logins(account_id, device_id, ip, login_time)
```

题目：

找出 7 天内同一 `device_id` 登录过 5 个及以上账户的设备，并列出账户数、登录次数、首次和最近登录时间。

参考答案：

```sql
SELECT
    device_id,
    COUNT(DISTINCT account_id) AS account_count,
    COUNT(*) AS login_count,
    MIN(login_time) AS first_login_time,
    MAX(login_time) AS last_login_time
FROM logins
WHERE login_time >= CURRENT_TIMESTAMP - INTERVAL '7 day'
  AND device_id IS NOT NULL
GROUP BY device_id
HAVING COUNT(DISTINCT account_id) >= 5
ORDER BY account_count DESC, login_count DESC;
```

追问：

- 公司公共电脑或网吧会不会造成误报？
- IP 和 device_id 哪个更可靠？
- 如何进一步判断是不是团伙？

优秀回答要点：

- 共享设备只是风险信号，不是定罪证据。
- IP 可能动态变化或被 NAT 共享，device_id 也可能被伪造。
- 要结合注册时间、推荐关系、资金路径、交易品种和行为同步性。

### SQL 题 11.5：活动渠道的转化和风险质量

表结构：

```text
users(user_id, channel, register_time)
kyc(user_id, pass_time)
deposits(user_id, deposit_time, amount)
trades(user_id, trade_time, amount)
alerts(user_id, alert_time, risk_level)
```

题目：

按渠道统计注册用户数、KYC 通过人数、入金人数、交易人数、高风险告警人数，以及各阶段转化率。

参考答案：

```sql
WITH user_flags AS (
    SELECT
        u.user_id,
        u.channel,
        CASE WHEN k.user_id IS NOT NULL THEN 1 ELSE 0 END AS has_kyc,
        CASE WHEN d.user_id IS NOT NULL THEN 1 ELSE 0 END AS has_deposit,
        CASE WHEN t.user_id IS NOT NULL THEN 1 ELSE 0 END AS has_trade,
        CASE WHEN a.user_id IS NOT NULL THEN 1 ELSE 0 END AS has_high_risk
    FROM users u
    LEFT JOIN (SELECT DISTINCT user_id FROM kyc) k
      ON u.user_id = k.user_id
    LEFT JOIN (SELECT DISTINCT user_id FROM deposits) d
      ON u.user_id = d.user_id
    LEFT JOIN (SELECT DISTINCT user_id FROM trades) t
      ON u.user_id = t.user_id
    LEFT JOIN (
        SELECT DISTINCT user_id
        FROM alerts
        WHERE risk_level = 'high'
    ) a
      ON u.user_id = a.user_id
)
SELECT
    channel,
    COUNT(*) AS registered_users,
    SUM(has_kyc) AS kyc_users,
    SUM(has_deposit) AS deposit_users,
    SUM(has_trade) AS trade_users,
    SUM(has_high_risk) AS high_risk_users,
    SUM(has_kyc) * 1.0 / NULLIF(COUNT(*), 0) AS kyc_rate,
    SUM(has_deposit) * 1.0 / NULLIF(SUM(has_kyc), 0) AS deposit_rate_after_kyc,
    SUM(has_trade) * 1.0 / NULLIF(SUM(has_deposit), 0) AS trade_rate_after_deposit,
    SUM(has_high_risk) * 1.0 / NULLIF(SUM(has_trade), 0) AS high_risk_rate_after_trade
FROM user_flags
GROUP BY channel
ORDER BY registered_users DESC;
```

追问：

- 为什么子查询里先 `SELECT DISTINCT user_id`？
- 这个 SQL 有没有时间窗口问题？
- 活动分析为什么不能只看注册到入金转化？

优秀回答要点：

- 先去重避免一对多 JOIN 把用户数放大。
- 应限制活动周期和注册后观察窗口，例如注册后 7 天内是否入金。
- 高转化但高风险的渠道可能不值得继续投放。

## 12. 专业压测题：面试官连续追问

这一部分用于训练抗压。你可以把每题当作 3 到 5 分钟口头作答。

### 压测 12.1：你说要用模型降低误报率，具体怎么证明降低了？

合格回答：

- 先定义误报率口径：已复核告警中最终判定为非风险的比例。
- 用历史时间外样本比较原规则排序和模型排序。
- 在相同人工复核量下比较确认率和严重案例覆盖率。
- 检查是否遗漏历史严重风险账户。
- 上线后做灰度：一部分按旧规则，一部分按模型排序，比较复核效率。

危险回答：

- “模型准确率更高，所以误报率会下降。”
- “我会调阈值，让命中数量变少。”

### 压测 12.2：你做的异常检测没有标签，业务方凭什么相信？

合格回答：

- 无标签异常检测只能产生候选风险，不等于最终风险判断。
- 我会做三类验证：历史案例重合、人工抽样复核、和规则系统对照。
- 输出要有可解释证据，例如金额突增、快速出金、多账户共享设备。
- 如果人工确认率长期低，就要调整特征、阈值或场景定义。

危险回答：

- “IsolationForest 算出来是异常，所以就是风险。”
- “没有标签也没关系，无监督模型可以自动发现问题。”

### 压测 12.3：如果业务方要求把所有高风险用户都拦截，你会怎么回应？

合格回答：

- 先确认“高风险”的判定来源和置信度。
- 区分拦截、延迟、人工复核、限额、二次验证等不同动作。
- 对强规则命中可以直接拦截，对弱信号高分账户更适合进入复核或限额。
- 需要评估误伤成本、合规要求、客户体验和风险损失。

危险回答：

- “高风险就全部封禁。”
- “为了用户体验，最好都不要拦。”

### 压测 12.4：如果 SQL 结果和 Python 结果不一致，你怎么查？

合格回答：

- 先确认数据源、时间范围、时区、状态过滤、去重口径是否一致。
- 抽取少量账户做明细级对账。
- 检查 JOIN 是否造成行数膨胀。
- 检查 Python 是否有缺失值填充、类型转换或排序窗口问题。
- 固化一组测试样本，让 SQL 和 Python 都跑出同样结果。

危险回答：

- “应该是数据延迟。”
- “我会重新跑一遍看看。”

### 压测 12.5：如何把一次分析变成长期可用的工作流？

合格回答：

- 把一次性 SQL 和 notebook 拆成数据抽取、特征生成、检测逻辑、结果输出、监控报警。
- 明确输入表、字段口径、刷新频率、负责人和失败处理。
- 输出固定格式的高风险账户清单和证据字段。
- 建立回测、复核反馈和规则版本管理。
- 定期检查误报率、漏报线索、数据质量和业务变化。

危险回答：

- “我会写成一个脚本每天跑。”
- “放到报表里就可以了。”
