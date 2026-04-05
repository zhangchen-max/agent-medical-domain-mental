

更新
你看一下这个代码，整体的架构生成一下，标注用中文
1








你看一下这个代码，整体的架构生成一下，标注用中文


上 4 条消息
这是一个“单进程 CLI + LangGraph 编排 + 多模型协作”的 Python 项目，不是前后端分离系统，也不是微服务。主流程是：患者输入 -> 输入安全检查 -> 阶段评估 -> 3 个 agent 并发给意见 -> 按风险分流 -> 生成对患者的话术 -> 输出安全收口。


否
是
患者输入
入口层CLI 会话循环
会话状态DebateState
模型中心DomesticModelHub
LangGraph 工作流
输入安全节点input_safety
阶段评估节点stage_assess
辩论汇聚节点run_debate
共情 AgentQwen Plus
完整性 AgentQwen Plus
风险审查 AgentDeepSeek Reasoner
结果汇总
风险是否高/危急
问题生成节点composeQwen Max
危机分支节点crisis
输出安全节点output_safety
助手回复
分层理解

入口层：main.py (line 39) 负责加载环境变量、初始化 DomesticModelHub、构建图、维护会话状态、循环收发消息。
编排层：graph.py (line 10) 定义固定工作流：input_safety -> stage_assess -> run_debate -> compose/crisis -> output_safety。
业务节点层：nodes.py (line 59) 承担真正的业务逻辑，包括阶段门控、风险关键词兜底、危机分支和最终话术处理。
模型适配层：models.py (line 31) 统一封装 Qwen/DeepSeek 调用，要求模型严格返回 JSON，并用 Pydantic 做校验和重试。
契约层：schema.py (line 27) 定义阶段评估、agent 意见、风险结果、最终输出和全局状态 DebateState。
Prompt 层：prompts.py (line 39) 给 5 个角色分别定义系统提示词和输出字段约束。
核心设计点

这个项目里的“debate”本质上不是多轮 agent 互相争论，而是一次 fan-out / gather：在 nodes.py (line 98) 并发拿到 3 个 agent 的意见，再统一汇总。
阶段控制是显式状态机，阶段名定义在 schema.py (line 11)，门控逻辑在 nodes.py (line 177)。
风险控制有双保险：一层是 Risk Review Agent 的结构化判断，另一层是关键词强制抬升风险，见 nodes.py (line 51)。
最后一定经过输出安全收口，避免确定性诊断表述，见 nodes.py (line 164)。
如果你要，我下一步可以直接给你再补一版“更适合汇报/文档”的架构图，比如：

模块分层图
时序图
ARCHITECTURE.md 中文文档版

保留主要的逻辑节点就行，什么LangGraph 工作流，不要

可以，精简成“主要逻辑节点”版本如下（不提 LangGraph）：


否
是
患者输入
输入检查
阶段评估
多角色分析（共情/完整性/风险）
风险是否高危
追问生成
危机干预
输出检查
助手回复
对应中文职责：

输入检查：空输入/基础安全过滤
阶段评估：判断当前访谈阶段与下一步目标
多角色分析：并行产出共情、信息完整性、风险意见
风险分流：高危走危机干预，非高危走追问生成
输出检查：避免不当措辞后再回复用户
如果你要，我可以再给你一版“PPT 可直接贴”的超简图（只保留 5 个框）。


保留三个agent的示意图


否
是
患者输入
阶段评估
Empathy Agent（共情）
Clinical CompletenessAgent（信息完整性）
Risk ReviewAgent（风险审查）
结果汇总
风险是否高危
追问生成并回复
危机干预并回复
这版只保留了主链路和 3 个 agent。


这里项目里有对话状态的概念吗，有的话也加到这个示意图里面

有，对话状态在这个项目里是核心概念，名字是 DebateState，定义在 schema.py (line 66)。它承载了 history、stage、risk_level、next_turn_goal、3 个 agent 的意见等字段，并在每个节点被持续读写（入口初始化见 main.py (line 13)）。


否
是
患者输入
对话状态 DebateState
阶段评估
Empathy Agent（共情）
Clinical CompletenessAgent（信息完整性）
Risk ReviewAgent（风险审查）
结果汇总
风险是否高危
追问生成并回复
危机干预并回复
助手输出
如果你愿意，我可以再给你一版“状态字段分层图”（基础字段/临床字段/安全字段），方便你做设计说明。


这里的状态是哪几种

这个项目里“状态”主要有 3 层：

访谈阶段状态（stage，6 种）
定义在 schema.py (line 11) 的 StageName：
rapport（建立关系）
broad_explore（广泛探索）
structured_clarify（结构化澄清）
risk_probe（风险探查）
hypothesis_test（假设验证）
closure（收束）
风险状态（risk_level，4 种）
定义在 schema.py (line 20) 的 RiskLevel：
low
medium
high
critical
流程相位（phase，2 种）
定义在 schema.py (line 88) 的 DebateState.phase：
normal
crisis
另外还有输入/输出安全状态 SafetyCheck（blocked: true/false + reason），用于判断是否拦截当前轮，定义在 schema.py (line 61)。


好，现在把这个项目优化一下，现在问题是：回答速度太慢了，在不改变框架和回答逻辑的前提下优化一下的答题速度


上 5 条消息
已完成一轮“只提速、不改框架和回答逻辑”的优化，核心流程节点和分流条件都保持原样。

主要改动在两处：

models.py (line 49)
增加了每类调用的 max_tokens 上限和统一请求超时，减少模型无效长输出与等待时间。
新增可配环境变量：STAGE_MAX_TOKENS、EMPATHY_MAX_TOKENS、COMPLETENESS_MAX_TOKENS、RISK_MAX_TOKENS、COMPOSE_MAX_TOKENS、MODEL_TIMEOUT_SECONDS。

nodes.py (line 43)
增加 payload 瘦身：对传给模型的 history、latest_user_input、next_turn_goal、agent 意见字段做长度截断与压缩，降低上下文 token 体积。
新增可配环境变量：HISTORY_WINDOW、MAX_HISTORY_CHARS、MAX_PAYLOAD_TEXT_CHARS。

逻辑不变说明：

节点顺序、分支结构、风险门控、危机分流都没改（只是请求参数和上下文体积优化）。
3 个 agent 并发机制保持不变。
校验情况：

已通过语法编译检查：PYTHONPYCACHEPREFIX=/tmp python3 -m compileall src/psy_debate。
未做真实接口压测（本地未接入你线上 key/环境），如果你愿意我可以继续帮你加一个本地 benchmark 脚本，直接对比优化前后单轮耗时分位数。

现在我怎么在终端启动


上 1 条消息
在项目目录 /Users/zhangchen/Desktop/1 里，按这个顺序启动就可以：

cd /Users/zhangchen/Desktop/1

# 建议先建虚拟环境（可选但推荐）
python3 -m venv .venv
source .venv/bin/activate

# 安装项目
python3 -m pip install -U pip
python3 -m pip install -e .

# 确认 .env 里至少填了两个 key
# QWEN_API_KEY=...
# DEEPSEEK_API_KEY=...

# 启动
psy-debate
启动后会看到 Psych Debate Agent (LangGraph) 已启动，输入内容即可对话，输入 exit 退出。
如果出现 psy-debate: command not found，用这个兜底启动命令：

python3 -m psy_debate.main






GPT-5.3-Codex

高


本地

默认权限

创建 Git 存储库
终端
zsh

