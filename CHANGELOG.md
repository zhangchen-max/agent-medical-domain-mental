# CHANGELOG

本文件记录项目的主要变更历史，供协作者和 Agent 了解项目演进背景。

---

## [0.1.0] — 2026-03-28

### 项目初始化

**背景**  
构建一个面向精神科初筛分诊的多智能体 Debate 系统，目标是让患者更愿意开口、症状描述更结构化、风险点更容易暴露，供正式问诊前信息采集使用。

**架构**  
使用 LangGraph 实现，包含 5 个角色节点：
- `Empathy Agent`（共情）
- `Clinical Completeness Agent`（临床完整性检查）
- `Risk Review Agent`（风险审查）
- `Question Composer Agent`（最终问题生成）
- `Stage Assessor Agent`（阶段评估）

全部节点当前配置为本地 `qwen3-32B-FP16`（通过 vLLM 服务暴露 OpenAI 兼容接口）。

**已实现功能**
- LangGraph 多节点 Debate 骨架（并发 fanout → 仲裁 → 路由）
- 六阶段状态机：`rapport → broad_explore → structured_clarify → risk_probe → hypothesis_test → closure`
- 硬门槛逻辑：high/critical 风险强制进入 crisis 分支；hypothesis_test 前验证槽位覆盖率和时间线置信度
- PHQ-9 / GAD-7 量表动态估分与条目追踪
- 社会支持评估模块
- 一致性检查（对话矛盾检测）
- 会话数据库持久化（SQLite/MySQL via SQLAlchemy）
- 输入/输出安全过滤节点
- CLI 演示入口（`psy-debate` 命令）

**对比测试（2026-03-16）**  
在 10 轮对话测试中对比了多节点系统与云端单模型 qwen-max：
- 多节点系统平均耗时：46.93s/轮（本地 qwen3-32B-FP16，需 65GB 显存）
- 云端 Qwen 平均耗时：4.24s/轮
- 多节点系统在共情深度、风险追问、临床完整性上表现更优

**会话测试（2026-03-28）**  
完整 11 轮模拟问诊（学生抑郁场景），关键结果：
- 总用时：211.1s（平均 19.2s/轮）
- PHQ-9 最终估分：14/27，GAD-7：8/21
- 最终阶段：`structured_clarify`，风险等级：medium
- 第 9 轮检出一致性矛盾（历史病史对比）

**文件结构**
```
src/psy_debate/
├── schema.py      # 状态与结构化输出 schema
├── prompts.py     # 5 个 agent 的系统提示词
├── models.py      # 模型路由与 JSON 输出解析
├── nodes.py       # LangGraph 节点逻辑和风险路由
├── graph.py       # 图编排
├── db.py          # 会话数据库持久化
└── main.py        # CLI 入口
scripts/
├── start_vllm.sh  # 启动本地 vLLM 服务
├── run.sh         # 运行 agent
├── setup_conda_env.sh
└── compare_multi_vs_single.py   # 多节点 vs 单模型对比脚本
```

---

## 后续计划（待开发）

- [ ] 接入云端模型（API key 模式），降低推理延迟
- [ ] 增加量表槽位覆盖率自动提升策略
- [ ] 前端界面或微信小程序接入
- [ ] 生产环境接入平台侧内容安全服务与人工兜底流程
- [ ] 多语言支持（普通话/粤语/方言）

---

> **注意**：本系统为初筛与信息采集助手，不是诊断系统。生产环境必须配合专业医生和人工兜底流程使用。
