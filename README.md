# AgentWorkbench · 企业方案研究与交付多智能体协作系统

基于 **LangGraph 状态机**编排的多智能体工作流系统：给定一个企业业务目标，系统自动完成
**任务规划 → 证据检索 → 事实分析 → 方案撰写 → 评审修订**，支持人工审批门（HITL）、
引用溯源、护栏检查与全链路追踪，最终产出带引用依据的企业方案文档。

[![CI](https://github.com/liuhemingmeng/Enterprise-multi-agent-collaboration/actions/workflows/ci.yml/badge.svg)](https://github.com/liuhemingmeng/Enterprise-multi-agent-collaboration/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-1C3C3C)
![Tests](https://img.shields.io/badge/tests-90%20passed-brightgreen)

> 定位：可复现、可评测、可审计的 LLM 应用工程实践 —— 所有外部依赖（LLM / 检索 API）
> 均通过开关注入，**无密钥时自动回退确定性桩，测试与评测零费用、离线可复现**。

---

## ✨ 功能特性

- **多智能体编排**：Planner / Retriever / Analyst / Writer / Reviewer 五个角色，LangGraph
  `StateGraph` 显式状态机驱动，条件分支（修订回环 / 证据不足回退 / 人工审核 / 直通导出）全部声明式定义
- **引用溯源**：每条方案结论可回溯到检索证据（文档名 / 页码 / 相关度得分），评审节点逐条核查引用
- **人工审批门（HITL）**：评审通过后可暂停等待人工 `approve` / `revise`，也可在修订超限时安全转人工
- **护栏层（Guardrails）**：输入注入检测（critical 阻断）、产物 schema 校验、工具结果注入检测（记录不篡改）
- **工具安全边界**：白名单注册制 + 参数校验 + 超时守卫 + 成本预算 + 错误归档，智能体只能调用已注册工具
- **全链路可观测**：节点级 Span（耗时 / 状态 / 成本）、SSE 实时事件流、护栏审计面板
- **持久化与恢复**：SQLite 逐节点落盘，进程重启后从最后完成节点续跑（断点恢复）
- **双模式运行**：真实 API（P1 检索 + 任意 OpenAI 兼容 LLM）与确定性桩一键切换
- **内置评测**：100 条任务数据集，单 Agent vs 多 Agent 对照实验（引用覆盖率 / 自动完成率 / 成本 / 证据数）
- **零构建前端**：单文件工作台（实时节点图 / SSE 进度流 / 人工审批 / Markdown 报告导出）

## 🏗️ 系统架构

```mermaid
flowchart LR
    U[用户目标] --> P[Planner<br/>任务拆解]
    P --> R[Retriever<br/>kb_search 工具]
    R --> A[Analyst<br/>事实/风险抽取]
    A --> W[Writer<br/>方案撰写]
    W --> V[Reviewer<br/>引用核查]

    V -->|approved| H{审批门<br/>HITL}
    V -->|revise| RJ[Revise 回环<br/>≤3 次] --> W
    V -->|insufficient| F[证据不足回退] --> R
    V -->|human| Q[人工队列]

    H -->|approve| E[Export 导出]
    H -->|revise| RJ
    Q --> E

    subgraph 基础设施
        R -.白名单/预算/超时.-> T[ToolRegistry]
        T --> EXT[(P1 检索 API<br/>或确定性桩)]
    end
    subgraph 横切层
        G[Guardrails 护栏]
        TR[Tracing 节点级 Span]
        PS[(SQLite 持久化<br/>逐节点落盘)]
    end
```

每个节点执行都被 `guarded()` 装饰器包裹：先记录 Span（耗时/状态/成本），再跑节点级护栏
（detection-only），节点返回后通过 `graph.stream` 增量落盘 —— 前端轮询即可看到实时进度。

## 🔌 快速开始

### 环境要求

- Python ≥ 3.11
- （可选）真实 LLM API Key —— 任意 OpenAI 兼容端点（DeepSeek / 通义 / 智谱 / OpenAI / 火山方舟）
- （可选）检索服务 API Key —— 缺省时自动使用内置确定性桩

### 安装与运行

```bash
git clone https://github.com/liuhemingmeng/Enterprise-multi-agent-collaboration.git
cd Enterprise-multi-agent-collaboration

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[test]"

# 配置外部服务（可跳过 —— 无 .env 时自动进入确定性桩模式）
cp .env.example .env             # Windows: copy .env.example .env
# 编辑 .env 填入 RAG_API_KEY / LLM_API_KEY 等

# 启动 API + 工作台前端
uvicorn p2_agent.main:app --host 0.0.0.0 --port 8000
```

浏览器打开 **http://127.0.0.1:8000** —— 提交任务后可实时观察节点图、SSE 进度流、
证据引用、评审意见，并在需要时进行人工审批 / 导出 Markdown 报告。

### Docker

```bash
docker build -t agent-workbench .
docker run -p 8000:8000 --env-file .env agent-workbench
```

## ⚙️ 配置项

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `P1_RAG_BASE_URL` | 空 | 检索服务地址；留空回退确定性桩 |
| `RAG_API_KEY` | 空 | 检索服务鉴权 Key（`X-API-Key` 头） |
| `P1_RAG_SCORE_THRESHOLD` | `0.3` | 证据相关度阈值，低于则丢弃（过滤低质量召回） |
| `P1_RAG_LIMIT` | `8` | 单次检索返回条数上限 |
| `P1_RAG_TIMEOUT` | `10` | 检索超时（秒） |
| `LLM_BASE_URL` | 空 | OpenAI 兼容 LLM 端点；留空回退确定性桩 |
| `LLM_API_KEY` | 空 | LLM 服务商 Key |
| `LLM_MODEL` | 空 | 模型名（如 `deepseek-chat`） |
| `LLM_TEMPERATURE` | `0.3` | 生成温度（方案写作取偏低值保稳定） |
| `LLM_TIMEOUT` | `120` | 单次 LLM 调用超时（秒）；长文生成需给足 |
| `LLM_MAX_TOKENS` | `2048` | 单次输出 token 上限，控制延迟与成本 |

双开关逻辑：`RAG_API_KEY` 与 `LLM_API_KEY` 任一缺失，对应组件自动切换到确定性桩，
其余链路（状态机 / 护栏 / 追踪 / 持久化 / 评测）行为完全一致。

## 📡 API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 存活检查 |
| `GET` | `/config` | 外部服务启用状态（不泄露密钥） |
| `POST` | `/tasks` | 提交任务（202 Accepted，后台执行） |
| `GET` | `/tasks/{id}` | 查询任务状态 / 计划 / 证据 / 草稿 / 评审 |
| `POST` | `/tasks/{id}/human-decision` | 人工审批（`approve` / `revise`） |
| `GET` | `/tasks/{id}/events` | 进度事件列表 |
| `GET` | `/tasks/{id}/stream` | SSE 实时流（事件 / Span / 护栏） |
| `GET` | `/tasks/{id}/trace` | 节点级执行追踪（含汇总耗时与成本） |
| `GET` | `/tasks/{id}/guardrails` | 护栏检查结果 |
| `GET` | `/tools` | 工具白名单与预算 |
| `GET` | `/tools/errors` | 工具错误归档 |
| `GET` | `/eval/dataset` | 100 条评测数据集 |
| `POST` | `/eval/run` | 运行单 vs 多 Agent 对照评测（离线桩，零费用） |

## 🛡️ 可靠性设计

| 机制 | 实现 |
| --- | --- |
| 失败安全 | 任何节点异常 → 状态置 `failed` + 错误入档，工作流不产生半成品 |
| 修订回环上限 | Reviewer 判 `revise` 最多重试 3 次，超限安全转人工队列（有限失败保护） |
| 证据不足短路 | 全部检索词已确认无结果时，不再重复检索，直接转人工（不烧预算） |
| 工具预算 | 每任务累计成本超上限后拒绝后续工具调用（`budget_exceeded`） |
| 外部服务重试 | 429 指数退避重试；LLM 解析失败自动回退确定性桩 |
| 断点恢复 | SQLite 逐节点落盘，重启后按 `trace` 最后节点路由续跑 |
| 输入护栏 | 提示注入检测（critical 级直接阻断任务） |

## 📊 评测：单 Agent vs 多 Agent（100 条任务）

`python -m scripts.run_eval`（确定性桩模式，离线、可复现）：

| 指标 | 多 Agent | 单 Agent | 相对变化 |
| --- | --- | --- | --- |
| 自动完成率 | 0.80 | 1.00 | 多 Agent 把修订 / 无证据任务转人工 |
| 安全终止率 | 1.00 | 1.00 | 一致 |
| **引用覆盖率** | **0.90** | **0.30** | **+200%** |
| 单任务成本（模拟计量） | ¥0.04 | ¥0.02 | +100%（多步检索） |
| 平均证据数 | 3.6 | 2.7 | +33% |

**诚实结论**：简单任务单 Agent 更便宜更快；需要检索、审核、人工确认的任务，
多 Agent 的引用覆盖率显著更高、失败处理更可控。多 Agent 的价值不在"更聪明"，
而在**结构化的职责分离带来的可审计性与可控性**。

## 🗂️ 项目结构

```
src/p2_agent/
├── schemas.py          # WorkflowState 状态模型（Pydantic v2，全链路单一状态）
├── graph/workflow.py   # LangGraph 状态机：节点、条件路由、修订回环
├── agents/stubs.py     # 五个智能体节点（桩/真实 LLM 双模式）
├── tools/              # 工具安全边界：白名单、校验、预算、错误归档
│   ├── registry.py     # ToolRegistry（注册制 + 五步执行管线）
│   └── kb_search.py    # kb_search 工具（P1 真实客户端 / 确定性桩同名切换）
├── retrieval.py        # P1 检索客户端（重试 / 阈值过滤 / 字段映射）
├── llm.py              # OpenAI 兼容 LLM 适配器（超时 / max_tokens / 重试 / JSON 抽取）
├── service.py          # 同步工作流服务（graph.stream 增量落盘）
├── async_service.py    # 后台执行器（202 提交 / 事件流 / 人工决策）
├── persistence.py      # SQLite 状态存储
├── tracing.py          # Span / TracingStore（节点级可观测性）
├── guardrails.py       # 三级护栏（输入阻断 / 产物 schema / 工具注入检测）
├── eval/               # 100 条数据集、指标、单/多 Agent 对照
├── events.py           # 进度事件存储
├── settings.py         # 双开关配置装载
└── main.py             # FastAPI 入口 + 前端托管
frontend/index.html     # 零构建单文件工作台（节点图 / SSE / 审批 / 导出）
tests/                  # 90 条测试（工作流 / API / 持久化 / 护栏 / 追踪 / 评测回归）
```

## 🧪 开发与质量门禁

```bash
pytest          # 90 条测试：单元 + 集成 + 故障注入 + 100 条评测回归
ruff check src tests
```

CI（GitHub Actions）：push / PR 到 `main` 自动执行 ruff + pytest，评测回归保证行为不漂移。

## 📖 关键设计决策

<details>
<summary><b>为什么用 LangGraph 状态机，而不是自由循环 Agent 或 AutoGen/CrewAI？</b></summary>

企业交付场景需要**可审计的控制流**：每个节点做什么、失败后去哪、何时转人工，必须是显式声明而非
模型自行决定。LangGraph 的条件边让回环 / 回退 / 人工闸门成为一等公民，且状态是强类型 Pydantic 模型，
天然可持久化、可恢复。自由循环 Agent 无法保证终止性和可复现性；AutoGen/CrewAI 更适合对话式协作，
但控制流隐含在框架里，定制审计与断点恢复的成本反而更高。
</details>

<details>
<summary><b>为什么持久化自建 SQLite 存储，而不用 LangGraph 官方 checkpointer？</b></summary>

状态即业务数据：`trace` / 证据 / 评审意见 / 人工决策都在 `WorkflowState` 里，自建
`SQLiteStateStore` 让"断点续跑"（`route_from_checkpoint` 按最后节点路由）与业务查询共用一套存储，
且演示环境零外部依赖。换 PostgreSQL 只需替换 store 实现（接口已隔离）。
</details>

<details>
<summary><b>为什么评测强制离线确定性桩？</b></summary>

评测要回答的是**架构差异**（单 vs 多 Agent 的引用覆盖率 / 终止行为），不是模型能力。
确定性桩消除了 LLM 输出随机性，使对照实验可复现、零费用、可进 CI；真实 API 的端到端
验证由独立的冒烟脚本（`scripts/live_smoke.py`）承担。
</details>

<details>
<summary><b>成本口径是什么？</b></summary>

成本是**工具调用级模拟计量**：每个工具声明 `cost_per_call`，`CostBudget` 按任务累计并在超限时
拒绝调用——它衡量的是"预算护栏机制"，不是真实账单。Span 的 `tokens` / `cost_usd` 字段已按生产
形态预留，接入真实计费只需在 LLM 适配器填充响应里的 usage 字段。
</details>

## 🗺️ 生产化方向

- 任务执行迁移到独立 worker（Celery / RQ）+ 消息队列，API 层无状态水平扩展
- SQLite → PostgreSQL（多写并发），追踪 / 护栏存储外部化（多实例共享）
- OpenTelemetry 导出、Prometheus 指标、SLO 告警
- API 鉴权（JWT）与租户隔离、密钥接入 KMS
- 熔断器与降级策略替代当前"解析失败回退桩"

---

本项目为个人学习与求职展示项目，欢迎交流讨论。
