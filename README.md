# 项目二：企业方案研究与交付多智能体系统（独立开发版）

本目录是项目二的独立实现，不读取项目一源码、不访问项目一数据库，也不要求项目一先完成。

## 当前阶段

**Stage 10：真实外部服务接入（P1 检索 API + 真实 LLM API）**

阶段 1–9 已完成状态机、条件回退、持久化、异步 API、工具安全边界、100 条评测对照、前端/Docker/CI、可观测性 Tracing/SSE、护栏层 Guardrails。阶段 10 把"确定性桩"换成真实外部服务，让系统真正产出可审核的企业方案：

- **P1 检索客户端**：`P1RetrievalClient` 调 P1 `/corpus/search`，含 429 退避重试、409/422 处理、score 阈值过滤、字段映射；经 `P1SearchTool` 接进既有工具白名单 / 预算 / 追踪。
- **真实 LLM 适配器**：`LLMClient`（OpenAI 兼容）驱动 Planner/Analyst/Writer/Reviewer；解析失败自动回退确定性桩，工作流不崩溃。
- **双开关设计**：`RAG_API_KEY` / `LLM_API_KEY` 缺失时自动回退桩，**CI 评测与单元测试零费用、可复现**。
- `GET /config` 汇报两套服务启用状态（不泄露密钥）。

配置见 `.env.example`（复制为 `.env` 填入）：P1 检索需 `RAG_API_KEY`；真实 LLM 需 `LLM_BASE_URL`+`LLM_API_KEY`+`LLM_MODEL`（DeepSeek / 通义 / 智谱 / OpenAI 任选）。

先用本地 deterministic stub 代替 LLM 与 P1 API，跑通：

`提交任务 -> Planner -> Retriever -> Analyst -> Writer -> Reviewer -> Human Approval -> Export`

当前已具备：工具白名单与安全边界（阶段 5）、SQLite 持久化与恢复（阶段 3）、异步任务 API（阶段 4），以及本阶段的 **100 条确定性评测集** 与 **单 Agent vs 多 Agent 对照实验**。

评测结论（100 条业务任务，`python -m scripts.run_eval`）：

| 指标 | 多 Agent | 单 Agent | 相对变化 |
| --- | --- | --- | --- |
| 自动完成率 | 0.80 | 1.00 | 多 Agent 把修订/无证据任务转人工 |
| 安全终止率 | 1.00 | 1.00 | 一致 |
| 引用覆盖率 | 0.90 | 0.30 | **+200%** |
| 单任务成本 | ¥0.04 | ¥0.02 | +100%（多步检索） |
| 平均证据数 | 3.6 | 2.7 | +33% |

诚实结论：对简单任务单 Agent 更便宜更快；对需要检索/审核/确认的任务，多 Agent 引用覆盖率显著更高、失败处理更可控。所有外部依赖都通过端口（protocol）注入，后续可替换为真实模型、P1 HTTP 客户端、PostgreSQL checkpointer、Redis worker。

## 分步路线

1. 状态模型与节点契约：字段归属、输入校验、确定性 stub、完整单测。
2. LangGraph 条件回退：Reviewer 的 revise / insufficient / human_review 分支与最大重试。
3. 持久化与恢复：SQLite 开发 checkpointer，记录轨迹，模拟重启恢复。
4. FastAPI 异步任务 API：创建、查询、审批、导出、健康检查。
5. 工具白名单与安全边界：参数校验、超时、权限、成本预算、错误归档。
6. 评测与对照实验：单 Agent vs 多 Agent，引用覆盖率、恢复成功率、成本与耗时。
7. 前端、Docker、CI 与部署：进度页、人工审批页、报告导出和线上演示。

每一步都必须先实现，再跑单元测试、集成测试、故障注入测试和静态检查；没有通过质量门禁不进入下一步。

## 运行

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -e ".[test]"
pytest
```

启动 API 与服务台 UI：

```bash
uvicorn p2_agent.main:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://127.0.0.1:8000/insight
```

容器化与 CI：

```bash
docker build -t p2-agent-workbench .
docker run -p 8000:8000 p2-agent-workbench
# CI：push/PR 到 main 自动跑 ruff + pytest（含 100 条评测回归）
```
