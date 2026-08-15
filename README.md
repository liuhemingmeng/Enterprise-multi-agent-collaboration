# 项目二：企业方案研究与交付多智能体系统（独立开发版）

本目录是项目二的独立实现，不读取项目一源码、不访问项目一数据库，也不要求项目一先完成。

## 当前阶段

**Stage 3：持久化与可恢复执行（SQLite 开发版）**

先用本地 deterministic stub 代替 LLM 与 P1 API，跑通：

`提交任务 -> Planner -> Retriever -> Analyst -> Writer -> Reviewer -> Human Approval -> Export`

当前已经增加 SQLite 快照与任务恢复接口；SQLite 是开发阶段的应用层状态存储，不等同于生产级 LangGraph PostgreSQL checkpointer。所有外部依赖都通过端口（protocol）注入，后续可以替换为真实模型、P1 HTTP 客户端、PostgreSQL checkpointer、Redis worker。

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
