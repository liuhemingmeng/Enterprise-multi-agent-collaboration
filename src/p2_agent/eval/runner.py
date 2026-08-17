from __future__ import annotations

import time
from pathlib import Path

from p2_agent.eval.dataset import EvalTask, build_evaluation_set
from p2_agent.eval.metrics import EvalOutcome, summarize, summarize_by_category
from p2_agent.eval.single_agent import single_agent_run
from p2_agent.persistence import SQLiteStateStore
from p2_agent.service import WorkflowService
from p2_agent.tools.kb_search import KBSearchTool
from p2_agent.tools.registry import CostBudget, ErrorArchive, ToolRegistry


def _run_multi_agent(task: EvalTask, service: WorkflowService) -> EvalOutcome:
    state = service.create_and_run(task.goal)
    evidence_count = len(state.evidence)
    cited = state.analysis.cited_evidence_ids if state.analysis else []
    coverage = (len(cited) / evidence_count) if evidence_count else 0.0
    cost = service.registry.budget.spent(str(state.task_id))
    return EvalOutcome(
        task_id=task.task_id,
        goal=task.goal,
        category=task.category,
        domain=task.domain,
        pipeline="multi_agent",
        status=state.status,
        draft=state.draft,
        evidence_count=evidence_count,
        cited_evidence_ids=cited,
        citation_coverage=coverage,
        cost=cost,
        duration_ms=0.0,  # filled by caller via wall clock
        reached_terminal=state.status in {"completed", "need_human", "failed"},
    )


def _build_single_registry(workdir: Path) -> ToolRegistry:
    registry = ToolRegistry(budget=CostBudget(max_cost=100.0))
    registry.register(KBSearchTool())
    registry.error_archive = ErrorArchive(str(workdir / "eval_errors.sqlite3"))
    return registry


def run_comparison(
    tasks: list[EvalTask] | None = None, workdir: str | Path | None = None
) -> dict:
    """Run both pipelines over the dataset and return a comparison report."""
    tasks = tasks or build_evaluation_set()

    # Use a stable directory rather than a TemporaryDirectory: the evaluation
    # leaves small SQLite files behind and reusing one path avoids the
    # "file in use" cleanup race on Windows while keeping runs reproducible.
    workdir = Path(workdir) if workdir is not None else Path("eval/runs")
    workdir.mkdir(parents=True, exist_ok=True)

    multi_store = SQLiteStateStore(workdir / "eval_state.sqlite3")
    service = WorkflowService(multi_store)
    service.registry.budget.max_cost = 100.0

    single_registry = _build_single_registry(workdir)

    multi_outcomes: list[EvalOutcome] = []
    single_outcomes: list[EvalOutcome] = []

    for task in tasks:
        t0 = time.monotonic()
        mo = _run_multi_agent(task, service)
        mo.duration_ms = round((time.monotonic() - t0) * 1000, 2)
        multi_outcomes.append(mo)

        so = single_agent_run(
            goal=task.goal,
            category=task.category,
            domain=task.domain,
            task_id=task.task_id,
            registry=single_registry,
        )
        single_outcomes.append(so)

    multi_summary = summarize(multi_outcomes)
    single_summary = summarize(single_outcomes)

    def delta(metric: str) -> float:
        return round(multi_summary[metric] - single_summary[metric], 4)

    def rel_delta(metric: str) -> str:
        base = single_summary[metric]
        if base == 0:
            return "n/a"
        return f"{(multi_summary[metric] / base - 1) * 100:+.1f}%"

    report = {
        "dataset_size": len(tasks),
        "single_agent": single_summary,
        "multi_agent": multi_summary,
        "deltas": {
            "auto_completion_rate": delta("auto_completion_rate"),
            "safe_termination_rate": delta("safe_termination_rate"),
            "mean_citation_coverage": delta("mean_citation_coverage"),
            "mean_cost": delta("mean_cost"),
            "mean_duration_ms": delta("mean_duration_ms"),
            "mean_evidence_count": delta("mean_evidence_count"),
        },
        "relative": {
            "mean_citation_coverage": rel_delta("mean_citation_coverage"),
            "mean_cost": rel_delta("mean_cost"),
            "mean_duration_ms": rel_delta("mean_duration_ms"),
            "mean_evidence_count": rel_delta("mean_evidence_count"),
        },
        "by_category_multi_agent": summarize_by_category(multi_outcomes),
        "by_category_single_agent": summarize_by_category(single_outcomes),
        "outcomes": [o.model_dump() for o in multi_outcomes + single_outcomes],
    }
    return report
