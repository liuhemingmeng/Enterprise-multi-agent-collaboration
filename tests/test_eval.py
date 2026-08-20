"""Stage 6 tests: evaluation set, single-agent baseline, comparison report.

The 100-task comparison run is deterministic (offline stubs) and takes
~5-40s depending on the machine; running it once per test would repeat the
full workload up to six times and can exhaust memory / kill the test runner
on constrained machines. The report is therefore computed once per module
and shared by every assertion below.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from p2_agent.eval.dataset import (
    build_evaluation_set,
    load_dataset,
    save_dataset,
)
from p2_agent.eval.runner import run_comparison
from p2_agent.eval.single_agent import single_agent_run
from p2_agent.tools.kb_search import KBSearchTool
from p2_agent.tools.registry import CostBudget, ErrorArchive, ToolRegistry

_REPORT: dict | None = None


@pytest.fixture(scope="module")
def report() -> dict:
    global _REPORT
    if _REPORT is None:
        _REPORT = run_comparison()
    return _REPORT


def _single_registry(tmp_path: Path) -> ToolRegistry:
    registry = ToolRegistry(budget=CostBudget(max_cost=100.0))
    registry.register(KBSearchTool())
    registry.error_archive = ErrorArchive(str(tmp_path / "e.sqlite3"))
    return registry


def test_dataset_has_at_least_100_deterministic_tasks():
    first = build_evaluation_set()
    second = build_evaluation_set()
    assert len(first) >= 100
    assert [t.task_id for t in first] == [t.task_id for t in second]
    assert [t.goal for t in first] == [t.goal for t in second]


def test_dataset_category_distribution():
    tasks = build_evaluation_set()
    cats = {t.category for t in tasks}
    assert cats == {"normal", "complex", "revise", "no_evidence"}
    # 10 domains x (5 normal + 3 complex + 1 revise + 1 no_evidence) = 100
    assert sum(1 for t in tasks if t.category == "normal") == 50
    assert sum(1 for t in tasks if t.category == "complex") == 30
    assert sum(1 for t in tasks if t.category == "revise") == 10
    assert sum(1 for t in tasks if t.category == "no_evidence") == 10


def test_dataset_save_and_load_round_trip(tmp_path):
    path = save_dataset(tmp_path / "tasks.json")
    loaded = load_dataset(path)
    assert len(loaded) == len(build_evaluation_set())
    assert loaded[0].task_id == "task-001"


def test_single_agent_cites_only_top_one(tmp_path):
    registry = _single_registry(tmp_path)
    outcome = single_agent_run(
        goal="为制造业预测性维护场景设计企业实施方案",
        category="normal",
        domain="制造业预测性维护",
        task_id="sa-001",
        registry=registry,
    )
    assert outcome.pipeline == "single_agent"
    assert outcome.status == "completed"
    # single search top_k=3 -> 3 evidence, but only top-1 cited
    assert outcome.evidence_count == 3
    assert len(outcome.cited_evidence_ids) == 1
    assert outcome.citation_coverage == 1 / 3


def test_single_agent_has_no_reviewer_gate_on_no_evidence(tmp_path):
    registry = _single_registry(tmp_path)
    outcome = single_agent_run(
        goal="__no_evidence__关于制造业预测性维护的机密内部资料方案",
        category="no_evidence",
        domain="制造业预测性维护",
        task_id="sa-002",
        registry=registry,
    )
    # single agent still "completes" with zero citations (hallucination risk)
    assert outcome.status == "completed"
    assert outcome.evidence_count == 0
    assert outcome.citation_coverage == 0.0


def test_comparison_report_is_complete_and_dataset_size_100(report):
    pass  # report from fixture
    assert report["dataset_size"] >= 100
    for key in ("single_agent", "multi_agent", "deltas", "relative"):
        assert key in report
    ma = report["multi_agent"]
    sa = report["single_agent"]
    assert ma["n"] == sa["n"] == report["dataset_size"]


def test_multi_agent_citation_coverage_beats_single_agent(report):
    pass  # report from fixture
    ma = report["multi_agent"]["mean_citation_coverage"]
    sa = report["single_agent"]["mean_citation_coverage"]
    assert ma > sa
    # multi-agent cites every retrieved evidence in the happy path (100%)
    assert ma >= 0.9


def test_multi_agent_costs_more_than_single_agent(report):
    pass  # report from fixture
    ma = report["multi_agent"]["mean_cost"]
    sa = report["single_agent"]["mean_cost"]
    assert ma > sa  # more retrieval calls -> higher cost
    assert ma > 0 and sa > 0


def test_both_pipelines_reach_terminal_state(report):
    pass  # report from fixture
    assert report["multi_agent"]["safe_termination_rate"] == 1.0
    assert report["single_agent"]["safe_termination_rate"] == 1.0


def test_no_evidence_category_routes_multi_agent_to_human(report):
    pass  # report from fixture
    by_cat = report["by_category_multi_agent"]["no_evidence"]
    assert by_cat["auto_completion_rate"] == 0.0  # goes to human, not auto-completed
    # but it terminates safely instead of emitting an uncited draft
    single_cat = report["by_category_single_agent"]["no_evidence"]
    assert single_cat["auto_completion_rate"] == 1.0
    assert single_cat["mean_citation_coverage"] == 0.0


def test_revise_category_routes_multi_agent_to_human(report):
    pass  # report from fixture
    by_cat = report["by_category_multi_agent"]["revise"]
    assert by_cat["auto_completion_rate"] == 0.0  # needs human after retries
