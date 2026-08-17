from __future__ import annotations

from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvalOutcome(BaseModel):
    """One row of evaluation output, identical shape for both pipelines."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    goal: str
    category: str
    domain: str
    pipeline: Literal["single_agent", "multi_agent"]
    status: str  # completed | need_human | failed
    draft: str = ""
    evidence_count: int = 0
    cited_evidence_ids: list[str] = Field(default_factory=list)
    citation_coverage: float = Field(default=0.0, ge=0, le=1)
    cost: float = Field(default=0.0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0)
    reached_terminal: bool = True
    error: str | None = None


def summarize(outcomes: list[EvalOutcome]) -> dict:
    """Aggregate a list of outcomes into headline metrics."""
    if not outcomes:
        return {}
    n = len(outcomes)
    completed = [o for o in outcomes if o.status == "completed"]
    cited_total = sum(o.cited_evidence_ids.__len__() for o in outcomes)
    evidence_total = sum(o.evidence_count for o in outcomes)
    return {
        "n": n,
        "auto_completion_rate": round(len(completed) / n, 4),
        "safe_termination_rate": round(sum(o.reached_terminal for o in outcomes) / n, 4),
        "mean_citation_coverage": round(mean(o.citation_coverage for o in outcomes), 4),
        "mean_cost": round(mean(o.cost for o in outcomes), 4),
        "mean_duration_ms": round(mean(o.duration_ms for o in outcomes), 2),
        "mean_evidence_count": round(mean(o.evidence_count for o in outcomes), 2),
        "total_cited_evidence": cited_total,
        "total_retrieved_evidence": evidence_total,
    }


def summarize_by_category(
    outcomes: list[EvalOutcome],
) -> dict[str, dict]:
    by_cat: dict[str, list[EvalOutcome]] = {}
    for o in outcomes:
        by_cat.setdefault(o.category, []).append(o)
    return {cat: summarize(group) for cat, group in sorted(by_cat.items())}
