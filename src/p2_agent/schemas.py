from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Subtask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    depends_on: list[str] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list, max_length=8)
    needs_human_review: bool = False


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=2000)
    subtasks: list[Subtask] = Field(min_length=1, max_length=20)
    deliverable_template: Literal["solution_brief"] = "solution_brief"

    @field_validator("subtasks")
    @classmethod
    def unique_ids(cls, value: list[Subtask]) -> list[Subtask]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("subtask ids must be unique")
        return value


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    source_type: Literal["local_stub", "local", "web"] = "local_stub"
    doc_id: str = Field(min_length=1)
    doc_name: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=5000)
    score: float = Field(ge=0, le=1)
    url: str | None = None


class Analysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved", "revise", "insufficient_evidence", "human_review"]
    reasons: list[str] = Field(default_factory=list)
    checked_evidence_ids: list[str] = Field(default_factory=list)


class WorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: UUID = Field(default_factory=uuid4)
    user_goal: str = Field(min_length=1, max_length=2000)
    plan: Plan | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    analysis: Analysis | None = None
    draft: str = ""
    review: Review | None = None
    retry_count: int = Field(default=0, ge=0, le=10)
    status: Literal[
        "pending",
        "queued",
        "running",
        "need_human",
        "approved",
        "completed",
        "failed",
    ] = "pending"
    errors: list[str] = Field(default_factory=list, max_length=50)
    trace: list[str] = Field(default_factory=list, max_length=100)
    empty_queries: list[str] = Field(default_factory=list, max_length=50)
    require_human_approval: bool = False
    human_decision: Literal["", "approve", "revise"] = ""

    def public_dict(self) -> dict:
        return self.model_dump(mode="json")
