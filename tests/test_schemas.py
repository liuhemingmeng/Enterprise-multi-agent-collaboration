import pytest
from pydantic import ValidationError

from p2_agent.schemas import Plan, Subtask, WorkflowState


def test_plan_rejects_duplicate_subtask_ids():
    with pytest.raises(ValidationError, match="unique"):
        Plan(
            goal="x",
            subtasks=[Subtask(id="s1", title="a"), Subtask(id="s1", title="b")],
        )


def test_state_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        WorkflowState(user_goal="x", unknown="bad")


def test_public_dict_contains_uuid_as_json_string():
    data = WorkflowState(user_goal="x").public_dict()
    assert isinstance(data["task_id"], str)
