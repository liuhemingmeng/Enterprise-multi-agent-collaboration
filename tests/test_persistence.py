from p2_agent.agents.stubs import planner_node
from p2_agent.persistence import SQLiteStateStore
from p2_agent.schemas import WorkflowState
from p2_agent.service import WorkflowService


def test_snapshot_survives_new_store_instance(tmp_path):
    path = tmp_path / "state.sqlite3"
    first_store = SQLiteStateStore(path)
    original = WorkflowState(user_goal="持久化测试", status="need_human", trace=["planner"])
    first_store.save(original)

    restarted_store = SQLiteStateStore(path)
    restored = restarted_store.get(original.task_id)
    assert restored is not None
    assert restored.task_id == original.task_id
    assert restored.status == "need_human"
    assert restored.trace == ["planner"]


def test_resume_from_planner_checkpoint_completes_after_restart(tmp_path):
    service = WorkflowService(SQLiteStateStore(tmp_path / "state.sqlite3"))
    planned = WorkflowState(user_goal="从规划节点恢复")
    update = planner_node(planned)
    checkpoint = planned.model_copy(update=update)
    service.store.save(checkpoint)
    restarted = WorkflowService(SQLiteStateStore(tmp_path / "state.sqlite3"))
    resumed = restarted.resume(checkpoint.task_id)
    assert resumed.status == "completed"
    assert resumed.trace[0] == "planner"


def test_service_saves_completed_state_and_resume_is_idempotent(tmp_path):
    service = WorkflowService(SQLiteStateStore(tmp_path / "state.sqlite3"))
    completed = service.create_and_run("服务重启恢复测试")
    assert completed.status == "completed"

    restarted = WorkflowService(SQLiteStateStore(tmp_path / "state.sqlite3"))
    resumed = restarted.resume(completed.task_id)
    assert resumed.status == "completed"
    assert resumed.trace == completed.trace


def test_missing_task_is_not_silently_created(tmp_path):
    service = WorkflowService(SQLiteStateStore(tmp_path / "state.sqlite3"))
    try:
        service.resume("missing-task")
    except KeyError as exc:
        assert "task not found" in str(exc)
    else:
        raise AssertionError("missing task must raise KeyError")


def test_corrupted_snapshot_fails_loudly(tmp_path):
    path = tmp_path / "state.sqlite3"
    store = SQLiteStateStore(path)
    state = WorkflowState(user_goal="损坏数据测试")
    store.save(state)
    with store._connect() as connection:
        connection.execute(
            "UPDATE task_snapshots SET state_json = ? WHERE task_id = ?",
            ("{not-json", str(state.task_id)),
        )
        connection.commit()
    try:
        store.get(state.task_id)
    except ValueError as exc:
        assert "corrupted snapshot" in str(exc)
    else:
        raise AssertionError("corrupted snapshot must be rejected")
