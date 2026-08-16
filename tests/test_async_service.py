from p2_agent.async_service import AsyncWorkflowService
from p2_agent.persistence import SQLiteStateStore


def test_submit_returns_before_waiting_for_worker(tmp_path):
    service = AsyncWorkflowService(SQLiteStateStore(tmp_path / "state.sqlite3"))
    queued = service.submit("异步任务测试")
    assert queued.status == "queued"
    completed = service.wait(queued.task_id)
    assert completed.status == "completed"
    events = service.events_for(queued.task_id)
    assert [event["event"] for event in events] == ["queued", "started", "completed"]


def test_async_service_persists_task_for_new_service(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = AsyncWorkflowService(SQLiteStateStore(path))
    queued = first.submit("跨服务查询")
    first.wait(queued.task_id)
    second = AsyncWorkflowService(SQLiteStateStore(path))
    restored = second.get(queued.task_id)
    assert restored is not None
    assert restored.status == "completed"
