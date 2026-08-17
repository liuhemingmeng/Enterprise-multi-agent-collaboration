from fastapi.testclient import TestClient

from p2_agent.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_task_returns_accepted_state_and_can_complete():
    with TestClient(app) as client:
        response = client.post("/tasks", json={"user_goal": "设计设备维护方案"})
    assert response.status_code == 202
    body = response.json()
    assert body["status"] in {"queued", "running", "completed"}
    with TestClient(app) as client:
        done = client.post(f"/tasks/{body['task_id']}/resume")
    assert done.status_code == 200
    assert done.json()["status"] == "completed"
    assert done.json()["draft"].startswith("# 企业方案初稿")


def test_create_task_validates_blank_goal():
    with TestClient(app) as client:
        response = client.post("/tasks", json={"user_goal": ""})
    assert response.status_code == 422


def test_task_can_be_fetched_after_creation():
    with TestClient(app) as client:
        created = client.post("/tasks", json={"user_goal": "查询任务状态"}).json()
        response = client.get(f"/tasks/{created['task_id']}")
    assert response.status_code == 200
    assert response.json()["task_id"] == created["task_id"]


def test_task_events_are_queryable():
    with TestClient(app) as client:
        created = client.post("/tasks", json={"user_goal": "查看进度事件"}).json()
        response = client.get(f"/tasks/{created['task_id']}/events")
    assert response.status_code == 200
    assert response.json()["events"]
    assert response.json()["events"][0]["event"] == "queued"


def test_unknown_task_returns_404():
    with TestClient(app) as client:
        response = client.get("/tasks/not-found")
    assert response.status_code == 404


def test_eval_dataset_endpoint_reports_100_tasks():
    with TestClient(app) as client:
        response = client.get("/eval/dataset")
    assert response.status_code == 200
    assert response.json()["size"] >= 100
    assert response.json()["sample"]


def test_eval_run_endpoint_returns_comparison_report():
    with TestClient(app) as client:
        response = client.post("/eval/run")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_size"] >= 100
    assert "multi_agent" in body and "single_agent" in body
    # multi-agent must beat single-agent on citation coverage
    ma_cov = body["multi_agent"]["mean_citation_coverage"]
    sa_cov = body["single_agent"]["mean_citation_coverage"]
    assert ma_cov > sa_cov


def test_human_approval_flow_pauses_then_completes():
    import time

    with TestClient(app) as client:
        created = client.post(
            "/tasks",
            json={"user_goal": "需要人工确认的设备维护方案", "require_human_approval": True},
        ).json()
        task_id = created["task_id"]
        # wait until the workflow pauses for human input
        state = None
        for _ in range(100):
            state = client.get(f"/tasks/{task_id}").json()
            if state["status"] in {"need_human", "completed", "failed"}:
                break
            time.sleep(0.1)
        assert state["status"] == "need_human"
        # human approves -> export -> completed
        decided = client.post(
            f"/tasks/{task_id}/human-decision", json={"decision": "approve"}
        ).json()
        assert decided["status"] == "completed"
        assert decided["draft"].startswith("# 企业方案初稿")


def test_human_decision_on_unknown_task_returns_404():
    with TestClient(app) as client:
        response = client.post(
            "/tasks/does-not-exist/human-decision", json={"decision": "approve"}
        )
    assert response.status_code == 404


def test_frontend_ui_is_served():
    with TestClient(app) as client:
        response = client.get("/insight")
    assert response.status_code == 200
    assert "企业方案多智能体工作流" in response.text
