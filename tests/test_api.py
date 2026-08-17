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
