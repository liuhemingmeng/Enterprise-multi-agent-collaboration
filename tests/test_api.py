from fastapi.testclient import TestClient

from p2_agent.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_task_returns_completed_state():
    with TestClient(app) as client:
        response = client.post("/tasks", json={"user_goal": "设计设备维护方案"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["draft"].startswith("# 企业方案初稿")


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


def test_unknown_task_returns_404():
    with TestClient(app) as client:
        response = client.get("/tasks/not-found")
    assert response.status_code == 404
