"""Basic tests for NovaTask sample project."""
from src.config import get_config
from src.worker import poll_and_execute_task

def test_config_loader():
    cfg = get_config()
    assert cfg["port"] == 8000
    assert "database_url" in cfg

def test_worker_execution():
    res = poll_and_execute_task("job-101")
    assert res["status"] == "completed"
    assert res["task_id"] == "job-101"
