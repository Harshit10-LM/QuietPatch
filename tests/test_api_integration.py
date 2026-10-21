"""Integration tests for QuietPatch REST API and end-to-end workflows."""
import json
from pathlib import Path
import pytest
from backend.app import create_app
from backend.config import Config

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_check_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["app"] == "QuietPatch"
    assert "mode" in data

def test_public_config_endpoint(client):
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.get_json()
    assert "allowed_root" in data
    assert "ai_provider" in data
    assert "sentry_configured" in data
    # Secrets must NEVER be present in public config
    assert "GEMMA_API_KEY" not in data
    assert "gemma_api_key" not in data
    assert "has_gemma_api_key" in data

def test_providers_status_endpoint(client):
    res = client.get("/api/providers/status")
    assert res.status_code == 200
    data = res.get_json()
    assert "ai" in data
    assert "observability" in data
    assert "status" in data["observability"]

def test_repositories_endpoints(client):
    res = client.get("/api/repositories")
    assert res.status_code == 200
    data = res.get_json()
    assert "repositories" in data
    assert len(data["repositories"]) > 0

    # Validate local sample repo path
    val_res = client.post("/api/repositories/validate", json={"type": "local", "path": "sample-repo"})
    assert val_res.status_code == 200
    val_data = val_res.get_json()
    assert val_data["valid"] is True

def test_demo_scan_lifecycle(client):
    # 1. Trigger demo scan
    res = client.post("/api/demo/scan")
    assert res.status_code == 200
    run_data = res.get_json()
    run_id = run_data["id"]
    assert run_data["status"] == "completed"
    assert len(run_data["findings"]) > 0
    assert run_data["files_scanned"] > 0
    assert len(run_data["patches"]) > 0

    # 2. Retrieve scan by ID
    get_res = client.get(f"/api/scans/{run_id}")
    assert get_res.status_code == 200
    assert get_res.get_json()["id"] == run_id

    # 3. Filter findings
    find_res = client.get(f"/api/scans/{run_id}/findings?category=configuration")
    assert find_res.status_code == 200
    findings = find_res.get_json()
    assert all(f["category"] == "configuration" for f in findings)

    # 4. Generate report
    rep_res = client.get(f"/api/scans/{run_id}/report")
    assert rep_res.status_code == 200
    rep_data = rep_res.get_json()
    assert rep_data["run_id"] == run_id
    assert rep_data["total_findings"] > 0
    assert "time_saved" in rep_data

    # 5. Export Markdown report
    md_res = client.get(f"/api/scans/{run_id}/report/markdown")
    assert md_res.status_code == 200
    assert "# QuietPatch Maintenance Report" in md_res.get_data(as_text=True)

def test_patch_approval_workflow(client):
    # Trigger scan to generate patches
    scan_res = client.post("/api/demo/scan")
    run_data = scan_res.get_json()
    assert len(run_data["patches"]) > 0
    patch = run_data["patches"][0]
    patch_id = patch["id"]

    # Approve patch
    app_res = client.post(f"/api/patches/{patch_id}/approve")
    assert app_res.status_code == 200
    assert app_res.get_json()["status"] == "approved"

    # Reject patch
    rej_res = client.post(f"/api/patches/{patch_id}/reject")
    assert rej_res.status_code == 200
    assert rej_res.get_json()["status"] == "rejected"

def test_frontend_serving(client):
    res = client.get("/")
    assert res.status_code == 200
    text = res.get_data(as_text=True)
    # Check that either built html is served or fallback message
    assert "<!doctype html>" in text.lower() or "QuietPatch" in text


def test_retry_ai_endpoint(client):
    # 1. Start a demo scan
    scan_res = client.post("/api/demo/scan")
    assert scan_res.status_code == 200
    run_data = scan_res.get_json()
    run_id = run_data["id"]

    # 2. Call retry-ai
    retry_res = client.post(f"/api/scans/{run_id}/retry-ai")
    assert retry_res.status_code == 200
    updated_run = retry_res.get_json()
    assert updated_run["id"] == run_id
    assert updated_run["status"] == "completed"
    assert len(updated_run["findings"]) == len(run_data["findings"])

    # 3. Non-existent run returns 404
    missing_res = client.post("/api/scans/non-existent-run-id/retry-ai")
    assert missing_res.status_code == 404

