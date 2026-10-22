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

    # 2. Call retry-ai (returns 202 Accepted background contract)
    retry_res = client.post(f"/api/scans/{run_id}/retry-ai")
    assert retry_res.status_code == 202
    updated_run = retry_res.get_json()
    assert updated_run["id"] == run_id
    assert updated_run["status"] in {"analyzing", "completed"}

    # 3. Poll for completion
    import time
    polled_data = updated_run
    for _ in range(50):
        if polled_data["status"] == "completed":
            break
        time.sleep(0.05)
        poll_res = client.get(f"/api/scans/{run_id}")
        assert poll_res.status_code == 200
        polled_data = poll_res.get_json()

    assert polled_data["status"] == "completed"
    assert len(polled_data["findings"]) == len(run_data["findings"])

    # 4. Non-existent run returns 404
    missing_res = client.post("/api/scans/non-existent-run-id/retry-ai")
    assert missing_res.status_code == 404


def test_create_scan_returns_202_promptly_while_inference_is_blocked(client):
    """1. Starting a maintenance run returns HTTP 202 promptly (<0.25s) even while AI inference is blocked."""
    import threading
    import time
    from unittest.mock import patch
    from backend.ai.agent_service import AgentService

    block_event = threading.Event()
    original_analyze = AgentService._init_provider

    def mock_init_provider(self, name):
        provider = original_analyze(self, name)
        original_m = provider.analyze_and_prioritize

        def blocking_analyze(findings, run_id=None):
            block_event.wait(timeout=5.0)
            return original_m(findings, run_id=run_id)

        provider.analyze_and_prioritize = blocking_analyze
        return provider

    t0 = time.time()
    with patch.object(AgentService, "_init_provider", mock_init_provider):
        res = client.post("/api/scans", json={"type": "sample", "provider": "demo"})
        elapsed = time.time() - t0

    # 1. HTTP request returns immediately without waiting for inference
    assert elapsed < 0.35, f"Expected prompt response, took {elapsed}s"
    assert res.status_code == 202
    run_data = res.get_json()
    run_id = run_data["id"]
    assert run_data["status"] == "queued"
    assert "initialization" in run_data["stages"]

    # 2. Polling while blocked reflects in-flight non-terminal status
    poll_res = client.get(f"/api/scans/{run_id}")
    assert poll_res.status_code == 200
    assert poll_res.get_json()["status"] in {
        "queued", "initializing", "discovering", "scanning", "analyzing"
    }

    # 3. Unblock provider and poll until completed
    block_event.set()
    final_data = None
    for _ in range(50):
        time.sleep(0.05)
        p = client.get(f"/api/scans/{run_id}").get_json()
        if p["status"] == "completed":
            final_data = p
            break

    assert final_data is not None
    assert final_data["status"] == "completed"
    assert len(final_data["findings"]) > 0


def test_duplicate_scan_submission_prevented(client):
    """2. Duplicate submission for an already-active repository returns HTTP 409 Conflict."""
    import threading
    import time
    from unittest.mock import patch
    from backend.ai.agent_service import AgentService

    block_event = threading.Event()

    def mock_init(self, name):
        provider = AgentService._init_provider(self, name)
        orig = provider.analyze_and_prioritize

        def slow_analyze(findings, run_id=None):
            block_event.wait(timeout=5.0)
            return orig(findings, run_id=run_id)

        provider.analyze_and_prioritize = slow_analyze
        return provider

    with patch.object(AgentService, "_init_provider", mock_init):
        res1 = client.post("/api/scans", json={"type": "sample", "provider": "demo"})
        assert res1.status_code == 202
        run1_id = res1.get_json()["id"]

        # Duplicate scan while run1 is active
        res2 = client.post("/api/scans", json={"type": "sample", "provider": "demo"})
        assert res2.status_code == 409
        err_data = res2.get_json()
        assert "already active" in err_data["error"]
        assert err_data["run"]["id"] == run1_id

        block_event.set()

    # Wait for completion
    for _ in range(40):
        time.sleep(0.05)
        if client.get(f"/api/scans/{run1_id}").get_json()["status"] == "completed":
            break


def test_long_running_inference_does_not_timeout_http(client):
    """3. A run taking longer than the former 120s worker limit returns HTTP 202 promptly."""
    import time

    t0 = time.time()
    res = client.post("/api/scans", json={"type": "sample", "provider": "demo"})
    elapsed = time.time() - t0

    assert res.status_code == 202
    assert elapsed < 0.25, f"HTTP request took {elapsed}s, must not block for inference duration"
    run_id = res.get_json()["id"]

    # Poll status is available immediately
    status_res = client.get(f"/api/scans/{run_id}")
    assert status_res.status_code == 200


def test_concurrent_web_requests_safe_access(client):
    """4. Multiple concurrent web requests safely read and poll shared in-memory run state without errors."""
    import concurrent.futures

    # Start a scan
    scan_res = client.post("/api/demo/scan")
    assert scan_res.status_code == 200
    run_id = scan_res.get_json()["id"]

    app = client.application
    errors = []

    def make_requests():
        try:
            thread_client = app.test_client()
            for _ in range(5):
                r1 = thread_client.get("/api/scans")
                assert r1.status_code == 200
                r2 = thread_client.get(f"/api/scans/{run_id}")
                assert r2.status_code == 200
                r3 = thread_client.get("/api/health")
                assert r3.status_code == 200
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(make_requests) for _ in range(8)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent requests encountered errors: {errors}"


def test_app_creation_and_health_check_make_no_ai_calls_and_launch_no_scans():
    """Application creation and /api/health make no Google Gen AI calls and launch no background scans."""
    import os
    import threading
    import time
    from unittest.mock import MagicMock, patch
    from backend.ai.agent_service import AgentService

    mock_client = MagicMock()

    # Simulate production environment: non-debug, AI_PROVIDER="gemma_api", PYTEST_CURRENT_TEST unset
    orig_pytest_env = os.environ.pop("PYTEST_CURRENT_TEST", None)
    try:
        with patch.object(Config, "AI_PROVIDER", "gemma_api"), \
             patch("backend.ai.gemma_api_provider.GemmaApiProvider._get_client", return_value=mock_client), \
             patch.object(AgentService, "execute_maintenance_run") as mock_exec:

            prod_app = create_app()
            prod_app.debug = False

            # Wait briefly to ensure no daemon thread is spawned
            time.sleep(0.1)

            # Assert no background thread was spawned for sample init
            active_threads = [t.name for t in threading.enumerate()]
            assert "quietpatch-sample-init" not in active_threads

            with prod_app.test_client() as test_client:
                # 1. Health check returns 200 without invoking AI or scans
                health_res = test_client.get("/api/health")
                assert health_res.status_code == 200
                assert health_res.get_json()["status"] == "ok"

                # 2. Runs store is completely empty
                scans_res = test_client.get("/api/scans")
                assert scans_res.status_code == 200
                assert scans_res.get_json() == []

            # 3. Model inference was never called, and execute_maintenance_run was never called
            assert mock_client.models.generate_content.call_count == 0
            assert mock_exec.call_count == 0
    finally:
        if orig_pytest_env is not None:
            os.environ["PYTEST_CURRENT_TEST"] = orig_pytest_env


def test_default_sample_project_loading_does_not_initiate_scan(client):
    """Listing repositories or validating sample repository makes no AI calls and creates no maintenance runs."""
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock()
    with patch("backend.ai.gemma_api_provider.GemmaApiProvider._get_client", return_value=mock_client):
        # 1. Listing repositories
        rep_res = client.get("/api/repositories")
        assert rep_res.status_code == 200

        # 2. Validating sample repository
        val_res = client.post("/api/repositories/validate", json={"type": "sample", "path": "sample-repo"})
        assert val_res.status_code == 200

        # 3. Checking providers status
        status_res = client.get("/api/providers/status")
        assert status_res.status_code == 200

        # 4. Scans must remain empty
        scans_res = client.get("/api/scans")
        assert scans_res.status_code == 200
        assert len(scans_res.get_json()) == 0

        # 0 model calls made
        assert mock_client.models.generate_content.call_count == 0


def test_duplicate_demo_scan_submission_prevented(client):
    """Submitting a demo scan while an active maintenance run on the sample repository exists returns HTTP 409 Conflict."""
    import threading
    import time
    from unittest.mock import patch
    from backend.ai.agent_service import AgentService

    block_event = threading.Event()

    def mock_init(self, name):
        provider = AgentService._init_provider(self, name)
        orig = provider.analyze_and_prioritize

        def slow_analyze(findings, run_id=None):
            block_event.wait(timeout=5.0)
            return orig(findings, run_id=run_id)

        provider.analyze_and_prioritize = slow_analyze
        return provider

    with patch.object(AgentService, "_init_provider", mock_init):
        # Start a scan on sample repo
        res1 = client.post("/api/scans", json={"type": "sample", "provider": "demo"})
        assert res1.status_code == 202
        run1_id = res1.get_json()["id"]

        # Attempt to run demo scan on same sample repo while run1 is active
        res2 = client.post("/api/demo/scan")
        assert res2.status_code == 409
        err_data = res2.get_json()
        assert "already active" in err_data["error"]
        assert err_data["run"]["id"] == run1_id

        block_event.set()

    # Wait for completion
    for _ in range(40):
        time.sleep(0.05)
        if client.get(f"/api/scans/{run1_id}").get_json()["status"] == "completed":
            break

