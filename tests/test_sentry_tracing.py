"""Unit and integration tests for Sentry Agent Tracing instrumentation."""
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import sentry_sdk
from sentry_sdk.transport import Transport

from backend.ai.agent_service import AgentService
from backend.config import Config
from backend.models import FindingCategory, Severity
from backend.telemetry.tracing import (
    RunTracer,
    SpanContext,
    agent_stage_span,
    get_observability_status,
    init_sentry,
    is_sentry_active,
    maintenance_run_trace,
    sanitize_scalar,
    sanitize_telemetry_tags,
)


class InMemorySentryTransport(Transport):
    """Test transport collecting captured Sentry envelopes in memory."""

    def __init__(self):
        super().__init__()
        self.envelopes = []

    def capture_envelope(self, envelope):
        self.envelopes.append(envelope)


@pytest.fixture(autouse=True)
def reset_sentry():
    """Ensures Sentry SDK state is cleaned up before and after each test."""
    yield
    try:
        # Reset sentry client
        sentry_sdk.init(dsn="")
    except Exception:
        pass


def test_disabled_sentry_configuration():
    """When SENTRY_DSN is empty, observability should report not_configured and runs should succeed offline."""
    success = init_sentry(dsn="")
    assert success is False
    assert is_sentry_active() is False

    status = get_observability_status(dsn="", environment="development", sample_rate=1.0)
    assert status["status"] == "not_configured"
    assert status["configured"] is False

    # Execute a run: must succeed without errors despite Sentry being disabled
    agent = AgentService(provider_name="demo")
    run = agent.execute_maintenance_run(
        repo_path=Config.SAMPLE_REPO_PATH,
        repo_name="NovaTask (Sample Project)",
        repo_type="sample",
    )
    assert run.status == "completed"
    assert len(run.stages) >= 6


def test_root_maintenance_transaction_and_all_eight_child_spans():
    """A full maintenance run must emit a root transaction with all 8 distinct quietpatch child spans."""
    transport = InMemorySentryTransport()
    init_sentry(
        dsn="https://fakeKey@o0.ingest.sentry.io/0",
        environment="test",
        traces_sample_rate=1.0,
        transport=transport,
    )
    assert is_sentry_active() is True

    agent = AgentService(provider_name="demo")
    run = agent.execute_maintenance_run(
        repo_path=Config.SAMPLE_REPO_PATH,
        repo_name="NovaTask (Sample Project)",
        repo_type="sample",
        run_id="test-trace-run-01",
    )
    assert run.status == "completed"

    # Inspect captured Sentry transactions
    transactions = []
    for env in transport.envelopes:
        for item in env.items:
            if item.type == "transaction":
                transactions.append(item.payload.json)

    assert len(transactions) >= 1
    root_tx = next((t for t in transactions if t.get("contexts", {}).get("trace", {}).get("op") == "agent.maintenance_run"), None)
    assert root_tx is not None, "Root maintenance transaction 'agent.maintenance_run' was not emitted."

    # Verify transaction metadata and safe tags
    assert "QuietPatch Maintenance Run" in root_tx.get("transaction", "")
    trace_ctx = root_tx.get("contexts", {}).get("trace", {})
    tags = root_tx.get("tags", {})
    assert tags.get("run_id") == "test-trace-run-01"
    assert tags.get("repo_type") == "sample"
    assert tags.get("provider") == "demo"
    assert tags.get("stage_status") == "completed"

    # Verify all 8 distinct child spans
    spans = root_tx.get("spans", [])
    span_ops = [s.get("op") for s in spans]

    expected_ops = [
        "quietpatch.initialize",
        "quietpatch.repository_discovery",
        "quietpatch.deterministic_scan",
        "quietpatch.gemma_analysis",
        "quietpatch.prioritization",
        "quietpatch.patch_generation",
        "quietpatch.verification",
        "quietpatch.report_generation",
    ]

    for expected_op in expected_ops:
        assert expected_op in span_ops, f"Missing required child span op '{expected_op}'. Found ops: {span_ops}"

    # Verify all spans have descriptions and duration
    for s in spans:
        if s.get("op") in expected_ops:
            assert s.get("description", "").startswith("QuietPatch:")
            assert s.get("timestamp") >= s.get("start_timestamp")


def test_failed_stage_representation_in_tracing():
    """Failed AI analysis or invalid run must accurately report failure status in Sentry without false success."""
    transport = InMemorySentryTransport()
    init_sentry(
        dsn="https://fakeKey@o0.ingest.sentry.io/0",
        environment="test",
        traces_sample_rate=1.0,
        transport=transport,
    )

    agent = AgentService(provider_name="ollama")

    # Simulate an Ollama failure raising RuntimeError
    with patch.object(agent.provider, "analyze_and_prioritize", side_effect=RuntimeError("Ollama inference timed out after 240 seconds")):
        run = agent.execute_maintenance_run(
            repo_path=Config.SAMPLE_REPO_PATH,
            repo_name="NovaTask (Sample Project)",
            repo_type="sample",
            run_id="test-failed-trace-run",
        )

        assert run.stages["analysis"].status == "failed"
        assert run.stages["proposal"].status == "skipped"

    # Find the emitted transaction
    transactions = []
    for env in transport.envelopes:
        for item in env.items:
            if item.type == "transaction":
                transactions.append(item.payload.json)

    root_tx = next((t for t in transactions if t.get("tags", {}).get("run_id") == "test-failed-trace-run"), None)
    assert root_tx is not None

    spans = root_tx.get("spans", [])
    analysis_span = next((s in spans and s for s in spans if s.get("op") == "quietpatch.gemma_analysis"), None)
    assert analysis_span is not None
    # Failed stage must NOT be marked successful
    assert analysis_span.get("tags", {}).get("outcome") in {"failed", "failure"}
    assert analysis_span.get("status") in {"internal_error", "unknown_error"} or analysis_span.get("tags", {}).get("stage_status") == "failed"

    # Proposal span must be marked skipped
    proposal_span = next((s in spans and s for s in spans if s.get("op") == "quietpatch.patch_generation"), None)
    assert proposal_span is not None
    assert proposal_span.get("tags", {}).get("outcome") == "skipped"


def test_strict_telemetry_sanitization():
    """Sensitive keys, raw source code, diffs, API keys, and local absolute paths must never be included in telemetry."""
    dirty_tags = {
        "run_id": "run-safe-01",
        "repo_path": "/Users/harshu/Desktop/QuietPatch/private/source.py",
        "home_path": "/home/ubuntu/repo/app.py",
        "gemma_key": "AIzaSyABC1234567890secretKey",
        "api_token": "sk-proj-99998888777766665555444433332222",
        "raw_code": "def secret_function(): return 'classified'",
        "diff": "--- a/file.py\n+++ b/file.py\n+password = 'admin'",
        "prompt": "You are Gemma. Fix this vulnerability in user file.",
        "provider": "ollama",
        "model": "gemma4:e4b",
        "files_count": 8,
    }

    sanitized = sanitize_telemetry_tags(dirty_tags)

    # Prohibited keys must be stripped
    assert "repo_path" not in sanitized
    assert "home_path" not in sanitized
    assert "gemma_key" not in sanitized
    assert "api_token" not in sanitized
    assert "raw_code" not in sanitized
    assert "diff" not in sanitized
    assert "prompt" not in sanitized

    # Allowed keys preserved
    assert sanitized["run_id"] == "run-safe-01"
    assert sanitized["provider"] == "ollama"
    assert sanitized["model"] == "gemma4:e4b"
    assert sanitized["files_count"] == "8"

    # Scalar path scrubbing test
    assert "/Users/" not in sanitize_scalar("/Users/harshu/repo/secret.txt")
    assert "[REDACTED_PATH]" in sanitize_scalar("/Users/harshu/repo/secret.txt")
    assert "[REDACTED_KEY]" in sanitize_scalar("Bearer AIzaSy12345678901234567890")


def test_background_worker_tracing_in_thread():
    """A maintenance run executed in a background worker thread must emit the root transaction and all spans."""
    transport = InMemorySentryTransport()
    init_sentry(
        dsn="https://fakeKey@o0.ingest.sentry.io/0",
        environment="test",
        traces_sample_rate=1.0,
        transport=transport,
    )

    worker_run_result = []

    def _worker():
        agent = AgentService(provider_name="demo")
        run = agent.execute_maintenance_run(
            repo_path=Config.SAMPLE_REPO_PATH,
            repo_name="NovaTask (Sample Project)",
            repo_type="sample",
            run_id="worker-thread-run-01",
        )
        worker_run_result.append(run)

    thread = threading.Thread(target=_worker, name="test-bg-worker")
    thread.start()
    thread.join(timeout=15.0)

    assert len(worker_run_result) == 1
    assert worker_run_result[0].status == "completed"

    transactions = []
    for env in transport.envelopes:
        for item in env.items:
            if item.type == "transaction":
                transactions.append(item.payload.json)

    worker_tx = next((t for t in transactions if t.get("tags", {}).get("run_id") == "worker-thread-run-01"), None)
    assert worker_tx is not None, "Background worker thread did not emit root Sentry transaction."

    span_ops = [s.get("op") for s in worker_tx.get("spans", [])]
    assert "quietpatch.initialize" in span_ops
    assert "quietpatch.repository_discovery" in span_ops
    assert "quietpatch.deterministic_scan" in span_ops
    assert "quietpatch.gemma_analysis" in span_ops
    assert "quietpatch.prioritization" in span_ops
    assert "quietpatch.patch_generation" in span_ops
    assert "quietpatch.verification" in span_ops
    assert "quietpatch.report_generation" in span_ops
