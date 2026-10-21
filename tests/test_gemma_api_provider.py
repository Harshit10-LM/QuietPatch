"""Tests for Hosted Gemma AI Provider (Google Gen AI SDK), retries, payload bounding, and tracing."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors
from sentry_sdk.transport import Transport

from backend.ai.agent_service import AgentService
from backend.ai.gemma_api_provider import (
    GemmaApiError,
    GemmaApiProvider,
    GemmaApiTransientError,
    extract_error_code,
    is_transient_error,
    sanitize_error_message,
)
from backend.app import create_app
from backend.models import Finding, FindingCategory, PatchStatus, ProposedPatch, Severity
from backend.telemetry.tracing import init_sentry


class InMemorySentryTransport(Transport):
    """Test transport collecting captured Sentry envelopes in memory."""

    def __init__(self):
        super().__init__()
        self.envelopes = []

    def capture_envelope(self, envelope):
        self.envelopes.append(envelope)


def test_sanitize_error_message():
    raw = "Request failed for key AIzaSyD1234567890abcdefghijklmnopqrstuv at /Users/harshu/secret.py: Internal error"
    sanitized = sanitize_error_message(raw)
    assert "[REDACTED_API_KEY]" in sanitized
    assert "AIzaSyD" not in sanitized
    assert "/Users/harshu" not in sanitized
    assert "[REDACTED_PATH]" in sanitized


def test_is_transient_error_detection():
    assert is_transient_error(genai_errors.ServerError(500, "Internal Server Error")) is True
    assert is_transient_error(genai_errors.APIError(503, "Service Unavailable")) is True
    assert is_transient_error(genai_errors.APIError(429, "Rate limit exceeded")) is True
    assert is_transient_error(TimeoutError("Connection timed out")) is True

    # Permanent client errors should NOT be transient
    assert is_transient_error(genai_errors.ClientError(401, "Invalid API key")) is False
    assert is_transient_error(genai_errors.ClientError(400, "Malformed request")) is False
    assert is_transient_error(genai_errors.ClientError(403, "Permission Denied")) is False


def test_transient_500_retry_success():
    """Verifies that transient 500 error is retried and succeeds on second attempt."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "change_summary": "Pin requests version",
        "proposed_content": "requests==2.31.0\n",
        "explanation": "Pins unpinned dependency",
        "risks_assumptions": "Low risk",
        "suggested_verification": "pip check",
    })

    # Call 1: ServerError 500, Call 2: Success
    mock_client.models.generate_content.side_effect = [
        genai_errors.ServerError(500, "Google internal server error"),
        mock_response,
    ]

    provider = GemmaApiProvider(
        api_key="test-key",
        model="gemma-4-26b-a4b-it",
        max_retries=2,
        backoff_factor=0.001,  # fast test
    )
    provider._client = mock_client

    finding = Finding(
        id="find-01",
        category=FindingCategory.CONFIGURATION,
        title="Unpinned dependency",
        file_path="requirements.txt",
        evidence="requests",
        explanation="Dependency is unpinned",
        suggested_action="Pin requests",
        severity=Severity.MEDIUM,
    )

    transport = InMemorySentryTransport()
    init_sentry(dsn="https://fake@fake.ingest.sentry.io/123", transport=transport)

    patch_res = provider.propose_patch(
        finding=finding,
        file_content="requests\n",
        repo_path=Path("/tmp"),
        run_id="run-test-retry",
    )

    assert patch_res is not None
    assert patch_res.proposed_content == "requests==2.31.0\n"
    assert mock_client.models.generate_content.call_count == 2


def test_transient_500_exhausted_retries_reports_failure():
    """Verifies that exhausting retries marks the span as failed and returns None."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = genai_errors.ServerError(500, "Google 500 error")

    provider = GemmaApiProvider(
        api_key="test-key",
        model="gemma-4-26b-a4b-it",
        max_retries=2,
        backoff_factor=0.001,
    )
    provider._client = mock_client

    finding = Finding(
        id="find-02",
        category=FindingCategory.CONFIGURATION,
        title="Missing license",
        file_path="LICENSE",
        evidence="Missing",
        explanation="Repository lacks standard license",
        suggested_action="Add MIT license",
        severity=Severity.LOW,
    )

    transport = InMemorySentryTransport()
    init_sentry(dsn="https://fake@fake.ingest.sentry.io/123", transport=transport)

    patch_res = provider.propose_patch(
        finding=finding,
        file_content="",
        repo_path=Path("/tmp"),
        run_id="run-exhausted",
    )

    assert patch_res is None
    # 1 initial + 2 retries = 3 calls
    assert mock_client.models.generate_content.call_count == 3


def test_permanent_client_error_fails_immediately_without_retries():
    """Verifies that permanent errors (e.g. 401 Unauthorized) are not retried."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = genai_errors.ClientError(401, "API key not valid")

    provider = GemmaApiProvider(
        api_key="bad-key",
        model="gemma-4-26b-a4b-it",
        max_retries=3,
        backoff_factor=0.001,
    )
    provider._client = mock_client

    finding = Finding(
        id="find-03",
        category=FindingCategory.DOCUMENTATION,
        title="Broken link",
        file_path="README.md",
        evidence="http://broken.test",
        explanation="Documentation contains dead link",
        suggested_action="Fix link",
        severity=Severity.LOW,
    )

    patch_res = provider.propose_patch(
        finding=finding,
        file_content="# Readme\nBroken link\n",
        repo_path=Path("/tmp"),
        run_id="run-permanent-err",
    )

    assert patch_res is None
    # Must only be called ONCE (no retries for 401)
    assert mock_client.models.generate_content.call_count == 1


def test_bounded_payload_truncation():
    """Verifies that large files and evidence are bounded to avoid exceeding token limits."""
    mock_client = MagicMock()
    captured_contents = []

    def mock_generate(**kwargs):
        captured_contents.append(kwargs.get("contents", ""))
        resp = MagicMock()
        resp.text = json.dumps({
            "change_summary": "Test fix",
            "proposed_content": "short content\n",
        })
        return resp

    mock_client.models.generate_content.side_effect = mock_generate

    provider = GemmaApiProvider(
        api_key="test-key",
        model="gemma-4-26b-a4b-it",
    )
    provider._client = mock_client

    # 10,000 characters file content
    large_file_content = "line of code here\n" * 500
    finding = Finding(
        id="find-large",
        category=FindingCategory.CONFIGURATION,
        title="Very large target",
        file_path="big_file.py",
        evidence="line of code here\n" * 20,
        explanation="Large target explanation",
        suggested_action="Refactor",
        severity=Severity.LOW,
    )

    provider.propose_patch(
        finding=finding,
        file_content=large_file_content,
        repo_path=Path("/tmp"),
        run_id="run-payload-bound",
    )

    assert len(captured_contents) == 1
    prompt = captured_contents[0]
    # The full 10,000 chars should NOT be in prompt
    assert len(prompt) < 3500
    assert len(large_file_content) > 8000


def test_partial_patch_failure_in_maintenance_run(tmp_path):
    """Verifies that when some patch generations fail and others succeed, the proposal stage is marked partial."""
    f1 = tmp_path / "reqs.txt"
    f1.write_text("flask\n", encoding="utf-8")
    f2 = tmp_path / "README.md"
    f2.write_text("# Doc\n", encoding="utf-8")

    mock_provider = MagicMock()
    mock_provider.name = "gemma_api"
    mock_provider.model = "gemma-4-26b-a4b-it"

    # Analyze succeeds
    def mock_analyze(findings, run_id=None):
        for f in findings:
            f.ai_assisted = True
            f.is_demo = False
        return findings
    mock_provider.analyze_and_prioritize.side_effect = mock_analyze

    # Propose patch: first finding succeeds, second finding raises ServerError
    patch1 = ProposedPatch(
        id="patch-f1",
        finding_id="f1",
        file_path="reqs.txt",
        change_summary="Pin flask",
        original_content="flask\n",
        proposed_content="flask==3.0.0\n",
        diff="--- a/reqs.txt\n+++ b/reqs.txt\n@@ -1 +1 @@\n-flask\n+flask==3.0.0\n",
        explanation="Fix",
        risks_assumptions="None",
        suggested_verification="pytest",
        status=PatchStatus.PENDING,
    )

    def mock_propose(finding, file_content, repo_path, run_id=None):
        if finding.file_path == "reqs.txt":
            return patch1
        # Second finding fails
        raise genai_errors.ServerError(500, "Google 500 error")

    mock_provider.propose_patch.side_effect = mock_propose

    agent = AgentService(provider_name="gemma_api")
    agent.provider = mock_provider

    finding1 = Finding(
        id="f1",
        category=FindingCategory.CONFIGURATION,
        title="Flask unpinned",
        file_path="reqs.txt",
        evidence="flask",
        explanation="Unpinned flask",
        suggested_action="Pin flask",
        severity=Severity.HIGH,
    )
    finding2 = Finding(
        id="f2",
        category=FindingCategory.DOCUMENTATION,
        title="Doc issue",
        file_path="README.md",
        evidence="doc",
        explanation="Doc link issue",
        suggested_action="Fix doc",
        severity=Severity.LOW,
    )

    with patch.object(agent.scan_engine, "discover_files", return_value=([f1, f2], 20)), \
         patch.object(agent.scan_engine, "run_scanners", return_value=([finding1, finding2], {})):

        run = agent.execute_maintenance_run(
            repo_path=tmp_path,
            repo_name="PartialTestRepo",
            repo_type="local",
            run_id="run-partial-test",
        )

    # Workflow completed without crashing
    assert run.status == "completed"
    assert len(run.patches) == 1
    # Stage outcome was partial, not completed or skipped
    assert run.stages["proposal"].status == "partial"
    assert "1/2 patches" in run.stages["proposal"].details


def test_api_error_handler_returns_sanitized_json():
    """Verifies that Flask global error handlers return sanitized JSON instead of HTML on 500 and transient AI errors."""
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    # 1. Test 404 returns JSON with error key
    res_404 = client.get("/api/definitely-not-an-endpoint")
    assert res_404.status_code == 404
    data_404 = res_404.get_json()
    assert "error" in data_404

    # 2. Pre-populate via demo scan so runs_store has an active run & finding
    scan_res = client.post("/api/demo/scan")
    assert scan_res.status_code == 200
    scan_data = scan_res.get_json()
    assert len(scan_data["findings"]) > 0
    target_finding_id = scan_data["findings"][0]["id"]

    # 3. Test propose-fix when provider raises transient Gemma error -> 503 JSON
    with patch("backend.ai.demo_provider.DemoProvider.propose_patch", side_effect=GemmaApiTransientError("Upstream Google 500 error", code=500)):
        res_fix = client.post(f"/api/findings/{target_finding_id}/propose-fix")
        assert res_fix.status_code == 503
        data_fix = res_fix.get_json()
        assert data_fix is not None
        assert "temporarily unavailable" in data_fix["error"]
        assert data_fix.get("error_type") == "upstream_transient_error"

    # 4. Test propose-fix when provider raises permanent Gemma error -> 502 JSON
    with patch("backend.ai.demo_provider.DemoProvider.propose_patch", side_effect=GemmaApiError("Permanent API failure", code=400)):
        res_err = client.post(f"/api/findings/{target_finding_id}/propose-fix")
        assert res_err.status_code == 502
        data_err = res_err.get_json()
        assert data_err is not None
        assert "Permanent API failure" in data_err["error"]
        assert data_err.get("error_type") == "upstream_error"
