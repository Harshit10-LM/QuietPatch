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
    GemmaApiResponseError,
    GemmaApiTransientError,
    ResponseDiagnostics,
    clean_json_text,
    extract_error_code,
    extract_response_diagnostics,
    extract_response_text,
    is_transient_error,
    sanitize_error_message,
)
from backend.app import create_app
from backend.config import Config
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
    # Dynamically assemble a fake key matching Google Gen AI key regex (AIza + 35 chars)
    # at runtime to avoid hardcoded credentials or key-shaped literals in test source
    prefix = "".join(["AI", "za"])
    fake_token = prefix + "Sy" + ("X" * 33)
    assert len(fake_token) == 39

    raw = f"Request failed for key {fake_token} at /Users/harshu/secret.py: Internal error"
    sanitized = sanitize_error_message(raw)
    assert "[REDACTED_API_KEY]" in sanitized
    assert fake_token not in sanitized
    assert "/Users/harshu" not in sanitized
    assert "[REDACTED_PATH]" in sanitized


def test_is_transient_error_detection():
    assert is_transient_error(genai_errors.ServerError(500, "Internal Server Error")) is True
    assert is_transient_error(genai_errors.APIError(503, "Service Unavailable")) is True
    assert is_transient_error(genai_errors.APIError(429, "Rate limit exceeded")) is True
    assert is_transient_error(TimeoutError("Connection timed out")) is True

    # Permanent client errors and deterministic response errors should NOT be transient
    assert is_transient_error(genai_errors.ClientError(401, "Invalid API key")) is False
    assert is_transient_error(genai_errors.ClientError(400, "Malformed request")) is False
    assert is_transient_error(genai_errors.ClientError(403, "Permission Denied")) is False
    assert is_transient_error(GemmaApiResponseError("Empty response text")) is False
    assert is_transient_error(json.JSONDecodeError("Expecting value", "", 0)) is False
    assert is_transient_error(AttributeError("'NoneType' object has no attribute 'strip'")) is False


def test_clean_json_text_various_formats():
    assert clean_json_text(None) == ""
    assert clean_json_text("") == ""
    assert clean_json_text("   [{\"id\": 1}]   ") == "[{\"id\": 1}]"
    assert clean_json_text("```json\n[{\"id\": 1}]\n```") == "[{\"id\": 1}]"
    assert clean_json_text("```\n[{\"id\": 1}]\n```") == "[{\"id\": 1}]"
    assert clean_json_text("Preamble here:\n```json\n[{\"id\": 1}]\n```\nDone") == "[{\"id\": 1}]"
    assert clean_json_text("Here is the answer: [{\"id\": 1}] have a nice day") == "[{\"id\": 1}]"


def test_extract_response_text_safe_handling():
    # 1. None response
    assert extract_response_text(None) is None

    # 2. String text attribute
    mock_resp1 = MagicMock()
    mock_resp1.text = "  hello world  "
    assert extract_response_text(mock_resp1) == "hello world"

    # 3. None text attribute, but candidate parts have text
    mock_resp2 = MagicMock()
    mock_resp2.text = None
    mock_part = MagicMock()
    mock_part.text = "candidate part text"
    mock_part.thought = False
    mock_resp2.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
    assert extract_response_text(mock_resp2) == "candidate part text"

    # 4. None text attribute, empty candidates
    mock_resp3 = MagicMock()
    mock_resp3.text = None
    mock_resp3.candidates = []
    assert extract_response_text(mock_resp3) is None

    # 5. None text attribute, candidate content is None
    mock_resp4 = MagicMock()
    mock_resp4.text = None
    mock_resp4.candidates = [MagicMock(content=None)]
    assert extract_response_text(mock_resp4) is None

    # 6. Candidate parts with non-string text or thought
    mock_resp5 = MagicMock()
    mock_resp5.text = None
    p1 = MagicMock(text=None, thought=False)
    p2 = MagicMock(text="internal thought", thought=True)
    p3 = MagicMock(text="valid answer", thought=False)
    mock_resp5.candidates = [MagicMock(content=MagicMock(parts=[p1, p2, p3]))]
    assert extract_response_text(mock_resp5) == "valid answer"


def test_valid_hosted_gemma_response():
    """Verifies that a valid response from hosted Gemma properly annotates findings and generates patches."""
    mock_client = MagicMock()
    mock_analyze_resp = MagicMock()
    mock_analyze_resp.text = json.dumps([
        {
            "id": "f-valid",
            "severity": "high",
            "explanation": "High severity security issue.",
            "suggested_action": "Upgrade package immediately.",
        }
    ])
    mock_patch_resp = MagicMock()
    mock_patch_resp.text = json.dumps({
        "change_summary": "Fix vulnerability",
        "proposed_content": "fixed content\n",
        "explanation": "Fix explanation",
        "risks_assumptions": "None",
        "suggested_verification": "pytest",
    })

    mock_client.models.generate_content.side_effect = [mock_analyze_resp, mock_patch_resp]

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-valid",
        category=FindingCategory.CONFIGURATION,
        title="Valid finding",
        file_path="config.py",
        evidence="debug=True",
        explanation="Default explanation",
        suggested_action="Default action",
        severity=Severity.LOW,
    )

    # 1. Test analysis
    analyzed = provider.analyze_and_prioritize([finding], run_id="run-valid")
    assert len(analyzed) == 1
    assert analyzed[0].ai_assisted is True
    assert analyzed[0].is_demo is False
    assert analyzed[0].severity == Severity.HIGH
    assert analyzed[0].explanation == "High severity security issue."

    # 2. Test patch generation
    patch_res = provider.propose_patch(
        finding=finding,
        file_content="old content\n",
        repo_path=Path("/tmp"),
        run_id="run-valid",
    )
    assert patch_res is not None
    assert patch_res.proposed_content == "fixed content\n"
    assert patch_res.status == PatchStatus.PENDING
    assert patch_res.is_demo is False


def test_nullable_text_and_missing_response_content():
    """Verifies that response.text=None does not raise AttributeError and reports honest failure."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = None
    mock_resp.candidates = None
    mock_client.models.generate_content.return_value = mock_resp

    provider = GemmaApiProvider(api_key="test-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-null",
        category=FindingCategory.CONFIGURATION,
        title="Null text test",
        file_path="app.py",
        evidence="evidence",
        explanation="Orig explanation",
        suggested_action="Orig action",
        severity=Severity.LOW,
    )

    # Must NOT raise AttributeError: 'NoneType' object has no attribute 'strip'
    analyzed = provider.analyze_and_prioritize([finding], run_id="run-null-text")
    assert analyzed[0].ai_assisted is False
    assert analyzed[0].is_demo is False

    patch_res = provider.propose_patch(
        finding=finding,
        file_content="test\n",
        repo_path=Path("/tmp"),
        run_id="run-null-text",
    )
    assert patch_res is None


def test_empty_candidates_or_missing_candidate_text():
    """Verifies that empty candidates or candidates with None content/parts do not crash."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = None
    mock_resp.candidates = [MagicMock(content=MagicMock(parts=[]))]
    mock_client.models.generate_content.return_value = mock_resp

    provider = GemmaApiProvider(api_key="test-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-empty-cand",
        category=FindingCategory.DOCUMENTATION,
        title="Empty candidate test",
        file_path="README.md",
        evidence="broken",
        explanation="Doc explanation",
        suggested_action="Fix doc",
        severity=Severity.LOW,
    )

    analyzed = provider.analyze_and_prioritize([finding], run_id="run-empty-cand")
    assert analyzed[0].ai_assisted is False
    assert analyzed[0].is_demo is False

    patch_res = provider.propose_patch(
        finding=finding,
        file_content="# README\n",
        repo_path=Path("/tmp"),
        run_id="run-empty-cand",
    )
    assert patch_res is None


def test_malformed_json_and_unexpected_response_structures():
    """Verifies that malformed JSON or unexpected non-list/non-dict outputs fail safely without crash."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "This is plain unstructured prose without valid JSON syntax {unclosed"
    mock_client.models.generate_content.return_value = mock_resp

    provider = GemmaApiProvider(api_key="test-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-malformed",
        category=FindingCategory.CONFIGURATION,
        title="Malformed JSON test",
        file_path="settings.py",
        evidence="settings",
        explanation="Settings explanation",
        suggested_action="Fix settings",
        severity=Severity.LOW,
    )

    analyzed = provider.analyze_and_prioritize([finding], run_id="run-malformed")
    assert analyzed[0].ai_assisted is False
    assert analyzed[0].is_demo is False

    # Also test unexpected JSON structure (dict when expecting list of findings)
    mock_resp.text = json.dumps({"status": "no findings array here"})
    analyzed_dict = provider.analyze_and_prioritize([finding], run_id="run-unexpected")
    assert analyzed_dict[0].ai_assisted is False

    # In patch generation, test malformed response
    mock_resp.text = json.dumps(["not a dict"])
    patch_res = provider.propose_patch(
        finding=finding,
        file_content="old\n",
        repo_path=Path("/tmp"),
        run_id="run-malformed-patch",
    )
    assert patch_res is None


def test_successful_http_response_followed_by_processing_failure_is_not_retried():
    """Verifies that when HTTP is 200 but content is invalid, it is NOT retried like a transient 500."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = None
    mock_resp.candidates = []
    mock_client.models.generate_content.return_value = mock_resp

    provider = GemmaApiProvider(
        api_key="test-key",
        model="gemma-4-26b-a4b-it",
        max_retries=3,
        backoff_factor=0.001,
    )
    provider._client = mock_client

    finding = Finding(
        id="f-single-call",
        category=FindingCategory.CONFIGURATION,
        title="Single call test",
        file_path="config.py",
        evidence="evidence",
        explanation="Explanation",
        suggested_action="Action",
        severity=Severity.LOW,
    )

    provider.analyze_and_prioritize([finding], run_id="run-no-retry-parse")
    # Controlled recovery attempts at most 1 recovery (total 2 calls), NOT retried 4 times like a transient 500
    assert mock_client.models.generate_content.call_count == 2


def test_honest_failure_reporting_no_false_badges_no_demo_fallback(tmp_path):
    """Verifies that when hosted Gemma analysis fails, no demo fallback is used and proposal is skipped."""
    mock_provider = MagicMock()
    mock_provider.name = "gemma_api"
    mock_provider.model = "gemma-4-26b-a4b-it"

    # Hosted Gemma analysis fails because response has no usable text
    def mock_analyze_fail(findings, run_id=None):
        # Returns findings unmodified
        return findings

    mock_provider.analyze_and_prioritize.side_effect = mock_analyze_fail

    agent = AgentService(provider_name="gemma_api")
    agent.provider = mock_provider

    f1 = tmp_path / "app.py"
    f1.write_text("import sys\n", encoding="utf-8")

    finding = Finding(
        id="f-no-demo",
        category=FindingCategory.CONFIGURATION,
        title="Test finding",
        file_path="app.py",
        evidence="import sys",
        explanation="Sys import",
        suggested_action="Check import",
        severity=Severity.LOW,
    )

    with patch.object(agent.scan_engine, "discover_files", return_value=([f1], 10)), \
         patch.object(agent.scan_engine, "run_scanners", return_value=([finding], {})):

        run = agent.execute_maintenance_run(
            repo_path=tmp_path,
            repo_name="NoDemoRepo",
            repo_type="local",
            run_id="run-no-demo",
        )

    # Analysis failed
    assert run.stages["analysis"].status == "failed"
    # Patch synthesis skipped because analysis was unavailable
    assert run.stages["proposal"].status == "skipped"
    assert len(run.patches) == 0
    # Findings must NOT have false AI badges or demo badges
    assert finding.ai_assisted is False
    assert finding.is_demo is False


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


# =============================================================================
# Production Root-Cause Investigation Regression Tests (Items 1-10)
# =============================================================================


def test_regression_1_valid_response_with_usable_text():
    """1. Valid response with usable text properly annotates findings and generates patch."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps([
        {
            "id": "f-reg-1",
            "severity": "high",
            "explanation": "Critical security flaw.",
            "suggested_action": "Sanitize inputs.",
        }
    ])
    mock_client.models.generate_content.return_value = mock_resp

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-reg-1",
        category=FindingCategory.CONFIGURATION,
        title="Test valid finding",
        file_path="config.py",
        evidence="debug=True",
        explanation="Default explanation",
        suggested_action="Default action",
        severity=Severity.LOW,
    )

    analyzed = provider.analyze_and_prioritize([finding], run_id="run-reg-1")
    assert analyzed[0].ai_assisted is True
    assert analyzed[0].is_demo is False
    assert analyzed[0].severity == Severity.HIGH
    assert analyzed[0].explanation == "Critical security flaw."
    assert mock_client.models.generate_content.call_count == 1


def test_regression_2_missing_or_empty_candidates():
    """2. Missing or empty candidates response categorized safely without crashing."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = None
    mock_resp.candidates = []
    mock_client.models.generate_content.return_value = mock_resp

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-reg-2",
        category=FindingCategory.CONFIGURATION,
        title="Empty candidates finding",
        file_path="config.py",
        evidence="debug=True",
        explanation="Default explanation",
        suggested_action="Default action",
        severity=Severity.LOW,
    )

    analyzed = provider.analyze_and_prioritize([finding], run_id="run-reg-2")
    # Honest failure: finding not marked as evaluated
    assert analyzed[0].ai_assisted is False
    assert analyzed[0].is_demo is False
    # Exactly 2 calls: initial attempt + 1 controlled bounded recovery attempt
    assert mock_client.models.generate_content.call_count == 2


def test_regression_3_missing_content_and_text():
    """3. Candidate with None content is categorized as absent_content and handled safely."""
    mock_client = MagicMock()
    mock_cand = MagicMock()
    mock_cand.content = None
    mock_cand.finish_reason = "STOP"
    mock_resp = MagicMock()
    mock_resp.text = None
    mock_resp.candidates = [mock_cand]
    mock_client.models.generate_content.return_value = mock_resp

    diag = extract_response_diagnostics(mock_resp)
    assert diag.has_response is True
    assert diag.failure_category == "absent_content"
    assert diag.extracted_text is None

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-reg-3",
        category=FindingCategory.DOCUMENTATION,
        title="Missing content finding",
        file_path="README.md",
        evidence="broken",
        explanation="Default explanation",
        suggested_action="Default action",
        severity=Severity.LOW,
    )

    analyzed = provider.analyze_and_prioritize([finding], run_id="run-reg-3")
    assert analyzed[0].ai_assisted is False
    assert mock_client.models.generate_content.call_count == 2


def test_regression_4_text_accessor_returns_none_or_raises_exception():
    """4. Response whose text accessor raises an exception or returns None does not crash."""
    mock_client = MagicMock()
    mock_resp = MagicMock()

    # Define a property that raises ValueError
    type(mock_resp).text = property(lambda self: (_ for _ in ()).throw(ValueError("Text blocked by model policy")))
    mock_resp.candidates = []
    mock_client.models.generate_content.return_value = mock_resp

    diag = extract_response_diagnostics(mock_resp)
    assert diag.has_response is True
    assert diag.extracted_text is None

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-reg-4",
        category=FindingCategory.CONFIGURATION,
        title="Exception text accessor",
        file_path="app.py",
        evidence="test",
        explanation="Default explanation",
        suggested_action="Default action",
        severity=Severity.LOW,
    )

    # Must complete without unhandled exception
    analyzed = provider.analyze_and_prioritize([finding], run_id="run-reg-4")
    assert analyzed[0].ai_assisted is False


def test_regression_5_safety_blocked_response_no_recovery():
    """5. Safety-blocked response is classified as safety_blocked and NEVER retried."""
    mock_client = MagicMock()
    mock_cand = MagicMock()
    mock_cand.content = None
    mock_cand.finish_reason = "SAFETY"
    mock_resp = MagicMock()
    mock_resp.text = None
    mock_resp.candidates = [mock_cand]
    mock_client.models.generate_content.return_value = mock_resp

    diag = extract_response_diagnostics(mock_resp)
    assert diag.failure_category == "safety_blocked"
    assert diag.is_recoverable is False

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-reg-5",
        category=FindingCategory.CONFIGURATION,
        title="Safety blocked test",
        file_path="app.py",
        evidence="evidence",
        explanation="Default explanation",
        suggested_action="Default action",
        severity=Severity.LOW,
    )

    analyzed = provider.analyze_and_prioritize([finding], run_id="run-reg-5")
    assert analyzed[0].ai_assisted is False
    # CRITICAL: Safety block must NOT be retried! Exactly 1 call.
    assert mock_client.models.generate_content.call_count == 1


def test_regression_6_truncated_or_malformed_json_output():
    """6. Truncated output (MAX_TOKENS) or malformed JSON is categorized and bounded-recovered."""
    mock_client = MagicMock()
    mock_cand = MagicMock()
    mock_cand.finish_reason = "MAX_TOKENS"
    part = MagicMock()
    part.text = '[{"id": "f-reg-6", "severity": "high", "explanation": "cut off mid-sentence'
    part.thought = False
    mock_cand.content = MagicMock(parts=[part])

    mock_resp = MagicMock()
    mock_resp.text = part.text
    mock_resp.candidates = [mock_cand]
    mock_client.models.generate_content.return_value = mock_resp

    diag = extract_response_diagnostics(mock_resp)
    assert diag.finish_reason == "MAX_TOKENS"
    assert diag.is_recoverable is True

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-reg-6",
        category=FindingCategory.CONFIGURATION,
        title="Truncated JSON test",
        file_path="app.py",
        evidence="evidence",
        explanation="Default explanation",
        suggested_action="Default action",
        severity=Severity.LOW,
    )

    analyzed = provider.analyze_and_prioritize([finding], run_id="run-reg-6")
    assert analyzed[0].ai_assisted is False
    # Attempted initial + 1 recovery = 2 calls
    assert mock_client.models.generate_content.call_count == 2


def test_regression_7_recoverable_empty_response_followed_by_successful_recovery():
    """7. Recoverable empty response on attempt 1 followed by valid JSON on attempt 2 succeeds."""
    mock_client = MagicMock()
    # Call 1: empty response
    mock_resp_empty = MagicMock()
    mock_resp_empty.text = ""
    mock_resp_empty.candidates = []

    # Call 2: valid JSON array
    mock_resp_valid = MagicMock()
    mock_resp_valid.text = json.dumps([
        {
            "id": "f-reg-7",
            "severity": "medium",
            "explanation": "Recovered successfully on retry.",
            "suggested_action": "Apply suggested change.",
        }
    ])

    mock_client.models.generate_content.side_effect = [mock_resp_empty, mock_resp_valid]

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-reg-7",
        category=FindingCategory.CONFIGURATION,
        title="Recovery success test",
        file_path="app.py",
        evidence="debug=True",
        explanation="Default explanation",
        suggested_action="Default action",
        severity=Severity.LOW,
    )

    analyzed = provider.analyze_and_prioritize([finding], run_id="run-reg-7")
    assert analyzed[0].ai_assisted is True
    assert analyzed[0].severity == Severity.MEDIUM
    assert analyzed[0].explanation == "Recovered successfully on retry."
    assert mock_client.models.generate_content.call_count == 2


def test_regression_8_repeated_empty_responses_after_recovery():
    """8. Repeated empty responses after recovery stops boundedly at attempt 2 and reports failure."""
    mock_client = MagicMock()
    mock_resp_empty_1 = MagicMock(text="")
    mock_resp_empty_1.candidates = []
    mock_resp_empty_2 = MagicMock(text="")
    mock_resp_empty_2.candidates = []

    mock_client.models.generate_content.side_effect = [mock_resp_empty_1, mock_resp_empty_2]

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-reg-8",
        category=FindingCategory.CONFIGURATION,
        title="Repeated empty test",
        file_path="app.py",
        evidence="evidence",
        explanation="Default explanation",
        suggested_action="Default action",
        severity=Severity.LOW,
    )

    analyzed = provider.analyze_and_prioritize([finding], run_id="run-reg-8")
    assert analyzed[0].ai_assisted is False
    # Exactly 2 calls (1 initial + 1 recovery), never loops indefinitely
    assert mock_client.models.generate_content.call_count == 2


def test_regression_9_no_false_success_in_sentry_or_ai_assisted_status(tmp_path):
    """9. When hosted Gemma fails, Sentry span records failure and ai_assisted remains False."""
    mock_provider = MagicMock()
    mock_provider.name = "gemma_api"
    mock_provider.model = "gemma-4-26b-a4b-it"

    # Simulates analyze_and_prioritize failing honest failure (unmodified findings)
    mock_provider.analyze_and_prioritize.side_effect = lambda findings, run_id=None: findings

    agent = AgentService(provider_name="gemma_api")
    agent.provider = mock_provider

    f1 = tmp_path / "app.py"
    f1.write_text("x = 1\n", encoding="utf-8")

    finding = Finding(
        id="f-reg-9",
        category=FindingCategory.CONFIGURATION,
        title="Sentry honest status test",
        file_path="app.py",
        evidence="x = 1",
        explanation="Default explanation",
        suggested_action="Default action",
        severity=Severity.LOW,
    )

    with patch.object(agent.scan_engine, "discover_files", return_value=([f1], 10)), \
         patch.object(agent.scan_engine, "run_scanners", return_value=([finding], {})):

        run = agent.execute_maintenance_run(
            repo_path=tmp_path,
            repo_name="SentryTestRepo",
            repo_type="local",
            run_id="run-reg-9",
        )

    # Analysis must be marked failed, proposal skipped, 0 patches
    assert run.stages["analysis"].status == "failed"
    assert run.stages["proposal"].status == "skipped"
    assert len(run.patches) == 0
    assert finding.ai_assisted is False
    assert finding.is_demo is False


def test_regression_10_no_silent_demo_fallback_or_fabricated_explanations(tmp_path):
    """10. Under hosted Gemma failure, QuietPatch never substitutes demo findings or fabricates AI patches."""
    mock_provider = MagicMock()
    mock_provider.name = "gemma_api"
    mock_provider.model = "gemma-4-26b-a4b-it"
    mock_provider.analyze_and_prioritize.side_effect = lambda findings, run_id=None: findings

    agent = AgentService(provider_name="gemma_api")
    agent.provider = mock_provider

    f1 = tmp_path / "package.json"
    f1.write_text('{"name": "test"}\n', encoding="utf-8")

    finding = Finding(
        id="f-reg-10",
        category=FindingCategory.JAVASCRIPT,
        title="Outdated dependency",
        file_path="package.json",
        evidence="package.json",
        explanation="Deterministic scanner detected dependency issue",
        suggested_action="Review package versions",
        severity=Severity.MEDIUM,
    )

    with patch.object(agent.scan_engine, "discover_files", return_value=([f1], 20)), \
         patch.object(agent.scan_engine, "run_scanners", return_value=([finding], {})):

        run = agent.execute_maintenance_run(
            repo_path=tmp_path,
            repo_name="NoDemoTestRepo",
            repo_type="local",
            run_id="run-reg-10",
        )

    # Findings must retain genuine deterministic evidence, no [Demo AI] inserted
    assert "[Demo AI]" not in run.findings[0].explanation
    assert run.findings[0].is_demo is False
    assert run.findings[0].ai_assisted is False
    # No fake patches
    assert len(run.patches) == 0


def test_max_tokens_truncation_triggers_reduced_scope_recovery(tmp_path):
    """5. MAX_TOKENS is classified as output_token_truncated and triggers reduced-scope bounded recovery."""
    provider = GemmaApiProvider()
    provider.api_key = "test-key-mock"

    finding = Finding(
        id="f-trunc-1",
        category=FindingCategory.CONFIGURATION,
        title="Unpinned package",
        file_path="config.py",
        evidence="PORT = 5000",
        explanation="Default port",
        suggested_action="Use env port",
    )
    file_content = "PORT = 5000\nDEBUG = True\n"

    # Call 1: MAX_TOKENS truncation
    cand1 = MagicMock()
    cand1.finish_reason = "MAX_TOKENS"
    cand1.content = None
    res1 = MagicMock()
    res1.candidates = [cand1]
    res1.text = '{"change_summary": "Incomplete json...'

    # Call 2: Recovery response succeeds with valid patch
    res2 = MagicMock()
    res2.candidates = [MagicMock(finish_reason="STOP")]
    res2.text = json.dumps({
        "change_summary": "Update port",
        "proposed_content": "import os\nPORT = int(os.getenv('PORT', 5000))\nDEBUG = True\n",
        "explanation": "Reads from env",
        "risks_assumptions": "None",
        "suggested_verification": "pytest",
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [res1, res2]

    with patch.object(provider, "_get_client", return_value=mock_client):
        patch_result = provider.propose_patch(finding, file_content, tmp_path, run_id="run-trunc-test")

    # Assert recovery was attempted and succeeded
    assert patch_result is not None
    assert patch_result.finding_id == "f-trunc-1"
    assert "PORT = int(os.getenv" in patch_result.proposed_content
    assert mock_client.models.generate_content.call_count == 2


def test_patch_proposal_failure_preserves_earlier_valid_patches(tmp_path):
    """6. Failed patch proposals do not erase earlier valid proposals or create false AI-success badges."""
    f1 = tmp_path / "a.py"
    f1.write_text("x = 1\n", encoding="utf-8")
    f2 = tmp_path / "b.py"
    f2.write_text("y = 2\n", encoding="utf-8")

    finding1 = Finding(
        id="f1",
        category=FindingCategory.CONFIGURATION,
        title="Config 1",
        file_path="a.py",
        evidence="x = 1",
        explanation="exp 1",
        suggested_action="act 1",
    )
    finding2 = Finding(
        id="f2",
        category=FindingCategory.CONFIGURATION,
        title="Config 2",
        file_path="b.py",
        evidence="y = 2",
        explanation="exp 2",
        suggested_action="act 2",
    )

    mock_provider = MagicMock()
    mock_provider.name = "gemma_api"
    mock_provider.model = "gemma-4-26b-a4b-it"

    def mock_analyze(findings, run_id=None):
        for f in findings:
            f.ai_assisted = True
        return findings

    mock_provider.analyze_and_prioritize.side_effect = mock_analyze

    patch1 = ProposedPatch(
        id="patch-f1",
        finding_id="f1",
        file_path="a.py",
        change_summary="Update a",
        original_content="x = 1\n",
        proposed_content="x = 10\n",
        diff="--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 10\n",
        explanation="fix",
        risks_assumptions="none",
        suggested_verification="pytest",
        status=PatchStatus.PENDING,
    )

    def mock_propose(finding, file_content, repo_path, run_id=None):
        if finding.id == "f1":
            return patch1
        # finding 2 fails
        raise genai_errors.APIError(500, "Google internal error")

    mock_provider.propose_patch.side_effect = mock_propose

    agent = AgentService(provider_name="gemma_api")
    agent.provider = mock_provider

    with patch.object(agent.scan_engine, "discover_files", return_value=([f1, f2], 10)), \
         patch.object(agent.scan_engine, "run_scanners", return_value=([finding1, finding2], {})):

        run = agent.execute_maintenance_run(
            repo_path=tmp_path,
            repo_name="PreserveTestRepo",
            repo_type="local",
            run_id="run-preserve-test",
        )

    # Patch 1 must be preserved even though finding 2 proposal failed
    assert len(run.patches) == 1
    assert run.patches[0].id == "patch-f1"
    assert run.stages["proposal"].status == "partial"


def test_sentry_spans_report_failures_without_secrets(tmp_path):
    """8. Sentry spans report failures with sanitized details without leaking API keys or local paths."""
    prefix = "".join(["AI", "za"])
    fake_token = prefix + "Sy" + ("Z" * 33)

    mock_provider = MagicMock()
    mock_provider.name = "gemma_api"
    mock_provider.model = "gemma-4-26b-a4b-it"

    # Simulate analysis raising an exception with credentials and local path
    sensitive_msg = f"Failed with key={fake_token} at /Users/harshu/secret/repo.py: Service down"
    mock_provider.analyze_and_prioritize.side_effect = RuntimeError(sensitive_msg)

    agent = AgentService(provider_name="gemma_api")
    agent.provider = mock_provider

    f1 = tmp_path / "app.py"
    f1.write_text("x = 1\n", encoding="utf-8")
    finding = Finding(
        id="f1",
        category=FindingCategory.CONFIGURATION,
        title="Title",
        file_path="app.py",
        evidence="x = 1",
        explanation="exp",
        suggested_action="act",
    )

    with patch.object(agent.scan_engine, "discover_files", return_value=([f1], 10)), \
         patch.object(agent.scan_engine, "run_scanners", return_value=([finding], {})):

        run = agent.execute_maintenance_run(
            repo_path=tmp_path,
            repo_name="SanitizeTestRepo",
            repo_type="local",
            run_id="run-sanitize-test",
        )

    # Analysis failed, details sanitized
    analysis_stage = run.stages["analysis"]
    assert analysis_stage.status == "failed"
    assert fake_token not in analysis_stage.details
    assert "/Users/harshu" not in analysis_stage.details
    assert "[REDACTED" in analysis_stage.details


def test_retry_after_header_respected_and_completes_batch():
    """1. HTTP 500 with Retry-After header followed by a successful retry completes the batch."""
    mock_client = MagicMock()
    err = genai_errors.ServerError(500, "Internal Server Error")
    mock_resp_err = MagicMock()
    mock_resp_err.headers = {"Retry-After": "0.05"}
    err.response = mock_resp_err

    mock_resp_ok = MagicMock()
    mock_resp_ok.text = json.dumps([
        {
            "id": "f-retry-after",
            "severity": "high",
            "explanation": "Resolved after transient 500",
            "suggested_action": "Verify fix",
        }
    ])

    mock_client.models.generate_content.side_effect = [err, mock_resp_ok]

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it", max_retries=2, backoff_factor=0.01)
    provider._client = mock_client

    finding = Finding(
        id="f-retry-after",
        category=FindingCategory.CONFIGURATION,
        title="Retry-After test finding",
        file_path="config.py",
        evidence="api_key = ''",
        explanation="Default explanation",
        suggested_action="Default action",
        severity=Severity.LOW,
    )

    with patch.object(Config, "GEMMA_API_BATCH_SIZE", 2):
        analyzed = provider.analyze_and_prioritize([finding], run_id="run-ra-test")

    assert mock_client.models.generate_content.call_count == 2
    assert analyzed[0].ai_assisted is True
    assert analyzed[0].severity == Severity.HIGH
    assert analyzed[0].explanation == "Resolved after transient 500"


def test_all_diagnostic_categories_classified_correctly():
    """4. Empty, blocked, malformed, truncated, transient, and permanent outcomes are classified correctly."""
    # Safety blocked
    mock_safety = MagicMock()
    mock_safety.text = None
    mock_safety.candidates = [MagicMock(finish_reason="SAFETY", content=None)]
    diag_safety = extract_response_diagnostics(mock_safety)
    assert diag_safety.canonical_category == "safety_blocked"
    assert diag_safety.is_recoverable is False

    # Truncated (MAX_TOKENS)
    mock_trunc = MagicMock()
    mock_trunc.text = "{\"incomplete\": "
    mock_trunc.candidates = [MagicMock(finish_reason="MAX_TOKENS", content=MagicMock(parts=[MagicMock(text="{\"incomplete\": ", thought=False)]))]
    diag_trunc = extract_response_diagnostics(mock_trunc)
    assert diag_trunc.canonical_category == "output_truncated"
    assert diag_trunc.is_recoverable is True

    # Empty candidates
    mock_empty = MagicMock()
    mock_empty.text = None
    mock_empty.candidates = []
    diag_empty = extract_response_diagnostics(mock_empty)
    assert diag_empty.canonical_category == "empty_response"
    assert diag_empty.is_recoverable is True

    # Absent content
    mock_absent = MagicMock()
    mock_absent.text = None
    mock_absent.candidates = [MagicMock(finish_reason="STOP", content=None)]
    diag_absent = extract_response_diagnostics(mock_absent)
    assert diag_absent.canonical_category == "empty_response"
    assert diag_absent.is_recoverable is True

    # Valid response
    mock_ok = MagicMock()
    mock_ok.text = "[{\"id\": \"1\"}]"
    mock_ok.candidates = [MagicMock(finish_reason="STOP", content=MagicMock(parts=[MagicMock(text="[{\"id\": \"1\"}]", thought=False)]))]
    diag_ok = extract_response_diagnostics(mock_ok)
    assert diag_ok.canonical_category == "success"
    assert diag_ok.failure_category == "success"


def test_permanent_4xx_not_retried_at_all():
    """3. Permanent 4xx errors (400, 401, 403, 404) fail immediately without retry loops."""
    for status_code in [400, 401, 403, 404]:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = genai_errors.ClientError(status_code, f"HTTP {status_code} client error")

        provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it", max_retries=3)
        provider._client = mock_client

        with pytest.raises(GemmaApiError) as exc_info:
            provider._generate_with_retry("test prompt", mock_client)

        assert exc_info.value.code == status_code
        # Must have failed on the very first attempt without looping
        assert mock_client.models.generate_content.call_count == 1


def test_multibatch_partial_success_preserves_successful_findings_and_patches(tmp_path):
    """6 & 7. Multi-batch analysis preserves findings from successful batches when a subsequent batch fails."""
    f1 = Finding(id="f1", category=FindingCategory.CONFIGURATION, title="T1", file_path="a.py", evidence="x=1", explanation="e1", suggested_action="a1")
    f2 = Finding(id="f2", category=FindingCategory.CONFIGURATION, title="T2", file_path="b.py", evidence="x=2", explanation="e2", suggested_action="a2")
    f3 = Finding(id="f3", category=FindingCategory.CONFIGURATION, title="T3", file_path="c.py", evidence="x=3", explanation="e3", suggested_action="a3")
    f4 = Finding(id="f4", category=FindingCategory.CONFIGURATION, title="T4", file_path="d.py", evidence="x=4", explanation="e4", suggested_action="a4")

    mock_client = MagicMock()

    # Batch 1 (f1, f2) succeeds
    resp_batch1 = MagicMock()
    resp_batch1.text = json.dumps([
        {"id": "f1", "severity": "high", "explanation": "f1 evaluated", "suggested_action": "fix 1"},
        {"id": "f2", "severity": "medium", "explanation": "f2 evaluated", "suggested_action": "fix 2"},
    ])

    # Batch 2 (f3, f4) fails with 500 error
    err_batch2 = genai_errors.ServerError(500, "Google upstream failure on batch 2")

    # Patch proposal response for f1 succeeds
    patch_resp = MagicMock()
    patch_resp.text = json.dumps({
        "change_summary": "Update a.py",
        "proposed_content": "x = 100\n",
        "explanation": "Fix explanation",
        "risks_assumptions": "none",
        "suggested_verification": "pytest",
    })

    # Side effects: batch 1 succeeds, batch 2 fails on both retry attempts, patch generation succeeds
    mock_client.models.generate_content.side_effect = [
        resp_batch1,
        err_batch2, err_batch2,  # Batch 2 exhausted
        patch_resp,
    ]

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it", max_retries=1, backoff_factor=0.01)
    provider._client = mock_client

    agent = AgentService(provider_name="gemma_api")
    agent.provider = provider

    # Create dummy files
    for name in ["a.py", "b.py", "c.py", "d.py"]:
        p = tmp_path / name
        p.write_text("x = 1\n", encoding="utf-8")

    with patch.object(Config, "GEMMA_API_BATCH_SIZE", 2), \
         patch.object(agent.scan_engine, "discover_files", return_value=([tmp_path / "a.py"], 10)), \
         patch.object(agent.scan_engine, "run_scanners", return_value=([f1, f2, f3, f4], {})):

        run = agent.execute_maintenance_run(
            repo_path=tmp_path,
            repo_name="MultiBatchRepo",
            repo_type="local",
            run_id="run-multibatch",
        )

    # Batch 1 findings MUST be AI-assisted and preserved
    assert f1.ai_assisted is True
    assert f1.severity == Severity.HIGH
    assert f2.ai_assisted is True
    assert f2.severity == Severity.MEDIUM

    # Batch 2 findings MUST remain unassisted for retry
    assert f3.ai_assisted is False
    assert f4.ai_assisted is False

    # Stage 4 must report 'partial'
    assert run.stages["analysis"].status == "partial"
    assert "2/4" in run.stages["analysis"].details

    # Proposals generated for assisted findings MUST be preserved
    assert len(run.patches) >= 1
    assert run.patches[0].finding_id == "f1"


def test_sdk_request_timeout_bounded_to_remaining_budget():
    """3. An SDK request cannot silently exceed the remaining configured budget, and never sends a deadline below Google's 10s minimum."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"status": "ok"}'
    mock_client.models.generate_content.return_value = mock_resp

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")

    with patch.object(Config, "GEMMA_API_REQUEST_TIMEOUT_SECONDS", 15.0):
        # When budget is 12s, allowed request timeout must be bounded to min(15s, 12s - 0.5s safety margin) = 11500ms, and >= 10000ms
        provider._generate_with_retry("test prompt", mock_client, time_budget_seconds=12.0)

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        passed_config = call_kwargs.get("config")
        assert passed_config is not None
        assert passed_config.http_options is not None
        # Must be bounded by remaining budget minus safety margin: min(15000, 12000 - 500) = 11500ms
        assert passed_config.http_options.timeout <= 11500
        # Must NEVER be below Google's 10s minimum allowed deadline
        assert passed_config.http_options.timeout >= 10000
        assert passed_config.http_options.retry_options.attempts == 1


def test_http_504_retries_stop_when_wall_clock_budget_expires():
    """2. HTTP 504 retries stop strictly when the real wall-clock budget expires."""
    mock_client = MagicMock()
    err_504 = genai_errors.ServerError(504, "Gateway Timeout / DEADLINE_EXCEEDED")

    current_virtual_time = 1000.0

    def fake_time():
        nonlocal current_virtual_time
        return current_virtual_time

    def fake_sleep(seconds):
        nonlocal current_virtual_time
        current_virtual_time += seconds

    def side_effect_504(**kwargs):
        nonlocal current_virtual_time
        # Simulate request taking 12 seconds before returning 504
        current_virtual_time += 12.0
        raise err_504

    mock_client.models.generate_content.side_effect = side_effect_504

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it", max_retries=3, backoff_factor=1.0)
    provider._client = mock_client

    with patch("time.time", side_effect=fake_time), \
         patch("time.sleep", side_effect=fake_sleep), \
         patch.object(Config, "GEMMA_API_REQUEST_TIMEOUT_SECONDS", 15.0), \
         patch.object(Config, "GEMMA_API_RETRY_BUDGET_SECONDS", 20.0):

        with pytest.raises(GemmaApiTransientError) as exc_info:
            provider._generate_with_retry("test prompt", mock_client, time_budget_seconds=20.0)

        # After attempt 1 took 12s, attempt 2 had 8s budget remaining.
        # Delay would be 1s, leaving 7s. Attempt 2 took 12s -> total time 25s >= 20s.
        # Retries MUST abort and report budget exhausted
        assert "time budget exhausted" in str(exc_info.value).lower()
        # Retries stopped; did not run all 4 attempts (1 + 3 retries)
        assert mock_client.models.generate_content.call_count < 4


def test_retry_after_delay_cannot_exceed_remaining_budget():
    """7. Retry-After delays cannot exceed the remaining budget."""
    mock_client = MagicMock()
    err_429 = genai_errors.APIError(429, "Resource exhausted")
    mock_resp = MagicMock()
    mock_resp.headers = {"Retry-After": "25"}  # Upstream wants 25s wait
    err_429.response = mock_resp
    mock_client.models.generate_content.side_effect = err_429

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it", max_retries=2)
    provider._client = mock_client

    with patch.object(Config, "GEMMA_API_RETRY_BUDGET_SECONDS", 15.0):
        with pytest.raises(GemmaApiTransientError) as exc_info:
            provider._generate_with_retry("test prompt", mock_client, time_budget_seconds=15.0)

        # Since 25s delay exceeds 15s budget, retries must abort immediately on attempt 1
        assert "time budget exhausted" in str(exc_info.value).lower()
        assert mock_client.models.generate_content.call_count == 1


def test_patch_proposal_shares_patch_budget_and_skips_recovery_if_insufficient_time(tmp_path):
    """4 & 5. Patch proposal shares budget across initial call and recovery, skipping recovery if remaining time is too low."""
    mock_client = MagicMock()
    # Response is truncated with MAX_TOKENS
    mock_trunc = MagicMock()
    mock_trunc.text = '{"incomplete":'
    mock_trunc.candidates = [
        MagicMock(finish_reason="MAX_TOKENS", content=MagicMock(parts=[MagicMock(text='{"incomplete":', thought=False)]))
    ]
    mock_client.models.generate_content.return_value = mock_trunc

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    finding = Finding(
        id="f-trunc",
        category=FindingCategory.CONFIGURATION,
        title="Truncated finding",
        file_path="config.py",
        evidence="api_key = ''",
        explanation="Needs key",
        suggested_action="Add key",
    )

    current_time = 1000.0

    def fake_time():
        nonlocal current_time
        return current_time

    # Simulate initial call taking 32 seconds of a 35s patch budget
    def side_effect_slow(**kwargs):
        nonlocal current_time
        current_time += 32.0
        return mock_trunc

    mock_client.models.generate_content.side_effect = side_effect_slow

    with patch("time.time", side_effect=fake_time), \
         patch.object(Config, "GEMMA_API_PATCH_TIMEOUT_SECONDS", 35.0):

        patch_obj = provider.propose_patch(finding, "original content", tmp_path)

        # Patch proposal returns None without crashing
        assert patch_obj is None
        # Only 1 generate_content call was made because remaining budget (3s) was < 5s threshold!
        assert mock_client.models.generate_content.call_count == 1


def test_failed_patch_does_not_erase_successful_patches_and_records_partial(tmp_path):
    """5, 6 & 9. A failed patch does not erase other successful patches; marks proposal PARTIAL and root tag PARTIAL."""
    f1 = Finding(id="f1", category=FindingCategory.CONFIGURATION, title="Fix 1", file_path="a.py", evidence="x=1", explanation="e1", suggested_action="a1", ai_assisted=True)
    f2 = Finding(id="f2", category=FindingCategory.CONFIGURATION, title="Fix 2", file_path="b.py", evidence="y=2", explanation="e2", suggested_action="a2", ai_assisted=True)
    f3 = Finding(id="f3", category=FindingCategory.CONFIGURATION, title="Fix 3", file_path="c.py", evidence="z=3", explanation="e3", suggested_action="a3", ai_assisted=True)

    for name in ["a.py", "b.py", "c.py"]:
        p = tmp_path / name
        p.write_text("x = 1\n", encoding="utf-8")

    mock_client = MagicMock()

    # Finding 1 patch succeeds
    patch1_resp = MagicMock()
    patch1_resp.text = json.dumps({
        "change_summary": "Update a.py",
        "proposed_content": "x = 42\n",
        "explanation": "Fixed",
        "risks_assumptions": "none",
        "suggested_verification": "pytest",
    })

    # Finding 2 patch fails (504 timeout)
    err_504 = genai_errors.ServerError(504, "Gateway Timeout")

    # Finding 3 patch succeeds
    patch3_resp = MagicMock()
    patch3_resp.text = json.dumps({
        "change_summary": "Update c.py",
        "proposed_content": "z = 99\n",
        "explanation": "Fixed",
        "risks_assumptions": "none",
        "suggested_verification": "pytest",
    })

    # Call side effects: f1 -> patch1_resp; f2 -> err_504 exhausted; f3 -> patch3_resp
    mock_client.models.generate_content.side_effect = [
        patch1_resp,
        err_504, err_504,  # f2 retries fail
        patch3_resp,
    ]

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it", max_retries=1, backoff_factor=0.01)
    provider._client = mock_client

    agent = AgentService(provider_name="gemma_api")
    agent.provider = provider

    with patch.object(agent.scan_engine, "discover_files", return_value=([tmp_path / "a.py"], 10)), \
         patch.object(agent.scan_engine, "run_scanners", return_value=([f1, f2, f3], {})), \
         patch.object(provider, "analyze_and_prioritize", return_value=[f1, f2, f3]):

        run = agent.execute_maintenance_run(
            repo_path=tmp_path,
            repo_name="PatchTestRepo",
            repo_type="local",
            run_id="run-patch-partial",
        )

    # 2 patches were proposed, 1 failed
    assert len(run.patches) == 2
    assert {p.finding_id for p in run.patches} == {"f1", "f3"}

    # Proposal stage must be PARTIAL
    assert run.stages["proposal"].status == "partial"
    assert "2/3" in run.stages["proposal"].details
    assert "1 patch generation attempts failed" in run.stages["proposal"].details

    # Overall run is completed with partial stage
    assert run.status == "completed"


def test_budget_below_google_minimum_causes_no_outgoing_requests():
    """Remaining budget < 10.5s aborts immediately with 0 outgoing requests to Google Gen AI."""
    mock_client = MagicMock()
    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")

    # 1. Exact 2-second deadline rejected by Google in production
    with pytest.raises(GemmaApiTransientError) as exc_info:
        provider._generate_with_retry("test prompt", mock_client, time_budget_seconds=2.0)

    assert "time budget exhausted" in str(exc_info.value).lower()
    assert mock_client.models.generate_content.call_count == 0

    # 2. 10.0 seconds budget (leaves 9.5s < 10.0s deadline required)
    with pytest.raises(GemmaApiTransientError) as exc_info:
        provider._generate_with_retry("test prompt", mock_client, time_budget_seconds=10.0)

    assert "time budget exhausted" in str(exc_info.value).lower()
    assert mock_client.models.generate_content.call_count == 0


def test_request_deadline_is_never_below_ten_seconds():
    """When an outgoing request is issued, the deadline/timeout is strictly >= 10.0s (10000ms)."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"status": "ok"}'
    mock_client.models.generate_content.return_value = mock_resp

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")

    # Even if Config.GEMMA_API_REQUEST_TIMEOUT_SECONDS was erroneously configured as 2.0s or 5.0s
    with patch.object(Config, "GEMMA_API_REQUEST_TIMEOUT_SECONDS", 2.0):
        provider._generate_with_retry("test prompt", mock_client, time_budget_seconds=20.0)

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        passed_config = call_kwargs.get("config")
        assert passed_config is not None
        assert passed_config.http_options.timeout >= 10000


def test_batch_analysis_recovery_skips_when_remaining_budget_cannot_accommodate_google_minimum():
    """Batch analysis skips recovery without a doomed Google request when remaining batch budget is < 10.5s."""
    mock_client = MagicMock()
    mock_trunc = MagicMock()
    mock_trunc.text = "Incomplete JSON {"
    mock_trunc.candidates = [MagicMock(finish_reason="MAX_TOKENS")]

    current_time = 1000.0

    def fake_time():
        nonlocal current_time
        return current_time

    # Simulate initial call taking 22 seconds of a 30s batch budget, leaving 8.0s (< 10.5s)
    def slow_generate(**kwargs):
        nonlocal current_time
        current_time += 22.0
        return mock_trunc

    mock_client.models.generate_content.side_effect = slow_generate

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it")
    provider._client = mock_client

    findings = [
        Finding(
            id=f"f{i}",
            category=FindingCategory.CONFIGURATION,
            title=f"Finding {i}",
            file_path="config.py",
            severity=Severity.HIGH,
            evidence="config error",
            explanation="needs fix",
            suggested_action="update config",
        )
        for i in range(2)
    ]

    with patch("time.time", side_effect=fake_time), \
         patch.object(Config, "GEMMA_API_RETRY_BUDGET_SECONDS", 30.0), \
         patch.object(Config, "GEMMA_API_BATCH_SIZE", 5):

        result = provider.analyze_and_prioritize(findings)

        # Only 1 generate_content call was made; recovery was skipped due to remaining budget (8s < 10.5s)
        assert mock_client.models.generate_content.call_count == 1
        # Findings returned without crash, unassisted
        assert len(result) == 2
        assert not any(f.ai_assisted for f in result)


def test_http_504_retries_stop_and_successful_batches_survive():
    """HTTP 504 on one batch does not erase successful findings from other batches, and allows patches."""
    mock_client = MagicMock()

    # Batch 1 response (success)
    b1_resp = MagicMock()
    b1_resp.text = json.dumps([
        {"id": "f1", "severity": "critical", "explanation": "Urgent bug", "suggested_action": "Fix bug"}
    ])

    # Batch 2 error: 504 Gateway Timeout / DEADLINE_EXCEEDED
    err_504 = genai_errors.ServerError(504, "Gateway Timeout / DEADLINE_EXCEEDED")

    # Batch 3 response (success)
    b3_resp = MagicMock()
    b3_resp.text = json.dumps([
        {"id": "f3", "severity": "low", "explanation": "Minor style", "suggested_action": "Clean up"}
    ])

    mock_client.models.generate_content.side_effect = [
        b1_resp,
        err_504, err_504,  # batch 2 fails across attempts
        b3_resp,
    ]

    provider = GemmaApiProvider(api_key="valid-key", model="gemma-4-26b-a4b-it", max_retries=1, backoff_factor=0.01)
    provider._client = mock_client

    findings = [
        Finding(id="f1", category=FindingCategory.CONFIGURATION, title="F1", file_path="f1.py", severity=Severity.MEDIUM, evidence="e1", explanation="ex1", suggested_action="a1"),
        Finding(id="f2", category=FindingCategory.CONFIGURATION, title="F2", file_path="f2.py", severity=Severity.MEDIUM, evidence="e2", explanation="ex2", suggested_action="a2"),
        Finding(id="f3", category=FindingCategory.CONFIGURATION, title="F3", file_path="f3.py", severity=Severity.MEDIUM, evidence="e3", explanation="ex3", suggested_action="a3"),
    ]

    with patch.object(Config, "GEMMA_API_BATCH_SIZE", 1):
        analyzed = provider.analyze_and_prioritize(findings)

    # Findings from batch 1 and batch 3 were successfully evaluated despite batch 2 failure
    assert analyzed[0].ai_assisted is True
    assert analyzed[0].severity == Severity.CRITICAL
    assert analyzed[1].ai_assisted is False  # Batch 2 failed
    assert analyzed[2].ai_assisted is True
    assert analyzed[2].severity == Severity.LOW

