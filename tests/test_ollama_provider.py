"""Unit and integration tests for Ollama Gemma provider, availability, and error handling."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests

from backend.ai.agent_service import AgentService
from backend.ai.ollama_provider import OllamaProvider, clean_json_text
from backend.config import Config
from backend.models import Finding, FindingCategory, Severity, VerificationStatus


def test_clean_json_text_strips_markdown_fences():
    raw_markdown = '```json\n{"findings": [{"id": "f1"}]}\n```'
    cleaned = clean_json_text(raw_markdown)
    assert cleaned == '{"findings": [{"id": "f1"}]}'

    raw_plain = '  {"findings": []}  '
    assert clean_json_text(raw_plain) == '{"findings": []}'


def test_ollama_check_availability_connected():
    provider = OllamaProvider(base_url="http://localhost:11434", model="gemma4:e4b")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {"name": "gemma4:e4b", "size": 4000000000},
            {"name": "qwen2.5:latest", "size": 3000000000},
        ]
    }

    with patch("requests.get", return_value=mock_resp) as mock_get:
        status = provider.check_availability()
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=4.0)
        assert status["available"] is True
        assert status["status"] == "connected"
        assert status["model_installed"] is True
        assert status["model"] == "gemma4:e4b"


def test_ollama_check_availability_missing_model():
    provider = OllamaProvider(base_url="http://localhost:11434", model="gemma4:e4b")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": "llama3:latest"}]
    }

    with patch("requests.get", return_value=mock_resp):
        status = provider.check_availability()
        assert status["available"] is False
        assert status["status"] == "model_missing"
        assert status["model_installed"] is False
        assert "gemma4:e4b" in status["instruction"]


def test_ollama_check_availability_offline():
    provider = OllamaProvider(base_url="http://localhost:11434", model="gemma4:e4b")

    with patch("requests.get", side_effect=requests.ConnectionError("Connection refused")):
        status = provider.check_availability()
        assert status["available"] is False
        assert status["status"] == "offline"
        assert status["reachable"] is False
        assert "Cannot reach" in status["message"]


def test_ollama_check_availability_timeout():
    provider = OllamaProvider(base_url="http://localhost:11434", model="gemma4:e4b")

    with patch("requests.get", side_effect=requests.Timeout("Read timed out")):
        status = provider.check_availability()
        assert status["available"] is False
        assert status["status"] == "offline"


def test_ollama_check_availability_non_200():
    provider = OllamaProvider(base_url="http://localhost:11434", model="gemma4:e4b")
    mock_resp = MagicMock()
    mock_resp.status_code = 502

    with patch("requests.get", return_value=mock_resp):
        status = provider.check_availability()
        assert status["available"] is False
        assert status["status"] == "error"
        assert "HTTP 502" in status["message"]


def test_ollama_analyze_and_prioritize_success_payload():
    provider = OllamaProvider(base_url="http://localhost:11434", model="gemma4:e4b")

    findings = [
        Finding(
            id="f-todo-01",
            category=FindingCategory.TODO,
            severity=Severity.LOW,
            title="Unfinished TODO in app.py",
            file_path="app.py",
            line_number=10,
            evidence="# TODO: Add error handling",
            explanation="Initial scanner explanation",
            suggested_action="Resolve TODO",
            confidence=1.0,
            verification_status=VerificationStatus.UNVERIFIED,
            ai_assisted=False,
            estimated_effort_mins=15,
        )
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {
            "content": json.dumps({
                "findings": [
                    {
                        "id": "f-todo-01",
                        "severity": "Medium",
                        "explanation": "Unhandled errors could crash the main Flask process.",
                        "suggested_action": "Wrap request in try/except with structured logging.",
                    }
                ]
            })
        }
    }

    with patch.object(provider, "check_availability", return_value={"available": True, "status": "connected"}), \
         patch("requests.post", return_value=mock_resp) as mock_post:
        updated = provider.analyze_and_prioritize(findings)

        # Verify request parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:11434/api/chat"
        payload = kwargs["json"]
        assert payload["model"] == "gemma4:e4b"
        assert payload["stream"] is False
        assert payload["format"] == "json"
        assert len(payload["messages"]) == 2

        # Verify finding updates
        f = updated[0]
        assert f.severity == Severity.MEDIUM
        assert f.explanation == "Unhandled errors could crash the main Flask process."
        assert f.suggested_action == "Wrap request in try/except with structured logging."
        assert f.ai_assisted is True
        assert f.is_demo is False
        assert f.ai_model == "gemma4:e4b"


def test_ollama_analyze_and_prioritize_bare_list():
    provider = OllamaProvider(base_url="http://localhost:11434", model="gemma4:e4b")

    findings = [
        Finding(
            id="f-sec-01",
            category=FindingCategory.CONFIGURATION,
            severity=Severity.LOW,
            title="Missing SECRET_KEY",
            file_path=".env.example",
            line_number=1,
            evidence="SECRET_KEY missing",
            explanation="Scan finding",
            suggested_action="Add key",
            confidence=1.0,
            verification_status=VerificationStatus.UNVERIFIED,
            ai_assisted=False,
            estimated_effort_mins=10,
        )
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {
            "content": json.dumps([
                {
                    "id": "f-sec-01",
                    "severity": "high",
                    "explanation": "Missing secret keys can allow token spoofing.",
                    "suggested_action": "Add SECRET_KEY placeholder.",
                }
            ])
        }
    }

    with patch.object(provider, "check_availability", return_value={"available": True}), \
         patch("requests.post", return_value=mock_resp):
        updated = provider.analyze_and_prioritize(findings)
        assert updated[0].severity == Severity.HIGH
        assert updated[0].ai_assisted is True
        assert updated[0].is_demo is False


def test_ollama_analyze_and_prioritize_malformed_json_resilience():
    provider = OllamaProvider(base_url="http://localhost:11434", model="gemma4:e4b")

    findings = [
        Finding(
            id="f-01",
            category=FindingCategory.GENERAL,
            severity=Severity.LOW,
            title="Scan Finding",
            file_path="README.md",
            line_number=1,
            evidence="evidence",
            explanation="Original explanation",
            suggested_action="Fix",
            confidence=1.0,
            verification_status=VerificationStatus.UNVERIFIED,
            ai_assisted=False,
            estimated_effort_mins=5,
        )
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "INVALID_NOT_JSON"}}

    with patch.object(provider, "check_availability", return_value={"available": True}), \
         patch("requests.post", return_value=mock_resp):
        # Should not raise exception and should preserve original finding
        updated = provider.analyze_and_prioritize(findings)
        assert updated[0].explanation == "Original explanation"
        assert updated[0].ai_assisted is False


def test_ollama_propose_patch_success():
    provider = OllamaProvider(base_url="http://localhost:11434", model="gemma4:e4b")

    finding = Finding(
        id="f-patch-01",
        category=FindingCategory.DOCUMENTATION,
        severity=Severity.MEDIUM,
        title="Broken link in README",
        file_path="README.md",
        line_number=5,
        evidence="[Docs](docs/missing.md)",
        explanation="404 link",
        suggested_action="Update link",
        confidence=1.0,
        verification_status=VerificationStatus.UNVERIFIED,
        ai_assisted=True,
        estimated_effort_mins=5,
    )

    original_content = "# Project\n[Docs](docs/missing.md)\n"
    proposed_content = "# Project\n[Docs](docs/index.md)\n"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {
            "content": json.dumps({
                "change_summary": "Fix broken docs link",
                "proposed_content": proposed_content,
                "explanation": "Points to index.md which exists.",
                "risks_assumptions": "Assumes index.md is entrypoint.",
                "suggested_verification": "Check link target exists.",
            })
        }
    }

    with patch.object(provider, "check_availability", return_value={"available": True}), \
         patch("requests.post", return_value=mock_resp):
        patch_obj = provider.propose_patch(finding, original_content, Path("/tmp"))
        assert patch_obj is not None
        assert patch_obj.finding_id == "f-patch-01"
        assert patch_obj.is_demo is False
        assert patch_obj.ai_model == "gemma4:e4b"
        assert "diff" in patch_obj.model_dump()
        assert "+[Docs](docs/index.md)" in patch_obj.diff


def test_no_silent_demo_fallback_when_ollama_fails():
    agent = AgentService(provider_name="ollama")

    # Simulate Ollama connection failure during analyze_and_prioritize
    with patch.object(agent.provider, "check_availability", return_value={"available": False, "status": "offline"}):
        run = agent.execute_maintenance_run(
            repo_path=Config.SAMPLE_REPO_PATH,
            repo_name="NovaTask (Sample Project)",
            repo_type="sample",
            run_id="test-no-silent-fallback",
        )

        assert run.status == "completed"
        # Deterministic scanner findings must be preserved
        assert len(run.findings) > 0
        # CRITICAL: Must not silently substitute demo explanations!
        for f in run.findings:
            assert f.ai_assisted is False
            assert f.is_demo is False
            assert "[Demo AI]" not in f.explanation
        # Stage 'analysis' must report the failed AI analysis
        assert run.stages["analysis"].status == "failed"


def test_configurable_timeout_keep_alive_and_batch_size():
    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="gemma4:e4b",
        timeout=180.0,
        keep_alive="30m",
        batch_size=2,
    )
    assert provider.timeout == 180.0
    assert provider.keep_alive == "30m"
    assert provider.batch_size == 2

    findings = [
        Finding(
            id=f"f-{i}",
            category=FindingCategory.TODO,
            severity=Severity.LOW,
            title=f"Finding {i}",
            file_path="app.py",
            line_number=i,
            evidence=f"# TODO {i}",
            explanation="Scan finding",
            suggested_action="Fix",
            confidence=1.0,
            verification_status=VerificationStatus.UNVERIFIED,
            ai_assisted=False,
            estimated_effort_mins=5,
        )
        for i in range(3)
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {
            "content": json.dumps({
                "findings": [
                    {"id": "f-0", "severity": "Medium", "explanation": "E0", "suggested_action": "A0"},
                    {"id": "f-1", "severity": "High", "explanation": "E1", "suggested_action": "A1"},
                    {"id": "f-2", "severity": "Low", "explanation": "E2", "suggested_action": "A2"},
                ]
            })
        }
    }

    with patch.object(provider, "check_availability", return_value={"available": True}), \
         patch("requests.post", return_value=mock_resp) as mock_post:
        updated = provider.analyze_and_prioritize(findings)

        # 3 findings with batch_size=2 should yield 2 HTTP requests
        assert mock_post.call_count == 2
        for call in mock_post.call_args_list:
            args, kwargs = call
            assert kwargs["timeout"] == 180.0
            assert kwargs["json"]["keep_alive"] == "30m"
            assert kwargs["json"]["model"] == "gemma4:e4b"

        assert len(updated) == 3
        assert all(f.ai_assisted for f in updated)


def test_partial_batch_failures_and_no_false_ai_badges():
    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="gemma4:e4b",
        timeout=90.0,
        batch_size=2,
    )

    findings = [
        Finding(
            id=f"f-{i}",
            category=FindingCategory.TODO,
            severity=Severity.LOW,
            title=f"Finding {i}",
            file_path="app.py",
            line_number=i,
            evidence=f"# TODO {i}",
            explanation=f"Scanner original {i}",
            suggested_action="Fix",
            confidence=1.0,
            verification_status=VerificationStatus.UNVERIFIED,
            ai_assisted=False,
            estimated_effort_mins=5,
        )
        for i in range(4)
    ]

    # Batch 1 succeeds, Batch 2 times out
    resp_success = MagicMock()
    resp_success.status_code = 200
    resp_success.json.return_value = {
        "message": {
            "content": json.dumps({
                "findings": [
                    {"id": "f-0", "severity": "High", "explanation": "Evaluated 0", "suggested_action": "Act 0"},
                    {"id": "f-1", "severity": "Medium", "explanation": "Evaluated 1", "suggested_action": "Act 1"},
                ]
            })
        }
    }

    with patch.object(provider, "check_availability", return_value={"available": True}), \
         patch("requests.post", side_effect=[resp_success, requests.Timeout("Batch 2 timed out")]):
        updated = provider.analyze_and_prioritize(findings)

        # First 2 findings are evaluated
        assert updated[0].ai_assisted is True
        assert updated[0].explanation == "Evaluated 0"
        assert updated[1].ai_assisted is True
        assert updated[1].explanation == "Evaluated 1"

        # Second 2 findings must NOT have false AI success badges!
        assert updated[2].ai_assisted is False
        assert updated[2].explanation == "Scanner original 2"
        assert updated[3].ai_assisted is False
        assert updated[3].explanation == "Scanner original 3"


def test_all_batches_timeout_raises_runtime_error():
    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="gemma4:e4b",
        timeout=60.0,
        batch_size=2,
    )

    findings = [
        Finding(
            id="f-1",
            category=FindingCategory.TODO,
            severity=Severity.LOW,
            title="Finding 1",
            file_path="app.py",
            line_number=1,
            evidence="# TODO",
            explanation="Scanner original",
            suggested_action="Fix",
            confidence=1.0,
            verification_status=VerificationStatus.UNVERIFIED,
            ai_assisted=False,
            estimated_effort_mins=5,
        )
    ]

    with patch.object(provider, "check_availability", return_value={"available": True}), \
         patch("requests.post", side_effect=requests.Timeout("Read timed out")):
        with pytest.raises(RuntimeError) as exc_info:
            provider.analyze_and_prioritize(findings)
        assert "timed out" in str(exc_info.value).lower()
        assert "gemma4:e4b" in str(exc_info.value)
