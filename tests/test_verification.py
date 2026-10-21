"""Unit tests for Verification Service."""
from pathlib import Path
import pytest
from backend.models import Finding, FindingCategory, ProposedPatch, Severity, VerificationStatus
from backend.services.verification_service import VerificationService

def test_verify_configuration_finding(tmp_path: Path):
    env_file = tmp_path / ".env.example"
    env_file.write_text("PORT=8000\nAPI_SECRET_KEY=placeholder\n")

    finding = Finding(
        id="f-conf",
        category=FindingCategory.CONFIGURATION,
        severity=Severity.HIGH,
        title="Undocumented environment variable: 'API_SECRET_KEY'",
        file_path=".env.example",
        evidence="API_SECRET_KEY",
        explanation="",
        suggested_action="",
    )

    patch = ProposedPatch(
        id="p-conf",
        finding_id="f-conf",
        file_path=".env.example",
        change_summary="Add API_SECRET_KEY",
        original_content="PORT=8000\n",
        proposed_content="PORT=8000\nAPI_SECRET_KEY=placeholder\n",
        diff="---",
        explanation="",
        risks_assumptions="",
        suggested_verification="",
    )

    verifier = VerificationService(allowed_root=tmp_path)
    res = verifier.verify_finding_resolution(tmp_path, finding, patch)

    assert res["passed"] is True
    assert finding.verification_status == VerificationStatus.PASSED

def test_verify_command_allowlist_enforcement(tmp_path: Path):
    verifier = VerificationService(allowed_root=tmp_path)

    # Reject non-allowlisted command
    res = verifier.execute_approved_test_command(tmp_path, "rm -rf /")
    assert res["success"] is False
    assert "not in the approved verification allowlist" in res["message"]

    # Allowlisted command execution
    res2 = verifier.execute_approved_test_command(tmp_path, "python-check")
    # Will succeed or fail based on python syntax check arguments, but command was allowlisted
    assert res2["exit_code"] is not None
