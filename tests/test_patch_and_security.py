"""Unit tests for Safe Patch and Undo System and Security Boundaries."""
import os
from pathlib import Path
import pytest
from backend.config import Config
from backend.models import PatchStatus, ProposedPatch
from backend.services.patch_service import PatchError, PatchService
from backend.services.repository_service import RepositoryService, SecurityError

def test_repository_service_rejects_path_traversal(tmp_path: Path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_dir = tmp_path / "secret"
    outside_dir.mkdir()

    service = RepositoryService(allowed_root=allowed_root)

    # Valid within allowed
    sub_dir = allowed_root / "project"
    sub_dir.mkdir()
    assert service.validate_local_path(str(sub_dir)) == sub_dir

    # Traversal attempt
    with pytest.raises(SecurityError, match="outside the permitted repository root"):
        service.validate_local_path(str(sub_dir / ".." / ".." / "secret"))

    # Direct outside path
    with pytest.raises(SecurityError, match="outside the permitted repository root"):
        service.validate_local_path(str(outside_dir))

def test_github_url_ssrf_and_domain_validation():
    service = RepositoryService()

    # Valid URLs
    owner, repo = service.validate_github_url("https://github.com/google/gemma")
    assert owner == "google"
    assert repo == "gemma"

    # Reject non-github domains
    with pytest.raises(SecurityError, match="Only public repositories from github.com"):
        service.validate_github_url("https://gitlab.com/owner/repo")

    # Reject localhost / private IPs
    with pytest.raises(SecurityError, match="Only public repositories from github.com"):
        service.validate_github_url("http://127.0.0.1:8000/malicious/repo")

    with pytest.raises(SecurityError, match="Only public repositories from github.com"):
        service.validate_github_url("http://169.254.169.254/latest/meta-data/")

def test_patch_service_apply_backup_and_undo(tmp_path: Path):
    repo = tmp_path / "test_repo"
    repo.mkdir()
    target_file = repo / "config.env"
    initial_content = "PORT=8080\n"
    target_file.write_text(initial_content)

    proposed_content = "PORT=8080\nDEBUG=true\n"
    patch = ProposedPatch(
        id="patch-01",
        finding_id="finding-01",
        file_path="config.env",
        change_summary="Add DEBUG setting",
        original_content=initial_content,
        proposed_content=proposed_content,
        diff="--- a/config.env\n+++ b/config.env\n",
        explanation="Explain",
        risks_assumptions="None",
        suggested_verification="Verify",
    )

    patch_service = PatchService(allowed_root=tmp_path)

    # 1. Apply patch
    updated_patch = patch_service.apply_patch(repo, patch)
    assert updated_patch.status == PatchStatus.APPLIED
    assert target_file.read_text() == proposed_content
    assert updated_patch.backup_path is not None
    assert (repo / updated_patch.backup_path).exists()

    # 2. Undo patch
    undone_patch = patch_service.undo_patch(repo, updated_patch)
    assert undone_patch.status == PatchStatus.UNDONE
    assert target_file.read_text() == initial_content
    # Backup should be cleaned up
    assert not (repo / updated_patch.backup_path).exists()

def test_patch_service_conflict_detection_prevents_overwrite(tmp_path: Path):
    repo = tmp_path / "test_repo"
    repo.mkdir()
    target_file = repo / "README.md"
    target_file.write_text("Version 1.0\n")

    # Patch was generated against "Version 1.0", but file has since changed to "Version 2.0"
    target_file.write_text("Version 2.0 (modified by user)\n")

    patch = ProposedPatch(
        id="patch-conflict",
        finding_id="finding-conflict",
        file_path="README.md",
        change_summary="Update readme",
        original_content="Version 1.0\n",
        proposed_content="Version 1.1\n",
        diff="---",
        explanation="",
        risks_assumptions="",
        suggested_verification="",
    )

    patch_service = PatchService(allowed_root=tmp_path)
    with pytest.raises(PatchError, match="Conflict detected"):
        patch_service.apply_patch(repo, patch)

    # Target file must remain untouched
    assert target_file.read_text() == "Version 2.0 (modified by user)\n"

def test_patch_service_hosted_mode_restrictions(tmp_path: Path, monkeypatch):
    repo = tmp_path / "test_repo"
    repo.mkdir()
    target_file = repo / "file.txt"
    target_file.write_text("original")

    patch = ProposedPatch(
        id="patch-hosted",
        finding_id="finding-hosted",
        file_path="file.txt",
        change_summary="Hosted fix",
        original_content="original",
        proposed_content="new",
        diff="---",
        explanation="",
        risks_assumptions="",
        suggested_verification="",
    )

    patch_service = PatchService(allowed_root=tmp_path)

    # Simulate Hosted Mode (Render)
    monkeypatch.setattr(Config, "APP_MODE", "hosted")

    with pytest.raises(PatchError, match="disabled in Hosted mode"):
        patch_service.apply_patch(repo, patch)
