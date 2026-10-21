"""Safe Patch and Undo System with path containment, conflict checks, and backups."""
import hashlib
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import Config
from backend.models import PatchStatus, ProposedPatch
from backend.telemetry.tracing import agent_stage_span

logger = logging.getLogger("quietpatch.services.patch")

class PatchError(Exception):
    """Raised when a patch operation cannot proceed safely."""
    pass

class PatchService:
    def __init__(self, allowed_root: Path = Config.QUIETPATCH_ALLOWED_ROOT):
        self.allowed_root = allowed_root.resolve()

    def _resolve_and_validate_path(self, repo_path: Path, rel_file_path: str) -> Path:
        """Resolves target file path against repo root and verifies containment."""
        if not rel_file_path:
            raise PatchError("File path cannot be empty.")

        # Reject path traversal patterns
        clean_rel = rel_file_path.strip().lstrip("/")
        if ".." in Path(clean_rel).parts:
            raise PatchError(f"Path traversal ('..') is prohibited: {rel_file_path}")

        target_file = (repo_path / clean_rel).resolve()

        # Ensure target file stays inside repo_path
        try:
            target_file.relative_to(repo_path.resolve())
        except ValueError:
            raise PatchError(f"Target file escapes repository root: {rel_file_path}")

        # Check for unexpected symlinks
        if target_file.is_symlink():
            raise PatchError(f"Modifying symbolic links is not permitted: {rel_file_path}")

        return target_file

    def get_backup_dir(self, repo_path: Path) -> Path:
        """Returns the repository backup directory, creating it if needed."""
        backup_dir = repo_path / ".quietpatch" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        return backup_dir

    def apply_patch(
        self,
        repo_path: Path,
        patch: ProposedPatch,
        run_id: Optional[str] = None
    ) -> ProposedPatch:
        """Applies an approved patch atomically after validating original content and backing up."""
        # 1. Enforce Hosted Mode restrictions (Render safety)
        if Config.APP_MODE == "hosted":
            raise PatchError(
                "Patch application is disabled in Hosted mode for server security. "
                "Run QuietPatch locally to review and apply changes directly to your files."
            )

        # 2. Path validation
        target_file = self._resolve_and_validate_path(repo_path, patch.file_path)

        with agent_stage_span("Patch Validation & Application", run_id=run_id, extra_tags={"file": patch.file_path}):
            current_content = ""
            file_existed = target_file.exists()

            if file_existed:
                try:
                    with open(target_file, "r", encoding="utf-8", errors="replace") as f:
                        current_content = f.read()
                except Exception as e:
                    raise PatchError(f"Failed to read existing file: {e}")

                # 3. Conflict detection: Verify file has not changed since patch was generated
                orig_normalized = patch.original_content.replace("\r\n", "\n")
                curr_normalized = current_content.replace("\r\n", "\n")
                if orig_normalized != curr_normalized:
                    raise PatchError(
                        f"Conflict detected: '{patch.file_path}' has been modified on disk since this patch was generated. "
                        "Refusing to overwrite conflicting newer edits."
                    )

            # 4. Create safe backup
            backup_dir = self.get_backup_dir(repo_path)
            safe_id = patch.id.replace("/", "_")
            backup_file = backup_dir / f"{safe_id}_{target_file.name}.bak"

            if file_existed:
                try:
                    shutil.copy2(target_file, backup_file)
                    patch.backup_path = str(backup_file.relative_to(repo_path))
                except Exception as e:
                    raise PatchError(f"Failed to create backup file: {e}")
            else:
                patch.backup_path = None  # File was newly created

            # 5. Atomic write of proposed content
            temp_target = target_file.with_suffix(f"{target_file.suffix}.qp_tmp")
            try:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_target, "w", encoding="utf-8") as f:
                    f.write(patch.proposed_content)
                temp_target.replace(target_file)
            except Exception as e:
                if temp_target.exists():
                    temp_target.unlink()
                raise PatchError(f"Failed to write patch to file: {e}")

            patch.status = PatchStatus.APPLIED
            patch.applied_at = datetime.now(timezone.utc).isoformat()
            logger.info("Successfully applied patch '%s' to '%s'", patch.id, patch.file_path)

        return patch

    def undo_patch(
        self,
        repo_path: Path,
        patch: ProposedPatch,
        run_id: Optional[str] = None
    ) -> ProposedPatch:
        """Safely reverts an applied patch if the file has not been modified since."""
        if Config.APP_MODE == "hosted":
            raise PatchError("Patch rollback is disabled in Hosted mode.")

        if patch.status != PatchStatus.APPLIED:
            raise PatchError(f"Cannot undo patch '{patch.id}' because it has status '{patch.status.value}'.")

        target_file = self._resolve_and_validate_path(repo_path, patch.file_path)

        with agent_stage_span("Patch Undo", run_id=run_id, extra_tags={"file": patch.file_path}):
            # Verify file currently matches proposed content
            if not target_file.exists():
                raise PatchError(f"Target file '{patch.file_path}' no longer exists.")

            try:
                with open(target_file, "r", encoding="utf-8", errors="replace") as f:
                    current_content = f.read()
            except Exception as e:
                raise PatchError(f"Failed to read file for undo verification: {e}")

            prop_normalized = patch.proposed_content.replace("\r\n", "\n")
            curr_normalized = current_content.replace("\r\n", "\n")

            if prop_normalized != curr_normalized:
                raise PatchError(
                    f"Conflict: '{patch.file_path}' has been edited since patch application. "
                    "Cannot safely restore original without overwriting subsequent manual changes."
                )

            # Restore original content from backup or delete if it was a new file
            if patch.backup_path:
                backup_file = repo_path / patch.backup_path
                if not backup_file.exists():
                    raise PatchError(f"Backup file '{patch.backup_path}' was not found.")
                try:
                    shutil.copy2(backup_file, target_file)
                    backup_file.unlink()  # Clean up backup
                except Exception as e:
                    raise PatchError(f"Failed to restore original file from backup: {e}")
            else:
                # File was newly created by the patch, delete it
                try:
                    target_file.unlink()
                except Exception as e:
                    raise PatchError(f"Failed to remove newly created file: {e}")

            patch.status = PatchStatus.UNDONE
            logger.info("Successfully undid patch '%s' on '%s'", patch.id, patch.file_path)

        return patch
