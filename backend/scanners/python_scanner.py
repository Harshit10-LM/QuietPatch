"""Python project and packaging scanner."""
import hashlib
from pathlib import Path
from typing import List
from backend.models import Finding, FindingCategory, Severity, VerificationStatus
from backend.scanners.base import BaseScanner

class PythonScanner(BaseScanner):
    category = "python"

    def scan(self, repo_path: Path, files: List[Path]) -> List[Finding]:
        findings: List[Finding] = []
        py_files = [f for f in files if f.suffix == ".py"]

        # Only scan if this is indeed a Python project
        if not py_files:
            return findings

        req_files = [f for f in files if f.name == "requirements.txt"]
        pyproject_files = [f for f in files if f.name == "pyproject.toml"]
        setup_files = [f for f in files if f.name == "setup.py"]

        # 1. Check for dependency manifest presence
        if not req_files and not pyproject_files and not setup_files:
            findings.append(
                Finding(
                    id=f"finding-py-no-manifest-{hashlib.md5(str(repo_path).encode()).hexdigest()[:8]}",
                    category=FindingCategory.PYTHON,
                    severity=Severity.HIGH,
                    title="Python project missing dependency manifest",
                    file_path="requirements.txt",
                    line_number=1,
                    evidence=f"Found {len(py_files)} Python source files but no requirements.txt or pyproject.toml",
                    explanation="Without a declared dependency manifest, builds cannot be reliably reproduced in CI/CD or production.",
                    suggested_action="Create a requirements.txt or pyproject.toml declaring required packages.",
                    confidence=0.95,
                    verification_status=VerificationStatus.UNVERIFIED,
                    estimated_effort_mins=20,
                )
            )

        # 2. Check requirements.txt for unpinned dependencies
        for req_file in req_files:
            content = self.read_text_safe(req_file)
            if not content:
                continue

            rel_req = self.get_relative_path(repo_path, req_file)
            unpinned_pkgs: List[str] = []

            for line_idx, line in enumerate(content.splitlines(), start=1):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#") or clean_line.startswith("-"):
                    continue

                # Check if package has version specifier: ==, >=, <=, ~=, <, >
                if not any(op in clean_line for op in ["==", ">=", "<=", "~=", "<", ">"]):
                    unpinned_pkgs.append(clean_line)

            if unpinned_pkgs:
                sample_unpinned = ", ".join(unpinned_pkgs[:4])
                findings.append(
                    Finding(
                        id=f"finding-py-unpinned-{hashlib.md5(rel_req.encode()).hexdigest()[:8]}",
                        category=FindingCategory.PYTHON,
                        severity=Severity.LOW,
                        title=f"Unpinned dependencies in {rel_req}",
                        file_path=rel_req,
                        line_number=1,
                        evidence=f"Unpinned packages: {sample_unpinned}",
                        explanation=(
                            f"{len(unpinned_pkgs)} packages in '{rel_req}' have no version constraints. "
                            f"Future package releases could introduce breaking changes or security regressions."
                        ),
                        suggested_action="Pin dependencies with minimum or exact versions (e.g. package>=1.0.0).",
                        confidence=0.90,
                        verification_status=VerificationStatus.UNVERIFIED,
                        estimated_effort_mins=15,
                    )
                )

        return findings
