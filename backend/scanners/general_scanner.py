"""General repository health and hygiene scanner."""
import hashlib
from pathlib import Path
from typing import List
from backend.models import Finding, FindingCategory, Severity, VerificationStatus
from backend.scanners.base import BaseScanner

class GeneralScanner(BaseScanner):
    category = "general"

    def scan(self, repo_path: Path, files: List[Path]) -> List[Finding]:
        findings: List[Finding] = []

        file_names = {f.name for f in files}

        # 1. Missing README.md check
        has_readme = any(f.name.lower().startswith("readme") for f in files)
        if not has_readme:
            findings.append(
                Finding(
                    id=f"finding-gen-no-readme-{hashlib.md5(str(repo_path).encode()).hexdigest()[:8]}",
                    category=FindingCategory.GENERAL,
                    severity=Severity.HIGH,
                    title="Missing repository README file",
                    file_path="README.md",
                    line_number=1,
                    evidence="No README or README.md found in repository root",
                    explanation="A repository without a README provides no entry point, context, or overview for users and maintainers.",
                    suggested_action="Create a README.md file outlining project purpose, installation, and basic usage.",
                    confidence=1.0,
                    verification_status=VerificationStatus.UNVERIFIED,
                    estimated_effort_mins=25,
                )
            )

        # 2. Missing LICENSE check
        has_license = any(f.name.lower().startswith("license") or f.name.lower().startswith("licence") for f in files)
        if not has_license:
            findings.append(
                Finding(
                    id=f"finding-gen-no-license-{hashlib.md5(str(repo_path).encode()).hexdigest()[:8]}",
                    category=FindingCategory.GENERAL,
                    severity=Severity.LOW,
                    title="Missing LICENSE file",
                    file_path="LICENSE",
                    line_number=1,
                    evidence="No LICENSE or LICENSE.md found in repository",
                    explanation="Without an explicit license, default copyright laws apply, meaning contributors and users retain all rights with no permission to use or modify.",
                    suggested_action="Add an appropriate open-source license (such as MIT, Apache-2.0, or BSD-3-Clause).",
                    confidence=0.95,
                    verification_status=VerificationStatus.UNVERIFIED,
                    estimated_effort_mins=10,
                )
            )

        # 3. Missing .gitignore check
        has_gitignore = ".gitignore" in file_names
        if not has_gitignore:
            findings.append(
                Finding(
                    id=f"finding-gen-no-gitignore-{hashlib.md5(str(repo_path).encode()).hexdigest()[:8]}",
                    category=FindingCategory.GENERAL,
                    severity=Severity.MEDIUM,
                    title="Missing .gitignore file",
                    file_path=".gitignore",
                    line_number=1,
                    evidence="No .gitignore configuration found in root",
                    explanation="Without a .gitignore file, temporary files, build outputs, and secrets may accidentally be committed.",
                    suggested_action="Add a .gitignore file tailored to your programming languages and tooling.",
                    confidence=0.95,
                    verification_status=VerificationStatus.UNVERIFIED,
                    estimated_effort_mins=10,
                )
            )

        return findings
