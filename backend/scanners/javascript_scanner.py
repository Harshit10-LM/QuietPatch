"""JavaScript / TypeScript project scanner."""
import hashlib
import json
from pathlib import Path
from typing import List
from backend.models import Finding, FindingCategory, Severity, VerificationStatus
from backend.scanners.base import BaseScanner

class JavaScriptScanner(BaseScanner):
    category = "javascript"

    def scan(self, repo_path: Path, files: List[Path]) -> List[Finding]:
        findings: List[Finding] = []
        package_json_files = [f for f in files if f.name == "package.json"]

        if not package_json_files:
            return findings

        # Check for lockfile presence in the repo
        lockfile_names = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb"}
        has_lockfile = any(f.name in lockfile_names for f in files)

        for pkg_file in package_json_files:
            content = self.read_text_safe(pkg_file)
            if not content:
                continue

            rel_pkg = self.get_relative_path(repo_path, pkg_file)

            try:
                data = json.loads(content)
            except Exception as e:
                findings.append(
                    Finding(
                        id=f"finding-js-invalid-json-{hashlib.md5(rel_pkg.encode()).hexdigest()[:8]}",
                        category=FindingCategory.JAVASCRIPT,
                        severity=Severity.CRITICAL,
                        title=f"Malformed JSON syntax in {rel_pkg}",
                        file_path=rel_pkg,
                        line_number=1,
                        evidence=str(e),
                        explanation="The package.json file cannot be parsed as valid JSON.",
                        suggested_action="Fix syntax errors in package.json to restore package management functionality.",
                        confidence=1.0,
                        verification_status=VerificationStatus.UNVERIFIED,
                        estimated_effort_mins=10,
                    )
                )
                continue

            scripts = data.get("scripts", {})

            # 1. Check for standard 'test' script
            if "test" not in scripts:
                findings.append(
                    Finding(
                        id=f"finding-js-no-test-{hashlib.md5(rel_pkg.encode()).hexdigest()[:8]}",
                        category=FindingCategory.JAVASCRIPT,
                        severity=Severity.LOW,
                        title=f"No 'test' script defined in {rel_pkg}",
                        file_path=rel_pkg,
                        line_number=1,
                        evidence=f"scripts: {list(scripts.keys())}",
                        explanation="No automated test script was declared in package.json. CI pipelines and verification checks rely on 'npm test'.",
                        suggested_action="Add a 'test' script to package.json (e.g., 'test': 'vitest run' or appropriate runner).",
                        confidence=0.90,
                        verification_status=VerificationStatus.UNVERIFIED,
                        estimated_effort_mins=15,
                    )
                )

            # 2. Check for missing lockfile
            if not has_lockfile and pkg_file.parent == repo_path:
                findings.append(
                    Finding(
                        id=f"finding-js-no-lockfile-{hashlib.md5(rel_pkg.encode()).hexdigest()[:8]}",
                        category=FindingCategory.JAVASCRIPT,
                        severity=Severity.MEDIUM,
                        title="Missing package manager lockfile",
                        file_path=rel_pkg,
                        line_number=1,
                        evidence="package.json found without package-lock.json, yarn.lock, or pnpm-lock.yaml",
                        explanation="Without a deterministic lockfile, different environments may install divergent transitive dependency versions.",
                        suggested_action="Run 'npm install' or 'pnpm install' and commit the resulting lockfile.",
                        confidence=0.95,
                        verification_status=VerificationStatus.UNVERIFIED,
                        estimated_effort_mins=10,
                    )
                )

        return findings
