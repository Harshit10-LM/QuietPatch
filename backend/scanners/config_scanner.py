"""Configuration scanner for environment variables and .env.example parity."""
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Set
from backend.models import Finding, FindingCategory, Severity, VerificationStatus
from backend.scanners.base import BaseScanner

# Matches Python env accesses: os.getenv("VAR"), os.environ.get("VAR"), os.environ["VAR"]
PY_ENV_REGEX = re.compile(
    r"""(?:os\.getenv|os\.environ\.get)\(\s*['"]([A-Z0-9_]{3,})['"]|os\.environ\[\s*['"]([A-Z0-9_]{3,})['"]\s*\]"""
)

# Matches JS/TS env accesses: process.env.VAR or import.meta.env.VAR
JS_ENV_REGEX = re.compile(
    r"""(?:process\.env|import\.meta\.env)(?:\.([A-Z0-9_]{3,})|\[\s*['"]([A-Z0-9_]{3,})['"]\s*\])"""
)

# Standard ignored common system variables
IGNORED_SYSTEM_VARS: Set[str] = {
    "PATH", "HOME", "USER", "SHELL", "TERM", "PWD", "LANG", "LC_ALL",
    "NODE_ENV", "PYTHONPATH", "PYTHONUNBUFFERED", "CI", "TZ"
}

class ConfigScanner(BaseScanner):
    category = "configuration"

    def scan(self, repo_path: Path, files: List[Path]) -> List[Finding]:
        findings: List[Finding] = []

        # Find .env.example / .env.sample / .env.template
        example_env_files = [
            f for f in files
            if f.name in {".env.example", ".env.sample", ".env.template", "env.example"}
        ]

        documented_vars: Set[str] = set()
        primary_example_file: Optional[Path] = None

        if example_env_files:
            primary_example_file = example_env_files[0]
            example_content = self.read_text_safe(primary_example_file)
            if example_content:
                for line in example_content.splitlines():
                    clean_line = line.strip()
                    if clean_line and not clean_line.startswith("#"):
                        key_part = clean_line.split("=")[0].strip()
                        if key_part:
                            documented_vars.add(key_part)

        # Map of detected var -> list of (rel_path, line_idx, line)
        detected_vars: Dict[str, List[tuple]] = {}

        for file_path in files:
            ext = file_path.suffix.lower()
            if ext in {".py", ".pyw"}:
                content = self.read_text_safe(file_path)
                if not content:
                    continue
                rel_path = self.get_relative_path(repo_path, file_path)
                for line_idx, line in enumerate(content.splitlines(), start=1):
                    for match in PY_ENV_REGEX.finditer(line):
                        var_name = match.group(1) or match.group(2)
                        if var_name and var_name not in IGNORED_SYSTEM_VARS:
                            detected_vars.setdefault(var_name, []).append((rel_path, line_idx, line.strip()))
            elif ext in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
                content = self.read_text_safe(file_path)
                if not content:
                    continue
                rel_path = self.get_relative_path(repo_path, file_path)
                for line_idx, line in enumerate(content.splitlines(), start=1):
                    for match in JS_ENV_REGEX.finditer(line):
                        var_name = match.group(1) or match.group(2)
                        if var_name and var_name not in IGNORED_SYSTEM_VARS:
                            detected_vars.setdefault(var_name, []).append((rel_path, line_idx, line.strip()))

        # Check for missing variables if .env.example exists
        if primary_example_file:
            example_rel = self.get_relative_path(repo_path, primary_example_file)
            for var_name, usages in detected_vars.items():
                if var_name not in documented_vars:
                    # Found undocumented environment variable!
                    primary_rel, primary_line, snippet = usages[0]
                    # Redact potential sensitive value assignment from snippet display
                    safe_evidence = re.sub(r'(=[\'"][^\'"]+[\'"])', '="***"', snippet)

                    # Assess severity
                    is_secret = any(term in var_name.lower() for term in ["secret", "key", "token", "password", "auth", "cred"])
                    severity = Severity.HIGH if is_secret else Severity.MEDIUM

                    finding_id = f"finding-config-missing-env-{hashlib.md5(f'{var_name}:{example_rel}'.encode()).hexdigest()[:8]}"
                    title = f"Undocumented environment variable: '{var_name}'"
                    explanation = (
                        f"The variable '{var_name}' is consumed in code ({primary_rel}:{primary_line}) "
                        f"but is absent from '{example_rel}'. New developers or deployments will fail or behave unexpectedly."
                    )
                    suggested_action = (
                        f"Document '{var_name}' with a placeholder or default value in '{example_rel}'."
                    )

                    findings.append(
                        Finding(
                            id=finding_id,
                            category=FindingCategory.CONFIGURATION,
                            severity=severity,
                            title=title,
                            file_path=example_rel,
                            line_number=None,
                            evidence=f"Used in {primary_rel}:{primary_line} -> {safe_evidence}",
                            explanation=explanation,
                            suggested_action=suggested_action,
                            confidence=0.95,
                            verification_status=VerificationStatus.UNVERIFIED,
                            estimated_effort_mins=15,
                        )
                    )
        elif detected_vars:
            # Code uses environment variables, but no .env.example exists at all!
            finding_id = f"finding-config-no-example-{hashlib.md5(str(repo_path).encode()).hexdigest()[:8]}"
            var_list = sorted(list(detected_vars.keys()))[:5]
            findings.append(
                Finding(
                    id=finding_id,
                    category=FindingCategory.CONFIGURATION,
                    severity=Severity.MEDIUM,
                    title="Missing .env.example configuration template",
                    file_path=".env.example",
                    line_number=1,
                    evidence=f"Detected env vars in code: {', '.join(var_list)}",
                    explanation="The repository accesses environment variables but provides no example template (.env.example).",
                    suggested_action="Create a .env.example template defining all expected environment variables.",
                    confidence=0.90,
                    verification_status=VerificationStatus.UNVERIFIED,
                    estimated_effort_mins=20,
                )
            )

        return findings
