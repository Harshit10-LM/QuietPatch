"""TODO and maintenance marker scanner."""
import hashlib
import re
from pathlib import Path
from typing import List, Set
from backend.models import Finding, FindingCategory, Severity, VerificationStatus
from backend.scanners.base import BaseScanner

MARKER_REGEX = re.compile(
    r"\b(?P<marker>TODO|FIXME|XXX|HACK|BUG)(?:\((?P<tag>[^)]+)\))?:\s*(?P<desc>.*)",
    re.IGNORECASE
)

class TodoScanner(BaseScanner):
    category = "todo"

    def scan(self, repo_path: Path, files: List[Path]) -> List[Finding]:
        findings: List[Finding] = []
        seen_keys: Set[str] = set()

        for file_path in files:
            # Skip doc files for marker scanning to avoid false positives in markdown descriptions
            if file_path.suffix.lower() in {".md", ".txt", ".rst"}:
                continue

            content = self.read_text_safe(file_path)
            if not content:
                continue

            rel_path = self.get_relative_path(repo_path, file_path)
            lines = content.splitlines()

            for line_idx, line in enumerate(lines, start=1):
                match = MARKER_REGEX.search(line)
                if not match:
                    continue

                marker = match.group("marker").upper()
                tag = (match.group("tag") or "").strip().lower()
                desc = match.group("desc").strip()

                # Deduplicate repeated identical markers in same file
                dedup_key = f"{rel_path}:{marker}:{tag}:{desc[:40]}"
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                # Determine severity based on marker type and tag
                severity = Severity.LOW
                if marker in {"FIXME", "BUG"}:
                    severity = Severity.HIGH
                elif marker == "TODO" and tag in {"security", "cve", "auth", "secret"}:
                    severity = Severity.HIGH
                elif marker in {"XXX", "HACK"}:
                    severity = Severity.MEDIUM
                elif marker == "TODO":
                    severity = Severity.LOW

                finding_id = f"finding-todo-{hashlib.md5(dedup_key.encode()).hexdigest()[:8]}"
                title = f"{marker}{f'({tag})' if tag else ''}: {desc[:60] if desc else 'Unaddressed maintenance marker'}"
                
                # Context snippet: the line itself
                evidence = line.strip()
                explanation = (
                    f"Found an active maintenance flag '{marker}' indicating unfinished work or a deferred fix. "
                    f"Description provided: '{desc or 'None'}'. Left unattended, these accumulate technical debt."
                )
                suggested_action = (
                    f"Resolve the '{desc or marker}' task or determine if the requirement is still applicable."
                )

                findings.append(
                    Finding(
                        id=finding_id,
                        category=FindingCategory.TODO,
                        severity=severity,
                        title=title,
                        file_path=rel_path,
                        line_number=line_idx,
                        evidence=evidence,
                        explanation=explanation,
                        suggested_action=suggested_action,
                        confidence=0.98,
                        verification_status=VerificationStatus.UNVERIFIED,
                        estimated_effort_mins=20,
                    )
                )

        return findings
