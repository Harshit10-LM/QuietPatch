"""Documentation scanner for broken relative markdown links and README completeness."""
import hashlib
import os
import re
from pathlib import Path
from typing import List, Set
from urllib.parse import urlparse
from backend.models import Finding, FindingCategory, Severity, VerificationStatus
from backend.scanners.base import BaseScanner

# Match markdown links: [text](target)
MD_LINK_REGEX = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

class DocScanner(BaseScanner):
    category = "documentation"

    def scan(self, repo_path: Path, files: List[Path]) -> List[Finding]:
        findings: List[Finding] = []
        md_files = [f for f in files if f.suffix.lower() in {".md", ".markdown"}]

        # 1. Broken relative Markdown links check
        for md_file in md_files:
            content = self.read_text_safe(md_file)
            if not content:
                continue

            rel_md_path = self.get_relative_path(repo_path, md_file)
            lines = content.splitlines()

            for line_idx, line in enumerate(lines, start=1):
                for match in MD_LINK_REGEX.finditer(line):
                    link_text = match.group(1).strip()
                    target = match.group(2).strip()

                    # Filter out external links, anchors, mailto
                    if (
                        target.startswith("http://")
                        or target.startswith("https://")
                        or target.startswith("mailto:")
                        or target.startswith("#")
                        or target.startswith("javascript:")
                    ):
                        continue

                    # Strip anchor fragment from target path: e.g. path/to/file.md#section -> path/to/file.md
                    target_file_part = target.split("#")[0].strip()
                    if not target_file_part:
                        continue

                    # Resolve target path relative to the directory containing the markdown file
                    target_resolved = (md_file.parent / target_file_part).resolve()
                    
                    # Check existence
                    if not target_resolved.exists():
                        finding_id = f"finding-doc-broken-{hashlib.md5(f'{rel_md_path}:{line_idx}:{target}'.encode()).hexdigest()[:8]}"
                        title = f"Broken relative link: '{target}' in {rel_md_path}"
                        evidence = f"[{link_text}]({target})"
                        explanation = (
                            f"The markdown link '{target}' refers to a local file that does not exist in the repository. "
                            f"Users reading documentation will encounter a 404 broken reference."
                        )
                        suggested_action = (
                            f"Create the missing file at '{self.get_relative_path(repo_path, target_resolved)}' "
                            f"or update the link to point to an existing documentation page."
                        )

                        findings.append(
                            Finding(
                                id=finding_id,
                                category=FindingCategory.DOCUMENTATION,
                                severity=Severity.MEDIUM,
                                title=title,
                                file_path=rel_md_path,
                                line_number=line_idx,
                                evidence=evidence,
                                explanation=explanation,
                                suggested_action=suggested_action,
                                confidence=0.99,
                                verification_status=VerificationStatus.UNVERIFIED,
                                estimated_effort_mins=10,
                            )
                        )

        # 2. README completeness check
        readme_path = repo_path / "README.md"
        if readme_path.exists():
            readme_content = self.read_text_safe(readme_path)
            if readme_content:
                lower_content = readme_content.lower()
                rel_readme = self.get_relative_path(repo_path, readme_path)

                has_install = any(kw in lower_content for kw in ["installation", "setup", "getting started", "install"])
                has_usage = any(kw in lower_content for kw in ["usage", "quickstart", "how to use", "running"])
                has_license = any(kw in lower_content for kw in ["license", "licence"])

                if not has_install:
                    findings.append(
                        Finding(
                            id=f"finding-doc-readme-install-{hashlib.md5(rel_readme.encode()).hexdigest()[:8]}",
                            category=FindingCategory.DOCUMENTATION,
                            severity=Severity.LOW,
                            title="README missing installation or setup instructions",
                            file_path=rel_readme,
                            line_number=1,
                            evidence="README.md has no 'Installation' or 'Setup' section",
                            explanation="Repository documentation lacks clear onboarding instructions for new contributors or operators.",
                            suggested_action="Add an 'Installation' section with prerequisite packages and setup commands.",
                            confidence=0.85,
                            verification_status=VerificationStatus.UNVERIFIED,
                            estimated_effort_mins=15,
                        )
                    )

                if not has_license:
                    findings.append(
                        Finding(
                            id=f"finding-doc-readme-license-{hashlib.md5(rel_readme.encode()).hexdigest()[:8]}",
                            category=FindingCategory.DOCUMENTATION,
                            severity=Severity.INFORMATIONAL,
                            title="README missing license mention",
                            file_path=rel_readme,
                            line_number=1,
                            evidence="No license terms mentioned in README.md",
                            explanation="Open-source contributors and users need clarity on copyright and licensing terms.",
                            suggested_action="Mention the project license terms and link to the LICENSE file.",
                            confidence=0.85,
                            verification_status=VerificationStatus.UNVERIFIED,
                            estimated_effort_mins=5,
                        )
                    )

        return findings
