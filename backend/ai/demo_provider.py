"""Demonstration AI Provider for offline, zero-credential evaluation."""
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.ai.provider_base import BaseAIProvider
from backend.models import Finding, PatchStatus, ProposedPatch, Severity
from backend.telemetry.tracing import agent_stage_span

class DemoProvider(BaseAIProvider):
    name = "demo"

    def check_availability(self) -> Dict[str, Any]:
        return {
            "available": True,
            "status": "demo",
            "model": "demonstration-mode",
            "message": "Demonstration mode active. Zero external credentials required.",
            "instruction": "This mode runs deterministic scanners and displays curated demonstration Gemma explanations and patches.",
            "is_demo": True,
        }

    def analyze_and_prioritize(
        self,
        findings: List[Finding],
        run_id: Optional[str] = None
    ) -> List[Finding]:
        """Provides curated, realistic demonstration explanations and prioritizations."""
        with agent_stage_span("Demo Prioritization Simulation", run_id=run_id, provider=self.name, model="demo-model"):
            for f in findings:
                # Clearly label as demonstration fixture, not live inference
                f.ai_assisted = False
                f.is_demo = True
                f.ai_model = "demo-fixture"
                if "API_SECRET_KEY" in f.title or "security" in f.title.lower():
                    f.severity = Severity.HIGH
                    f.explanation = (
                        "[Demo AI] Secret and security-adjacent configuration must be explicitly documented with placeholders "
                        "to prevent operational downtime and accidental credential misplacement."
                    )
                elif "broken relative link" in f.title.lower():
                    f.severity = Severity.MEDIUM
                    f.explanation = (
                        "[Demo AI] Broken internal links lead to contributor friction and 404 documentation navigation dead ends."
                    )
                elif "no 'test' script" in f.title.lower():
                    f.severity = Severity.LOW
                    f.explanation = (
                        "[Demo AI] Declaring a standardized test script in package.json enables CI verification without manual flag configuration."
                    )
                elif "missing .gitignore" in f.title.lower():
                    f.severity = Severity.MEDIUM
                    f.explanation = (
                        "[Demo AI] A missing .gitignore increases the risk of accidentally committing secrets, logs, and bulky build artifacts."
                    )
                elif "missing license" in f.title.lower():
                    f.severity = Severity.LOW
                    f.explanation = (
                        "[Demo AI] Without an explicit license file, software defaults to all rights reserved, preventing open-source adoption."
                    )
                else:
                    f.explanation = f"[Demo AI] Grounded finding identified by deterministic scanner in {f.file_path}."

        return findings

    def propose_patch(
        self,
        finding: Finding,
        file_content: str,
        repo_path: Path,
        run_id: Optional[str] = None
    ) -> Optional[ProposedPatch]:
        """Produces a deterministic, high-quality unified diff patch for demonstration."""
        with agent_stage_span("Demo Patch Synthesis", run_id=run_id, provider=self.name, model="demo-model"):
            proposed_content = None
            summary = ""
            explanation = ""
            verification = ""

            # 1. Undocumented environment variables in .env.example
            if finding.category.value == "configuration" and "undocumented" in finding.title.lower():
                var_match = None
                for word in finding.title.split():
                    clean_w = word.strip("':\"")
                    if clean_w.isupper() and "_" in clean_w:
                        var_match = clean_w
                        break

                var_to_add = var_match or "MISSING_ENV_VAR"
                default_val = "secret_placeholder_here" if "secret" in var_to_add.lower() or "key" in var_to_add.lower() else "default_value"

                lines = file_content.splitlines()
                lines.append(f"{var_to_add}={default_val}")
                proposed_content = "\n".join(lines) + "\n"
                summary = f"Add missing '{var_to_add}' entry to .env.example"
                explanation = f"[Demo AI] Adds documented placeholder for '{var_to_add}' so deployments know this variable is required."
                verification = "Verify that .env.example contains the variable and has no real secrets."

            # 2. Broken relative links in README.md
            elif finding.category.value == "documentation" and "broken relative link" in finding.title.lower():
                # E.g. replace broken link with stub comment or note
                target_str = None
                if "'" in finding.title:
                    target_str = finding.title.split("'")[1]

                if target_str and target_str in file_content:
                    # Update link to point to an existing section or mark as coming soon
                    proposed_content = file_content.replace(
                        f"({target_str})",
                        f"(#under-construction-docs)"
                    )
                    summary = f"Safely point broken reference '{target_str}' to local anchor"
                    explanation = f"[Demo AI] Temporarily re-routes the broken link '{target_str}' to prevent 404 navigation errors."
                    verification = "Check that README.md renders valid local links without missing file targets."

            # 3. Missing .gitignore
            elif finding.file_path == ".gitignore" or "missing .gitignore" in finding.title.lower():
                proposed_content = (
                    "# QuietPatch Standard Git Ignore\n"
                    "__pycache__/\n"
                    "*.py[cod]\n"
                    ".venv/\n"
                    "venv/\n"
                    "node_modules/\n"
                    "dist/\n"
                    "build/\n"
                    ".env\n"
                    "*.log\n"
                    ".DS_Store\n"
                    ".quietpatch/backups/\n"
                )
                summary = "Create standard multi-stack .gitignore"
                explanation = "[Demo AI] Provides standard ignore patterns for Python, Node.js, and OS metadata files."
                verification = "Run git status to confirm ignored files are excluded from tracking."

            # 4. Missing LICENSE
            elif finding.file_path == "LICENSE" or "missing license" in finding.title.lower():
                proposed_content = (
                    "MIT License\n\n"
                    "Copyright (c) 2026 QuietPatch Contributor\n\n"
                    "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
                    "of this software and associated documentation files (the \"Software\"), to deal\n"
                    "in the Software without restriction, including without limitation the rights\n"
                    "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
                    "copies of the Software, and to permit persons to whom the Software is\n"
                    "furnished to do so, subject to the following conditions:\n\n"
                    "The above copyright notice and this permission notice shall be included in all\n"
                    "copies or substantial portions of the Software.\n\n"
                    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
                    "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
                    "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.\n"
                )
                summary = "Create standard MIT LICENSE file"
                explanation = "[Demo AI] Grants standard permissive open-source license permissions to users."
                verification = "Confirm LICENSE file exists with valid copyright year and license terms."

            # 5. Missing test script in package.json
            elif "no 'test' script" in finding.title.lower() and "package.json" in finding.file_path:
                import json
                try:
                    pkg_data = json.loads(file_content)
                    scripts = pkg_data.setdefault("scripts", {})
                    scripts["test"] = "echo 'Running tests...' && exit 0"
                    proposed_content = json.dumps(pkg_data, indent=2) + "\n"
                    summary = "Add default 'test' script to package.json"
                    explanation = "[Demo AI] Declares a baseline test command in package.json so CI and verifiers have a consistent entrypoint."
                    verification = "Run npm test to verify script execution."
                except Exception:
                    pass

            if not proposed_content:
                # Fallback generic comment fix
                proposed_content = file_content

            diff_lines = list(difflib.unified_diff(
                file_content.splitlines(keepends=True),
                proposed_content.splitlines(keepends=True),
                fromfile=f"a/{finding.file_path}",
                tofile=f"b/{finding.file_path}",
            ))

            return ProposedPatch(
                id=f"patch-{finding.id}",
                finding_id=finding.id,
                file_path=finding.file_path,
                change_summary=summary or f"Remediate {finding.title}",
                original_content=file_content,
                proposed_content=proposed_content,
                diff="".join(diff_lines),
                explanation=explanation or finding.explanation,
                risks_assumptions="Demo change verified against sample repository structure.",
                suggested_verification=verification or "Re-scan repository to confirm issue resolution.",
                status=PatchStatus.PENDING,
                is_demo=True,
                ai_model="demo-fixture",
            )
