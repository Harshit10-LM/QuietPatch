"""Maintenance report generation and export service."""
from datetime import datetime, timezone
from typing import Any, Dict, List
from backend.models import MaintenanceReport, MaintenanceRun, PatchStatus, VerificationStatus
from backend.services.time_saved_service import calculate_time_saved
from backend.telemetry.tracing import agent_stage_span

class ReportService:
    @staticmethod
    def generate_report(run: MaintenanceRun) -> MaintenanceReport:
        """Generates a complete structured maintenance report from a completed run."""
        with agent_stage_span("Report Generation", op="quietpatch.report_generation", run_id=run.id, provider=run.ai_provider, model=run.ai_model):
            findings_by_sev: Dict[str, int] = {}
            findings_by_cat: Dict[str, int] = {}

            for f in run.findings:
                sev = f.severity.value
                cat = f.category.value
                findings_by_sev[sev] = findings_by_sev.get(sev, 0) + 1
                findings_by_cat[cat] = findings_by_cat.get(cat, 0) + 1

            applied_count = sum(1 for p in run.patches if p.status == PatchStatus.APPLIED)
            proposed_count = len(run.patches)

            passed_count = sum(1 for f in run.findings if f.verification_status == VerificationStatus.PASSED)
            failed_count = sum(1 for f in run.findings if f.verification_status == VerificationStatus.FAILED)
            skipped_count = sum(1 for f in run.findings if f.verification_status == VerificationStatus.SKIPPED)

            time_saved = calculate_time_saved(run.findings, run.patches)

            remaining_work: List[Dict[str, Any]] = []
            for f in run.findings:
                if f.verification_status != VerificationStatus.PASSED:
                    remaining_work.append({
                        "id": f.id,
                        "title": f.title,
                        "severity": f.severity.value,
                        "file_path": f.file_path,
                        "action": f.suggested_action,
                    })

            stage_durations = {name: s.duration_seconds for name, s in run.stages.items()}

            return MaintenanceReport(
                run_id=run.id,
                repository_name=run.repository_name,
                repository_path=run.repository_path,
                repository_type=run.repository_type,
                generated_at=datetime.now(timezone.utc).isoformat(),
                ai_provider=run.ai_provider,
                ai_model=run.ai_model,
                total_duration_seconds=run.duration_seconds,
                files_scanned=run.files_scanned,
                total_findings=len(run.findings),
                findings_by_severity=findings_by_sev,
                findings_by_category=findings_by_cat,
                fixes_proposed=proposed_count,
                fixes_applied=applied_count,
                checks_passed=passed_count,
                checks_failed=failed_count,
                checks_skipped=skipped_count,
                time_saved=time_saved,
                remaining_work=remaining_work[:10],
                stage_durations=stage_durations,
            )

    @staticmethod
    def export_markdown(report: MaintenanceReport) -> str:
        """Formats the maintenance report as clean, publication-ready GitHub Flavored Markdown."""
        lines = [
            f"# QuietPatch Maintenance Report: {report.repository_name}",
            f"*Generated: {report.generated_at} | Run ID: `{report.run_id}`*",
            "",
            "## Executive Summary",
            f"- **Repository Type:** {report.repository_type.title()}",
            f"- **AI Caretaker:** `{report.ai_provider}` ({report.ai_model})",
            f"- **Files Scanned:** {report.files_scanned}",
            f"- **Total Findings:** {report.total_findings}",
            f"- **Fixes Proposed / Applied:** {report.fixes_proposed} / {report.fixes_applied}",
            f"- **Estimated Time Saved:** ~{report.time_saved.total_hours} hours ({report.time_saved.total_minutes} mins)",
            f"- **Scan & Analysis Duration:** {report.total_duration_seconds:.2f}s",
            "",
            "## Findings by Severity",
            "| Severity | Count |",
            "|---|---|",
        ]

        for sev in ["critical", "high", "medium", "low", "informational"]:
            if sev in report.findings_by_severity:
                lines.append(f"| **{sev.upper()}** | {report.findings_by_severity[sev]} |")

        lines.extend([
            "",
            "## Findings by Category",
            "| Category | Count |",
            "|---|---|",
        ])
        for cat, count in report.findings_by_category.items():
            lines.append(f"| {cat.title()} | {count} |")

        lines.extend([
            "",
            "## Verification & Outcomes",
            f"- **Passed Checks:** {report.checks_passed}",
            f"- **Failed Checks:** {report.checks_failed}",
            f"- **Skipped Checks:** {report.checks_skipped}",
            "",
            "## Stage Performance",
            "| Stage | Duration |",
            "|---|---|",
        ])
        for stage, dur in report.stage_durations.items():
            lines.append(f"| {stage.title()} | {dur:.3f}s |")

        lines.extend([
            "",
            "## Remaining Work Requiring Human Review",
        ])
        if report.remaining_work:
            for item in report.remaining_work:
                lines.append(f"- **[{item['severity'].upper()}]** `{item['file_path']}`: {item['title']}")
                lines.append(f"  *Suggested Action:* {item['action']}")
        else:
            lines.append("*All identified findings have been resolved and verified!*")

        lines.extend([
            "",
            "---",
            "*Report generated deterministically by [QuietPatch](https://github.com/quietpatch/quietpatch) — Slow and Steady repository care.*"
        ])

        return "\n".join(lines)
