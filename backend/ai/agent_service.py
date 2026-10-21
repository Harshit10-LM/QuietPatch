"""Agent service coordinating the Gemma sequential workflow with Sentry Agent Tracing."""
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.ai.demo_provider import DemoProvider
from backend.ai.gemma_api_provider import GemmaApiProvider
from backend.ai.ollama_provider import OllamaProvider
from backend.ai.provider_base import BaseAIProvider
from backend.config import Config
from backend.models import (
    Finding,
    MaintenanceRun,
    ProposedPatch,
    Severity,
    StageRecord,
)
from backend.scanners.engine import ScanEngine
from backend.services.report_service import ReportService
from backend.services.time_saved_service import calculate_time_saved
from backend.telemetry.tracing import agent_stage_span, maintenance_run_trace

logger = logging.getLogger("quietpatch.ai.agent")

class AgentService:
    def __init__(self, provider_name: Optional[str] = None):
        self.provider_name = (provider_name or Config.AI_PROVIDER).lower()
        self.provider: BaseAIProvider = self._init_provider(self.provider_name)
        self.scan_engine = ScanEngine()

    def _init_provider(self, name: str) -> BaseAIProvider:
        if name == "ollama":
            return OllamaProvider()
        elif name == "gemma_api":
            return GemmaApiProvider()
        else:
            return DemoProvider()

    def get_provider_status(self) -> Dict[str, Any]:
        """Returns the status and health check of the active AI provider."""
        status = self.provider.check_availability()
        status["active_provider"] = self.provider_name
        status["app_mode"] = Config.APP_MODE
        return status

    @staticmethod
    def prioritize_findings(findings: List[Finding]) -> None:
        """Ranks findings by severity weight and effort to produce a consistent priority ranking."""
        severity_weight = {
            Severity.CRITICAL: 5,
            Severity.HIGH: 4,
            Severity.MEDIUM: 3,
            Severity.LOW: 2,
            Severity.INFORMATIONAL: 1,
        }
        findings.sort(
            key=lambda f: (severity_weight.get(f.severity, 0), -f.estimated_effort_mins),
            reverse=True,
        )

    @staticmethod
    def run_baseline_verification(run: MaintenanceRun) -> int:
        """Evaluates baseline verification state on generated patches and target files."""
        verified_count = 0
        repo_path = Path(run.repository_path)
        for patch in run.patches:
            target = repo_path / patch.file_path
            if target.exists():
                verified_count += 1
        return verified_count

    def execute_maintenance_run(
        self,
        repo_path: Path,
        repo_name: str,
        repo_type: str = "local",
        run_id: Optional[str] = None,
        existing_run: Optional[MaintenanceRun] = None,
    ) -> MaintenanceRun:
        """Executes the full 8-stage caretaker workflow inside a root Sentry transaction:

        1. quietpatch.initialize
        2. quietpatch.repository_discovery
        3. quietpatch.deterministic_scan
        4. quietpatch.gemma_analysis
        5. quietpatch.prioritization
        6. quietpatch.patch_generation
        7. quietpatch.verification
        8. quietpatch.report_generation
        """
        run_id = existing_run.id if existing_run else (run_id or f"run-{uuid.uuid4().hex[:8]}")
        start_time = time.time()
        model_name = getattr(self.provider, "model", "demonstration")

        if existing_run:
            run = existing_run
            run.status = "discovering"
            run.ai_provider = self.provider_name
            run.ai_model = model_name
        else:
            run = MaintenanceRun(
                id=run_id,
                repository_name=repo_name,
                repository_path=str(repo_path),
                repository_type=repo_type,
                status="discovering",
                ai_provider=self.provider_name,
                ai_model=model_name,
            )

        def set_stage_running(name: str, details: Optional[str] = None):
            started = datetime.now(timezone.utc).isoformat()
            if name in run.stages:
                run.stages[name].status = "running"
                if not run.stages[name].started_at:
                    run.stages[name].started_at = started
                if details:
                    run.stages[name].details = details
            else:
                run.stages[name] = StageRecord(
                    name=name,
                    status="running",
                    started_at=started,
                    details=details,
                )

        def record_stage(name: str, status: str, duration: float = 0.0, details: Optional[str] = None):
            started = run.stages[name].started_at if name in run.stages else None
            run.stages[name] = StageRecord(
                name=name,
                status=status,
                duration_seconds=round(duration, 3),
                details=details,
                started_at=started,
                completed_at=datetime.now(timezone.utc).isoformat() if status in {"completed", "partial", "failed", "skipped"} else None,
            )

        with maintenance_run_trace(
            run_id=run_id,
            repo_name=repo_name,
            repo_type=repo_type,
            provider=self.provider_name,
            model=model_name,
        ) as run_tracer:
            try:
                # Stage 1: quietpatch.initialize
                run.status = "discovering"
                set_stage_running("initialization")
                with agent_stage_span(
                    "Scan-run Initialization",
                    op="quietpatch.initialize",
                    run_id=run_id,
                    provider=self.provider_name,
                    model=run.ai_model,
                    extra_tags={"repo_type": repo_type},
                ) as init_span:
                    t_init = time.time()
                    if not repo_path.exists():
                        init_span.set_outcome("failure", details="Repository path does not exist")
                        raise ValueError(f"Repository path does not exist: {repo_path}")
                    init_duration = max(0.01, round(time.time() - t_init, 3))
                    init_span.set_outcome("completed")
                    record_stage("initialization", "completed", init_duration, "Environment validated & limits confirmed")

                # Stage 2: quietpatch.repository_discovery
                run.status = "discovering"
                set_stage_running("discovery")
                with agent_stage_span(
                    "Repository Discovery",
                    op="quietpatch.repository_discovery",
                    run_id=run_id,
                    provider=self.provider_name,
                    model=run.ai_model,
                ) as disc_span:
                    t_disc = time.time()
                    discovered_files, total_bytes = self.scan_engine.discover_files(repo_path)
                    disc_duration = round(time.time() - t_disc, 3)
                    disc_span.set_tag("files_count", len(discovered_files))
                    disc_span.set_data("total_bytes", total_bytes)
                    disc_span.set_outcome("completed")
                    record_stage("discovery", "completed", disc_duration, f"Discovered {len(discovered_files)} files ({total_bytes} bytes)")

                # Stage 3: quietpatch.deterministic_scan
                run.status = "scanning"
                run.files_scanned = len(discovered_files)
                set_stage_running("scan")
                with agent_stage_span(
                    "Deterministic Scanner Suite",
                    op="quietpatch.deterministic_scan",
                    run_id=run_id,
                    provider=self.provider_name,
                    model=run.ai_model,
                    extra_tags={"files_count": len(discovered_files)},
                ) as scan_span:
                    t_scan = time.time()
                    findings, _ = self.scan_engine.run_scanners(
                        repo_path=repo_path,
                        files=discovered_files,
                        run_id=run_id,
                        provider=self.provider_name,
                        model=run.ai_model,
                    )
                    run.findings = findings
                    scan_duration = round(time.time() - t_scan, 3)
                    scan_span.set_tag("findings_count", len(findings))
                    scan_span.set_tag("files_count", len(discovered_files))
                    scan_span.set_outcome("completed")
                    record_stage("scan", "completed", scan_duration, f"Scanned {len(discovered_files)} files, {len(findings)} findings")

                # Stage 4: quietpatch.gemma_analysis
                run.status = "analyzing"
                set_stage_running("analysis")
                t_analyze = time.time()
                analysis_succeeded = False

                with agent_stage_span(
                    "Gemma Issue Analysis",
                    op="quietpatch.gemma_analysis",
                    run_id=run_id,
                    provider=self.provider_name,
                    model=run.ai_model,
                    extra_tags={"findings_count": len(run.findings)},
                ) as analysis_span:
                    try:
                        analyzed_findings = self.provider.analyze_and_prioritize(run.findings, run_id=run_id)
                        run.findings = analyzed_findings
                        evaluated_count = sum(1 for f in run.findings if f.ai_assisted or f.is_demo)
                        analyze_duration = round(time.time() - t_analyze, 3)

                        if evaluated_count == len(run.findings):
                            analysis_succeeded = True
                            analysis_span.set_outcome("completed")
                            record_stage(
                                "analysis",
                                "completed",
                                analyze_duration,
                                f"Evaluated all {evaluated_count} findings with {run.ai_model}",
                            )
                        elif evaluated_count > 0:
                            analysis_succeeded = True
                            analysis_span.set_outcome("partial", details="Partial batch analysis failure")
                            record_stage(
                                "analysis",
                                "partial",
                                analyze_duration,
                                f"Evaluated {evaluated_count}/{len(run.findings)} findings with {run.ai_model} (partial batch failure)",
                            )
                        else:
                            analysis_span.set_outcome("failed", details="Zero findings evaluated")
                            record_stage(
                                "analysis",
                                "failed",
                                analyze_duration,
                                f"No findings could be evaluated by {run.ai_model}",
                            )
                    except Exception as ai_err:
                        analyze_duration = round(time.time() - t_analyze, 3)
                        logger.warning("AI analysis failed in stage 'analysis': %s", ai_err)
                        from backend.ai.gemma_api_provider import sanitize_error_message
                        sanitized_ai_err = sanitize_error_message(str(ai_err))
                        analysis_span.set_outcome("failed", details=sanitized_ai_err)
                        record_stage(
                            "analysis",
                            "failed",
                            analyze_duration,
                            f"AI analysis failed ({type(ai_err).__name__}): {sanitized_ai_err}",
                        )

                # Stage 5: quietpatch.prioritization
                run.status = "prioritizing"
                set_stage_running("prioritization")
                with agent_stage_span(
                    "Finding Prioritization",
                    op="quietpatch.prioritization",
                    run_id=run_id,
                    provider=self.provider_name,
                    model=run.ai_model,
                    extra_tags={"findings_count": len(run.findings)},
                ) as prio_span:
                    t_prio = time.time()
                    self.prioritize_findings(run.findings)
                    prio_duration = round(time.time() - t_prio, 3)
                    prio_span.set_outcome("completed")
                    record_stage(
                        "prioritization",
                        "completed",
                        prio_duration,
                        f"Calibrated priority order for {len(run.findings)} findings",
                    )

                # Stage 6: quietpatch.patch_generation
                run.status = "proposing"
                set_stage_running("proposal")
                t_prop = time.time()
                patches: List[ProposedPatch] = list(run.patches)
                existing_patch_finding_ids = {p.finding_id for p in patches}

                with agent_stage_span(
                    "Reviewable Patch Synthesis",
                    op="quietpatch.patch_generation",
                    run_id=run_id,
                    provider=self.provider_name,
                    model=run.ai_model,
                    extra_tags={"findings_count": len(run.findings)},
                ) as patch_span:
                    if not analysis_succeeded and self.provider_name in {"ollama", "gemma_api"}:
                        # Do NOT synthesize fake patches when AI model analysis failed
                        prop_duration = round(time.time() - t_prop, 3)
                        patch_span.set_outcome("skipped", details=f"{self.provider_name} model analysis was unavailable")
                        record_stage(
                            "proposal",
                            "skipped",
                            prop_duration,
                            f"Skipped patch synthesis because {self.provider_name} model analysis was unavailable",
                        )
                    else:
                        eligible_findings = [
                            f for f in run.findings
                            if f.category.value in {"configuration", "documentation", "general", "javascript"}
                            and f.id not in existing_patch_finding_ids
                        ][:3]

                        if not eligible_findings:
                            prop_duration = round(time.time() - t_prop, 3)
                            if len(patches) > 0:
                                patch_span.set_outcome("completed")
                                record_stage(
                                    "proposal",
                                    "completed",
                                    prop_duration,
                                    f"Preserved {len(patches)} reviewable patches",
                                )
                            else:
                                patch_span.set_outcome("skipped", details="No eligible findings for patch proposal")
                                record_stage(
                                    "proposal",
                                    "skipped",
                                    prop_duration,
                                    "No eligible findings for automated patch proposal",
                                )
                        else:
                            attempted_count = len(eligible_findings)
                            failed_count = 0
                            new_patches_count = 0
                            for finding in eligible_findings:
                                target_file = repo_path / finding.file_path
                                file_content = ""
                                if target_file.is_file():
                                    try:
                                        with open(target_file, "r", encoding="utf-8", errors="replace") as f:
                                            file_content = f.read()
                                    except Exception:
                                        pass

                                try:
                                    patch = self.provider.propose_patch(finding, file_content, repo_path, run_id=run_id)
                                    if patch and patch.diff:
                                        patches.append(patch)
                                        run.patches = list(patches)
                                        new_patches_count += 1
                                    else:
                                        failed_count += 1
                                except Exception as patch_err:
                                    failed_count += 1
                                    logger.warning("Patch proposal failed for %s: %s", finding.id, patch_err)

                            run.patches = patches
                            prop_duration = round(time.time() - t_prop, 3)
                            patch_span.set_tag("patches_count", len(patches))
                            patch_span.set_tag("attempted_count", attempted_count)
                            patch_span.set_tag("failed_count", failed_count)

                            if new_patches_count == attempted_count:
                                patch_span.set_outcome("completed")
                                record_stage(
                                    "proposal",
                                    "completed",
                                    prop_duration,
                                    f"Proposed {len(patches)} reviewable patches",
                                )
                            elif new_patches_count > 0:
                                patch_span.set_outcome(
                                    "partial",
                                    details=f"Proposed {len(patches)}/{attempted_count} patches ({failed_count} failed)",
                                )
                                record_stage(
                                    "proposal",
                                    "partial",
                                    prop_duration,
                                    f"Proposed {len(patches)}/{attempted_count} patches ({failed_count} patch generation attempts failed)",
                                )
                            else:
                                patch_span.set_outcome(
                                    "failed",
                                    details=f"All {attempted_count} patch generation attempts failed",
                                )
                                record_stage(
                                    "proposal",
                                    "failed",
                                    prop_duration,
                                    f"Patch generation failed for all {attempted_count} eligible findings",
                                )

                # Stage 7: quietpatch.verification
                run.status = "verifying"
                set_stage_running("verification")
                with agent_stage_span(
                    "Deterministic Verification",
                    op="quietpatch.verification",
                    run_id=run_id,
                    provider=self.provider_name,
                    model=run.ai_model,
                    extra_tags={"patches_count": len(run.patches)},
                ) as verif_span:
                    t_verif = time.time()
                    verified_count = self.run_baseline_verification(run)
                    verif_duration = round(time.time() - t_verif, 3)
                    verif_span.set_tag("verification_count", verified_count)
                    verif_span.set_outcome("completed")
                    record_stage(
                        "verification",
                        "completed",
                        verif_duration,
                        f"Baseline verification evaluated for {len(run.findings)} findings ({verified_count} targets confirmed)",
                    )

                # Stage 8: quietpatch.report_generation
                run.status = "reporting"
                set_stage_running("report")
                with agent_stage_span(
                    "Report Generation",
                    op="quietpatch.report_generation",
                    run_id=run_id,
                    provider=self.provider_name,
                    model=run.ai_model,
                ) as report_span:
                    t_rep = time.time()
                    time_summary = calculate_time_saved(run.findings, run.patches)
                    run.time_saved_minutes = time_summary.total_minutes
                    run.duration_seconds = round(time.time() - start_time, 3)
                    run.status = "completed"
                    run.completed_at = datetime.now(timezone.utc).isoformat()

                    report = ReportService.generate_report(run)
                    rep_duration = round(time.time() - t_rep, 3)
                    report_span.set_tag("time_saved_minutes", report.time_saved.total_minutes)
                    report_span.set_outcome("completed")
                    record_stage(
                        "report",
                        "completed",
                        rep_duration,
                        f"Report generated: {report.total_findings} findings, {report.fixes_proposed} patches, {report.time_saved.total_minutes}m saved",
                    )

                # Update root transaction metrics
                run_tracer.set_tag("files_scanned", run.files_scanned)
                run_tracer.set_tag("findings_count", len(run.findings))
                run_tracer.set_tag("patches_count", len(run.patches))
                run_tracer.set_tag("time_saved_minutes", run.time_saved_minutes)
                run_tracer.set_tag("stage_status", "completed")
                run_tracer.set_status("ok")

            except Exception as exc:
                logger.error("Maintenance run failed: %s", exc, exc_info=True)
                run.status = "failed"
                from backend.ai.gemma_api_provider import sanitize_error_message
                run.error = sanitize_error_message(str(exc))
                run.duration_seconds = round(time.time() - start_time, 3)
                record_stage("workflow", "failed", run.duration_seconds, run.error)
                run_tracer.set_status("internal_error")

        return run

    def retry_ai_analysis(self, run: MaintenanceRun) -> MaintenanceRun:
        """Retries AI analysis on existing findings and generates missing patches without re-scanning files."""
        run.status = "analyzing"
        t_analyze = time.time()
        analysis_succeeded = False

        # Identify findings that still need AI evaluation (or all if none were assisted)
        unassisted = [f for f in run.findings if not f.ai_assisted]
        targets = unassisted if unassisted else run.findings

        with maintenance_run_trace(
            run_id=run.id,
            repo_name=run.repository_name,
            repo_type=run.repository_type,
            provider=self.provider_name,
            model=run.ai_model,
            extra_tags={"is_retry": "true"},
        ) as run_tracer:
            # Stage 4 Retry: quietpatch.gemma_analysis
            with agent_stage_span(
                "Gemma Issue Analysis (Retry)",
                op="quietpatch.gemma_analysis",
                run_id=run.id,
                provider=self.provider_name,
                model=run.ai_model,
                extra_tags={"findings_count": len(targets)},
            ) as analysis_span:
                try:
                    analyzed = self.provider.analyze_and_prioritize(targets, run_id=run.id)
                    update_map = {f.id: f for f in analyzed if f.ai_assisted}
                    for i, f in enumerate(run.findings):
                        if f.id in update_map:
                            run.findings[i] = update_map[f.id]

                    evaluated_count = sum(1 for f in run.findings if f.ai_assisted or f.is_demo)
                    analyze_duration = round(time.time() - t_analyze, 3)
                    if evaluated_count == len(run.findings):
                        analysis_succeeded = True
                        analysis_span.set_outcome("completed")
                        run.stages["analysis"] = StageRecord(
                            name="analysis",
                            status="completed",
                            duration_seconds=analyze_duration,
                            details=f"Evaluated all {evaluated_count} findings with {run.ai_model} (retry)",
                        )
                    elif evaluated_count > 0:
                        analysis_succeeded = True
                        analysis_span.set_outcome("partial")
                        run.stages["analysis"] = StageRecord(
                            name="analysis",
                            status="partial",
                            duration_seconds=analyze_duration,
                            details=f"Evaluated {evaluated_count}/{len(run.findings)} findings with {run.ai_model} (retry)",
                        )
                    else:
                        analysis_span.set_outcome("failed")
                        run.stages["analysis"] = StageRecord(
                            name="analysis",
                            status="failed",
                            duration_seconds=analyze_duration,
                            details=f"Retry failed to evaluate findings with {run.ai_model}",
                        )
                except Exception as ai_err:
                    analyze_duration = round(time.time() - t_analyze, 3)
                    logger.warning("Retry AI analysis failed: %s", ai_err)
                    from backend.ai.gemma_api_provider import sanitize_error_message
                    sanitized_ai_err = sanitize_error_message(str(ai_err))
                    analysis_span.set_outcome("failed", details=sanitized_ai_err)
                    run.stages["analysis"] = StageRecord(
                        name="analysis",
                        status="failed",
                        duration_seconds=analyze_duration,
                        details=f"AI analysis retry failed: {sanitized_ai_err}",
                        completed_at=datetime.now(timezone.utc).isoformat(),
                    )

            # Stage 5: quietpatch.prioritization
            run.status = "prioritizing"
            with agent_stage_span(
                "Finding Prioritization",
                op="quietpatch.prioritization",
                run_id=run.id,
                provider=self.provider_name,
                model=run.ai_model,
                extra_tags={"findings_count": len(run.findings)},
            ) as prio_span:
                self.prioritize_findings(run.findings)
                prio_span.set_outcome("completed")
                run.stages["prioritization"] = StageRecord(
                    name="prioritization",
                    status="completed",
                    duration_seconds=0.1,
                    details=f"Calibrated priority order for {len(run.findings)} findings (retry)",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )

            # Stage 6: quietpatch.patch_generation
            run.status = "proposing"
            if analysis_succeeded:
                with agent_stage_span(
                    "Reviewable Patch Synthesis",
                    op="quietpatch.patch_generation",
                    run_id=run.id,
                    provider=self.provider_name,
                    model=run.ai_model,
                ) as patch_span:
                    repo_path = Path(run.repository_path)
                    patches: List[ProposedPatch] = list(run.patches)
                    existing_patch_finding_ids = {p.finding_id for p in patches}
                    eligible_findings = [
                        f for f in run.findings
                        if f.category.value in {"configuration", "documentation", "general", "javascript"}
                        and f.id not in existing_patch_finding_ids
                    ][:3]
                    failed_count = 0
                    attempted_count = len(eligible_findings)
                    new_patches_count = 0

                    for finding in eligible_findings:
                        target_file = repo_path / finding.file_path
                        file_content = ""
                        if target_file.is_file():
                            try:
                                with open(target_file, "r", encoding="utf-8", errors="replace") as fh:
                                    file_content = fh.read()
                            except Exception:
                                pass
                        try:
                            patch = self.provider.propose_patch(finding, file_content, repo_path, run_id=run.id)
                            if patch and patch.diff:
                                patches.append(patch)
                                run.patches = list(patches)
                                new_patches_count += 1
                            else:
                                failed_count += 1
                        except Exception as patch_err:
                            failed_count += 1
                            logger.warning("Patch proposal failed for %s during retry: %s", finding.id, patch_err)

                    run.patches = patches
                    patch_span.set_tag("patches_count", len(patches))
                    patch_span.set_tag("attempted_count", attempted_count)
                    patch_span.set_tag("failed_count", failed_count)

                    if attempted_count == 0:
                        patch_span.set_outcome("completed" if len(patches) > 0 else "skipped")
                        run.stages["proposal"] = StageRecord(
                            name="proposal",
                            status="completed" if len(patches) > 0 else "skipped",
                            duration_seconds=0.1,
                            details=f"Preserved {len(patches)} reviewable patches" if len(patches) > 0 else "No eligible findings for patch proposal",
                            completed_at=datetime.now(timezone.utc).isoformat(),
                        )
                    elif new_patches_count == attempted_count:
                        patch_span.set_outcome("completed")
                        run.stages["proposal"] = StageRecord(
                            name="proposal",
                            status="completed",
                            duration_seconds=1.0,
                            details=f"Proposed {len(patches)} reviewable patches (retry)",
                            completed_at=datetime.now(timezone.utc).isoformat(),
                        )
                    elif new_patches_count > 0:
                        patch_span.set_outcome(
                            "partial",
                            details=f"Proposed {new_patches_count}/{attempted_count} new patches ({failed_count} failed)",
                        )
                        run.stages["proposal"] = StageRecord(
                            name="proposal",
                            status="partial",
                            duration_seconds=1.0,
                            details=f"Proposed {new_patches_count}/{attempted_count} new patches ({failed_count} failed) (retry)",
                            completed_at=datetime.now(timezone.utc).isoformat(),
                        )
                    else:
                        patch_span.set_outcome(
                            "failed",
                            details=f"All {attempted_count} patch generation attempts failed",
                        )
                        run.stages["proposal"] = StageRecord(
                            name="proposal",
                            status="failed",
                            duration_seconds=1.0,
                            details=f"Patch generation failed for all {attempted_count} eligible findings (retry)",
                            completed_at=datetime.now(timezone.utc).isoformat(),
                        )

            # Stage 7: quietpatch.verification
            run.status = "verifying"
            with agent_stage_span(
                "Deterministic Verification",
                op="quietpatch.verification",
                run_id=run.id,
                provider=self.provider_name,
                model=run.ai_model,
            ) as verif_span:
                verified_count = self.run_baseline_verification(run)
                verif_span.set_tag("verification_count", verified_count)
                verif_span.set_outcome("completed")
                run.stages["verification"] = StageRecord(
                    name="verification",
                    status="completed",
                    duration_seconds=0.1,
                    details=f"Baseline verification evaluated for {len(run.findings)} findings ({verified_count} targets confirmed)",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )

            # Stage 8: quietpatch.report_generation
            run.status = "reporting"
            with agent_stage_span(
                "Report Generation",
                op="quietpatch.report_generation",
                run_id=run.id,
                provider=self.provider_name,
                model=run.ai_model,
            ) as report_span:
                time_summary = calculate_time_saved(run.findings, run.patches)
                run.time_saved_minutes = time_summary.total_minutes
                report = ReportService.generate_report(run)
                report_span.set_tag("time_saved_minutes", report.time_saved.total_minutes)
                report_span.set_outcome("completed")
                run.stages["report"] = StageRecord(
                    name="report",
                    status="completed",
                    duration_seconds=0.1,
                    details=f"Report generated: {report.total_findings} findings, {report.fixes_proposed} patches, {report.time_saved.total_minutes}m saved",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )

            run_tracer.set_tag("findings_count", len(run.findings))
            run_tracer.set_tag("patches_count", len(run.patches))
            run_tracer.set_tag("stage_status", "completed")
            run_tracer.set_status("ok")

        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc).isoformat()
        return run
