"""QuietPatch Flask Application and REST API."""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from backend.ai.agent_service import AgentService
from backend.config import Config
from backend.models import MaintenanceRun, PatchStatus
from backend.services.patch_service import PatchError, PatchService
from backend.services.report_service import ReportService
from backend.services.repository_service import RepositoryService, SecurityError
from backend.services.verification_service import VerificationService
from backend.telemetry.tracing import get_observability_status, init_sentry

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("quietpatch.api")

def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(Config.FRONTEND_DIST))
    CORS(app)

    # Initialize Sentry with Flask integration
    init_sentry(
        app=app,
        dsn=Config.SENTRY_DSN,
        environment=Config.SENTRY_ENVIRONMENT,
        traces_sample_rate=Config.SENTRY_TRACES_SAMPLE_RATE,
    )

    # In-memory storage for active sessions/runs (ephemeral & thread-safe for MVP)
    runs_store: Dict[str, MaintenanceRun] = {}
    repo_service = RepositoryService()
    patch_service = PatchService()
    verification_service = VerificationService()

    # Pre-populate sample run so the first dashboard visit is immediately populated!
    # During automated testing (pytest), skip pre-population to keep test execution fast and deterministic
    is_reloader_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if not os.environ.get("PYTEST_CURRENT_TEST") and not app.config.get("TESTING") and (is_reloader_child or not app.debug):
        import threading

        def _init_sample_run():
            try:
                if Config.SAMPLE_REPO_PATH.exists():
                    sample_agent = AgentService(provider_name=Config.AI_PROVIDER)
                    initial_run = sample_agent.execute_maintenance_run(
                        repo_path=Config.SAMPLE_REPO_PATH,
                        repo_name="NovaTask (Sample Project)",
                        repo_type="sample",
                        run_id="run-sample-01",
                    )
                    runs_store[initial_run.id] = initial_run
                    logger.info(
                        "Initialized default sample run '%s' with %d findings (provider=%s).",
                        initial_run.id,
                        len(initial_run.findings),
                        Config.AI_PROVIDER,
                    )
            except Exception as e:
                logger.warning("Could not pre-populate sample run: %s", e)

        threading.Thread(target=_init_sample_run, daemon=True, name="quietpatch-sample-init").start()

    # =========================================================================
    # System and Config Endpoints
    # =========================================================================

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Render health check endpoint."""
        return jsonify({
            "status": "ok",
            "app": "QuietPatch",
            "version": "1.0.0",
            "mode": Config.APP_MODE,
            "provider": Config.AI_PROVIDER,
            "sample_repo": Config.SAMPLE_REPO_PATH.exists(),
        }), 200

    @app.route("/api/config", methods=["GET"])
    def get_public_config():
        """Returns safe environment and runtime settings (NO secrets)."""
        return jsonify(Config.get_sanitized_public_config()), 200

    @app.route("/api/providers/status", methods=["GET"])
    def get_provider_status():
        """Returns health of AI model and Sentry observability."""
        provider_name = request.args.get("provider", Config.AI_PROVIDER)
        agent = AgentService(provider_name=provider_name)
        ai_status = agent.get_provider_status()

        obs_status = get_observability_status(
            dsn=Config.SENTRY_DSN,
            environment=Config.SENTRY_ENVIRONMENT,
            sample_rate=Config.SENTRY_TRACES_SAMPLE_RATE,
        )

        return jsonify({
            "ai": ai_status,
            "observability": obs_status,
        }), 200

    # =========================================================================
    # Repository Management Endpoints
    # =========================================================================

    @app.route("/api/repositories", methods=["GET"])
    def list_repositories():
        """Lists accessible repository options."""
        return jsonify({
            "repositories": repo_service.list_recent_repositories(),
            "allowed_root": str(Config.QUIETPATCH_ALLOWED_ROOT),
        }), 200

    @app.route("/api/repositories/validate", methods=["POST"])
    def validate_repository():
        """Validates path containment or remote GitHub URL."""
        data = request.get_json() or {}
        repo_type = data.get("type", "local")
        path_or_url = data.get("path", "").strip()

        try:
            if repo_type == "github":
                owner, repo = repo_service.validate_github_url(path_or_url)
                return jsonify({
                    "valid": True,
                    "type": "github",
                    "owner": owner,
                    "repo": repo,
                    "message": f"Valid public repository: {owner}/{repo}",
                }), 200
            else:
                validated_path = repo_service.validate_local_path(path_or_url)
                return jsonify({
                    "valid": True,
                    "type": "local",
                    "resolved_path": str(validated_path),
                    "message": f"Valid local repository at {validated_path}",
                }), 200
        except SecurityError as se:
            return jsonify({
                "valid": False,
                "error": str(se),
            }), 400

    # =========================================================================
    # Maintenance Runs & Scan Endpoints
    # =========================================================================

    @app.route("/api/scans", methods=["POST"])
    def create_scan():
        """Starts a full caretaker maintenance run on a target repository."""
        data = request.get_json() or {}
        repo_type = data.get("type", "local")
        path_or_url = data.get("path", "").strip()
        provider_name = data.get("provider", Config.AI_PROVIDER)

        try:
            if repo_type == "sample":
                resolved_path = Config.SAMPLE_REPO_PATH
                repo_name = "NovaTask (Sample Project)"
            elif repo_type == "github":
                resolved_path = repo_service.download_public_github_repo(path_or_url)
                owner, repo = repo_service.validate_github_url(path_or_url)
                repo_name = f"{owner}/{repo}"
            else:
                resolved_path = repo_service.validate_local_path(path_or_url)
                repo_name = resolved_path.name
        except SecurityError as se:
            return jsonify({"error": str(se)}), 400

        agent = AgentService(provider_name=provider_name)
        run = agent.execute_maintenance_run(
            repo_path=resolved_path,
            repo_name=repo_name,
            repo_type=repo_type,
        )

        runs_store[run.id] = run
        return jsonify(run.model_dump()), 201

    @app.route("/api/demo/scan", methods=["POST"])
    def start_demo_scan():
        """Fast 1-click demonstration scan of bundled sample repository."""
        agent = AgentService(provider_name="demo")
        run = agent.execute_maintenance_run(
            repo_path=Config.SAMPLE_REPO_PATH,
            repo_name="NovaTask (Sample Project)",
            repo_type="sample",
        )
        runs_store[run.id] = run
        return jsonify(run.model_dump()), 200

    @app.route("/api/scans", methods=["GET"])
    def list_scans():
        """Returns all completed or active maintenance runs."""
        sorted_runs = sorted(
            runs_store.values(),
            key=lambda r: r.created_at,
            reverse=True,
        )
        return jsonify([r.model_dump() for r in sorted_runs]), 200

    @app.route("/api/scans/<run_id>", methods=["GET"])
    def get_scan(run_id: str):
        """Returns status, findings, and patches for a specific run."""
        run = runs_store.get(run_id)
        if not run:
            return jsonify({"error": f"Scan run '{run_id}' not found."}), 404
        return jsonify(run.model_dump()), 200

    @app.route("/api/scans/<run_id>/findings", methods=["GET"])
    def get_scan_findings(run_id: str):
        """Returns findings for a specific run with optional severity/category filtering."""
        run = runs_store.get(run_id)
        if not run:
            return jsonify({"error": f"Scan run '{run_id}' not found."}), 404

        sev = request.args.get("severity")
        cat = request.args.get("category")

        findings = run.findings
        if sev:
            findings = [f for f in findings if f.severity.value == sev]
        if cat:
            findings = [f for f in findings if f.category.value == cat]

        return jsonify([f.model_dump() for f in findings]), 200

    @app.route("/api/scans/<run_id>/report", methods=["GET"])
    def get_scan_report(run_id: str):
        """Generates and returns structured maintenance report."""
        run = runs_store.get(run_id)
        if not run:
            return jsonify({"error": f"Scan run '{run_id}' not found."}), 404
        report = ReportService.generate_report(run)
        return jsonify(report.model_dump()), 200

    @app.route("/api/scans/<run_id>/report/markdown", methods=["GET"])
    def export_scan_report_markdown(run_id: str):
        """Exports the maintenance report as GitHub Flavored Markdown."""
        run = runs_store.get(run_id)
        if not run:
            return jsonify({"error": f"Scan run '{run_id}' not found."}), 404
        report = ReportService.generate_report(run)
        md_text = ReportService.export_markdown(report)
        return md_text, 200, {"Content-Type": "text/markdown; charset=utf-8"}

    @app.route("/api/scans/<run_id>/retry-ai", methods=["POST"])
    def retry_scan_ai(run_id: str):
        """Retries AI analysis and patch proposal on an existing maintenance run without re-scanning repository."""
        run = runs_store.get(run_id)
        if not run:
            return jsonify({"error": f"Scan run '{run_id}' not found."}), 404

        agent = AgentService(provider_name=run.ai_provider)
        updated_run = agent.retry_ai_analysis(run)
        runs_store[run_id] = updated_run
        return jsonify(updated_run.model_dump()), 200

    # =========================================================================
    # Fix Queue & Patch Management Endpoints
    # =========================================================================

    @app.route("/api/findings/<finding_id>/propose-fix", methods=["POST"])
    def propose_fix_for_finding(finding_id: str):
        """Generates a targeted patch for a specific finding."""
        target_run: Optional[MaintenanceRun] = None
        target_finding = None

        for r in runs_store.values():
            for f in r.findings:
                if f.id == finding_id:
                    target_run = r
                    target_finding = f
                    break
            if target_finding:
                break

        if not target_finding or not target_run:
            return jsonify({"error": f"Finding '{finding_id}' not found in any active run."}), 404

        repo_path = Path(target_run.repository_path)
        file_path = repo_path / target_finding.file_path
        file_content = ""
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                    file_content = fh.read()
            except Exception:
                pass

        agent = AgentService(provider_name=target_run.ai_provider)
        patch = agent.provider.propose_patch(
            finding=target_finding,
            file_content=file_content,
            repo_path=repo_path,
            run_id=target_run.id,
        )

        if not patch:
            return jsonify({"error": "AI provider was unable to generate a valid patch."}), 422

        # Add or update patch in run
        existing_idx = next((i for i, p in enumerate(target_run.patches) if p.id == patch.id), -1)
        if existing_idx >= 0:
            target_run.patches[existing_idx] = patch
        else:
            target_run.patches.append(patch)

        return jsonify(patch.model_dump()), 200

    @app.route("/api/patches/<patch_id>/approve", methods=["POST"])
    def approve_patch(patch_id: str):
        for run in runs_store.values():
            for patch in run.patches:
                if patch.id == patch_id:
                    patch.status = PatchStatus.APPROVED
                    return jsonify(patch.model_dump()), 200
        return jsonify({"error": f"Patch '{patch_id}' not found."}), 404

    @app.route("/api/patches/<patch_id>/reject", methods=["POST"])
    def reject_patch(patch_id: str):
        for run in runs_store.values():
            for patch in run.patches:
                if patch.id == patch_id:
                    patch.status = PatchStatus.REJECTED
                    return jsonify(patch.model_dump()), 200
        return jsonify({"error": f"Patch '{patch_id}' not found."}), 404

    @app.route("/api/patches/<patch_id>/apply", methods=["POST"])
    def apply_patch_endpoint(patch_id: str):
        """Applies an approved patch locally."""
        for run in runs_store.values():
            for patch in run.patches:
                if patch.id == patch_id:
                    repo_path = Path(run.repository_path)
                    try:
                        updated_patch = patch_service.apply_patch(
                            repo_path=repo_path,
                            patch=patch,
                            run_id=run.id,
                        )
                        return jsonify(updated_patch.model_dump()), 200
                    except PatchError as pe:
                        return jsonify({"error": str(pe)}), 400

        return jsonify({"error": f"Patch '{patch_id}' not found."}), 404

    @app.route("/api/patches/<patch_id>/undo", methods=["POST"])
    def undo_patch_endpoint(patch_id: str):
        """Reverts an applied patch."""
        for run in runs_store.values():
            for patch in run.patches:
                if patch.id == patch_id:
                    repo_path = Path(run.repository_path)
                    try:
                        undone_patch = patch_service.undo_patch(
                            repo_path=repo_path,
                            patch=patch,
                            run_id=run.id,
                        )
                        return jsonify(undone_patch.model_dump()), 200
                    except PatchError as pe:
                        return jsonify({"error": str(pe)}), 400

        return jsonify({"error": f"Patch '{patch_id}' not found."}), 404

    @app.route("/api/patches/<patch_id>/verify", methods=["POST"])
    def verify_patch_endpoint(patch_id: str):
        """Runs deterministic verification for an applied patch."""
        for run in runs_store.values():
            for patch in run.patches:
                if patch.id == patch_id:
                    repo_path = Path(run.repository_path)
                    # Find associated finding
                    finding = next((f for f in run.findings if f.id == patch.finding_id), None)
                    if not finding:
                        return jsonify({"error": "Associated finding not found for verification."}), 404

                    result = verification_service.verify_finding_resolution(
                        repo_path=repo_path,
                        finding=finding,
                        patch=patch,
                        run_id=run.id,
                    )
                    return jsonify(result), 200

        return jsonify({"error": f"Patch '{patch_id}' not found."}), 404

    @app.route("/api/verify/test-command", methods=["POST"])
    def run_test_command_endpoint():
        """Executes an allowlisted test command if approved by user."""
        data = request.get_json() or {}
        run_id = data.get("run_id")
        command_key = data.get("command", "pytest")

        run = runs_store.get(run_id)
        if not run:
            return jsonify({"error": "Run not found."}), 404

        repo_path = Path(run.repository_path)
        result = verification_service.execute_approved_test_command(
            repo_path=repo_path,
            command_key=command_key,
            run_id=run.id,
        )
        return jsonify(result), 200

    # =========================================================================
    # Production Frontend Static Asset Serving
    # =========================================================================

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        """Serves built React assets in production with SPA fallback to index.html."""
        if path.startswith("api/"):
            return jsonify({"error": "Endpoint not found"}), 404

        dist = Config.FRONTEND_DIST
        target_path = dist / path
        if path and target_path.exists() and target_path.is_file():
            return send_from_directory(str(dist), path)

        index_file = dist / "index.html"
        if index_file.exists():
            return send_from_directory(str(dist), "index.html")

        return jsonify({
            "message": "QuietPatch Backend is running.",
            "note": "Frontend build assets not found in frontend/dist. In development, run 'npm run dev' in frontend/.",
            "mode": Config.APP_MODE,
            "provider": Config.AI_PROVIDER,
        }), 200

    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)
