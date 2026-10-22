"""Sentry Agent Tracing Integration and telemetry sanitization for QuietPatch."""
import logging
import re
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from backend.config import Config

logger = logging.getLogger("quietpatch.telemetry")

_sentry_initialized = False

# Strict allowlist of tags allowed to be sent to Sentry
SAFE_TAG_KEYS = {
    "run_id", "stage", "stage_name", "stage_status", "op",
    "provider", "model", "outcome", "error_category", "error_type",
    "files_count", "files_scanned", "findings_count", "patches_count",
    "verification_count", "repo_type", "repo_name", "app_mode",
    "command", "duration_seconds", "time_saved_minutes", "batch_index",
    "total_batches", "eligible_count", "patches_verified", "checks_passed",
    "has_response", "candidate_count", "text_part_count", "thought_part_count",
    "finish_reason", "block_reason", "failure_category",
    "prompt_tokens", "candidate_tokens", "total_tokens",
    "recovery_attempt_count", "recovery_attempted", "recovery_outcome",
    "attempted_count", "failed_count", "successful_batches", "failed_batches",
    "is_retry", "is_transient", "exhausted_attempts", "error_code",
    "successful_attempt", "time_budget_exhausted", "finding_id",
    "patch_budget_seconds", "remaining_budget", "request_timeout_ms", "recovery_tokens",
}

# Mapping of human-friendly stage names to required quietpatch span operations
STAGE_NAME_TO_OP = {
    "Scan-run Initialization": "quietpatch.initialize",
    "initialize": "quietpatch.initialize",
    "Repository Discovery": "quietpatch.repository_discovery",
    "discovery": "quietpatch.repository_discovery",
    "Deterministic Scanner Suite": "quietpatch.deterministic_scan",
    "Deterministic Scan": "quietpatch.deterministic_scan",
    "scan": "quietpatch.deterministic_scan",
    "Gemma Issue Analysis": "quietpatch.gemma_analysis",
    "Gemma Issue Analysis (Retry)": "quietpatch.gemma_analysis",
    "Gemma AI Analysis": "quietpatch.gemma_analysis",
    "analysis": "quietpatch.gemma_analysis",
    "Finding Prioritization": "quietpatch.prioritization",
    "prioritization": "quietpatch.prioritization",
    "Reviewable Patch Synthesis": "quietpatch.patch_generation",
    "Gemma Patch Generation": "quietpatch.patch_generation",
    "Patch Validation & Application": "quietpatch.patch_generation",
    "Patch Rollback & Restore": "quietpatch.patch_generation",
    "proposal": "quietpatch.patch_generation",
    "Deterministic Verification": "quietpatch.verification",
    "Deterministic Verification Check": "quietpatch.verification",
    "Allowlisted Test Command Execution": "quietpatch.verification",
    "verification": "quietpatch.verification",
    "Report Generation": "quietpatch.report_generation",
    "report": "quietpatch.report_generation",
}

def sanitize_scalar(val: Any) -> str:
    """Sanitizes an arbitrary value by scrubbing filesystem paths, API keys, and sensitive tokens."""
    if val is None:
        return ""
    s = str(val).strip()
    # Scrub local absolute filesystem paths (/Users/..., /home/..., /root/..., /tmp/..., C:\...)
    s = re.sub(r'(?:/(?:Users|home|root|tmp|private|var)/|[A-Za-z]:\\)[^\s,;:"\'\]\)]+', '[REDACTED_PATH]', s)
    # Scrub API keys and tokens (Gemini, Google, Sentry, OpenAI, Bearer tokens)
    s = re.sub(r'(?:AIzaSy|sk-[A-Za-z0-9_-]{20,}|gemma-[A-Za-z0-9_-]{10,}|Bearer\s+[A-Za-z0-9_.-]+)', '[REDACTED_KEY]', s)
    return s[:120]


def sanitize_telemetry_tags(tags: Dict[str, Any]) -> Dict[str, str]:
    """Sanitizes metadata to ensure NO source code, diffs, paths, or secrets leak into telemetry."""
    sanitized: Dict[str, str] = {}
    for k, v in tags.items():
        k_clean = str(k).lower().replace(" ", "_").strip()
        if k_clean in SAFE_TAG_KEYS:
            sanitized[k_clean] = sanitize_scalar(v)
    return sanitized


def sanitize_event(event: Dict[str, Any], hint: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Defense-in-depth Sentry before_send hook: scrubs error messages and breadcrumbs."""
    if not event:
        return event

    # Scrub exception values
    if "exception" in event and "values" in event["exception"]:
        for exc_val in event["exception"]["values"]:
            if "value" in exc_val and isinstance(exc_val["value"], str):
                exc_val["value"] = sanitize_scalar(exc_val["value"])

    # Scrub breadcrumbs
    if "breadcrumbs" in event and "values" in event["breadcrumbs"]:
        for b in event["breadcrumbs"]["values"]:
            if "message" in b and isinstance(b["message"], str):
                b["message"] = sanitize_scalar(b["message"])

    # Scrub request headers
    if "request" in event and "headers" in event["request"]:
        for h in ["authorization", "cookie", "x-api-key"]:
            if h in event["request"]["headers"]:
                event["request"]["headers"][h] = "[REDACTED]"

    return event


def sanitize_transaction(event: Dict[str, Any], hint: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Defense-in-depth Sentry before_send_transaction hook: scrubs transaction data and span descriptions."""
    if not event:
        return event

    if "transaction" in event and isinstance(event["transaction"], str):
        event["transaction"] = sanitize_scalar(event["transaction"])

    if "spans" in event and isinstance(event["spans"], list):
        for sp in event["spans"]:
            if "description" in sp and isinstance(sp["description"], str):
                sp["description"] = sanitize_scalar(sp["description"])
            if "data" in sp and isinstance(sp["data"], dict):
                cleaned_data = {}
                for k, v in sp["data"].items():
                    if str(k).lower() in SAFE_TAG_KEYS:
                        cleaned_data[k] = sanitize_scalar(v)
                sp["data"] = cleaned_data

    return event


def init_sentry(
    app=None,
    dsn: str = "",
    environment: str = "development",
    traces_sample_rate: float = 0.1,
    transport=None,
) -> bool:
    """Initializes Sentry SDK with Flask integration if a valid DSN is provided."""
    global _sentry_initialized

    if not dsn:
        logger.info("Sentry DSN not provided. Observability running in local offline tracking mode.")
        _sentry_initialized = False
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration

        # Avoid redundant re-initialization if already connected with same active client,
        # unless a custom test transport is provided
        if _sentry_initialized and transport is None:
            client = sentry_sdk.get_client()
            if client and client.is_active():
                return True

        integrations = [FlaskIntegration()]

        init_kwargs: Dict[str, Any] = {
            "dsn": dsn,
            "environment": environment,
            "traces_sample_rate": traces_sample_rate,
            "integrations": integrations,
            "attach_stacktrace": True,
            "before_send": sanitize_event,
            "before_send_transaction": sanitize_transaction,
        }
        if transport is not None:
            init_kwargs["transport"] = transport

        sentry_sdk.init(**init_kwargs)
        _sentry_initialized = True
        logger.info("Sentry Agent Tracing successfully initialized for environment: %s", environment)
        return True
    except Exception as e:
        logger.warning("Failed to initialize Sentry: %s", e)
        _sentry_initialized = False
        return False


def is_sentry_active() -> bool:
    """Returns whether Sentry SDK is actively initialized and connected."""
    return _sentry_initialized


def get_observability_status(dsn: str, environment: str, sample_rate: float) -> Dict[str, Any]:
    """Returns human-readable and structured observability status for UI."""
    if _sentry_initialized:
        return {
            "status": "connected",
            "message": f"Sentry Agent Tracing active ({environment})",
            "environment": environment,
            "sample_rate": sample_rate,
            "configured": True,
        }
    return {
        "status": "not_configured",
        "message": "Observability not configured (Sentry DSN omitted)",
        "environment": environment,
        "sample_rate": sample_rate,
        "configured": False,
    }


class RunTracer:
    """Helper wrapper around active root Sentry maintenance transaction."""

    def __init__(self, transaction=None):
        self.transaction = transaction
        self.outcome = "success"

    def set_tag(self, key: str, val: Any) -> None:
        if self.transaction:
            try:
                k_clean = str(key).lower().replace(" ", "_").strip()
                if k_clean in SAFE_TAG_KEYS:
                    self.transaction.set_tag(k_clean, sanitize_scalar(val))
            except Exception:
                pass

    def set_data(self, key: str, val: Any) -> None:
        if self.transaction:
            try:
                self.transaction.set_data(str(key)[:50], val)
            except Exception:
                pass

    def set_status(self, status: str) -> None:
        if self.transaction:
            try:
                self.transaction.set_status(status)
            except Exception:
                pass


class SpanContext:
    """Helper object yielded by agent_stage_span to record stage details, tags, and outcomes."""

    def __init__(self, span=None):
        self.span = span
        self.outcome = "success"
        self.stage_status = "completed"
        self.error_category: Optional[str] = None
        self.details: Optional[str] = None

    def set_outcome(self, outcome: str, details: Optional[str] = None) -> None:
        """Sets stage outcome ('completed', 'partial', 'failed', 'skipped')."""
        self.outcome = outcome
        if outcome in {"failed", "failure"}:
            self.stage_status = "failed"
            if self.span:
                try:
                    self.span.set_status("internal_error")
                except Exception:
                    pass
        elif outcome == "partial":
            self.stage_status = "partial"
        elif outcome == "skipped":
            self.stage_status = "skipped"
        elif outcome in {"completed", "success"}:
            self.stage_status = "completed"
            if self.span:
                try:
                    self.span.set_status("ok")
                except Exception:
                    pass

        if details:
            self.details = sanitize_scalar(details)

    def set_tag(self, key: str, val: Any) -> None:
        if self.span:
            try:
                k_clean = str(key).lower().replace(" ", "_").strip()
                if k_clean in SAFE_TAG_KEYS:
                    self.span.set_tag(k_clean, sanitize_scalar(val))
            except Exception:
                pass

    def set_data(self, key: str, val: Any) -> None:
        if self.span:
            try:
                self.span.set_data(str(key)[:50], val)
            except Exception:
                pass


@contextmanager
def maintenance_run_trace(
    run_id: str,
    repo_name: str,
    repo_type: str,
    provider: str,
    model: str,
    extra_tags: Optional[Dict[str, Any]] = None,
):
    """Creates a root Sentry transaction context around an entire maintenance run.

    Ensures the trace is created inside the background worker / execution thread
    where the work executes, rather than relying exclusively on Flask's incoming HTTP request transaction.
    """
    start_time = time.time()
    clean_repo_type = sanitize_scalar(repo_type or "local")
    clean_repo_name = sanitize_scalar(repo_name or "repository")
    tx_name = f"QuietPatch Maintenance Run: {clean_repo_type}"
    root_op = "agent.maintenance_run"

    sentry_tx = None
    if _sentry_initialized:
        try:
            import sentry_sdk

            sentry_tx = sentry_sdk.start_transaction(
                op=root_op,
                name=tx_name,
                custom_sampling_context={"run_id": run_id, "repo_type": clean_repo_type},
            )
            sentry_tx.__enter__()

            initial_tags = {
                "run_id": run_id,
                "repo_type": clean_repo_type,
                "repo_name": clean_repo_name,
                "provider": provider,
                "model": model,
                "app_mode": Config.APP_MODE,
            }
            if extra_tags:
                initial_tags.update(extra_tags)

            for k, v in sanitize_telemetry_tags(initial_tags).items():
                sentry_tx.set_tag(k, v)
        except Exception as e:
            logger.debug("Failed to start root Sentry maintenance transaction: %s", e)
            sentry_tx = None

    tracer = RunTracer(sentry_tx)
    outcome = "success"
    try:
        yield tracer
    except Exception as exc:
        outcome = "failure"
        if sentry_tx:
            try:
                sentry_tx.set_status("internal_error")
                sentry_tx.set_tag("outcome", "failure")
                sentry_tx.set_tag("error_category", type(exc).__name__)
            except Exception:
                pass
        raise
    finally:
        duration = round(time.time() - start_time, 3)
        if sentry_tx:
            try:
                if outcome == "success":
                    sentry_tx.set_status("ok")
                sentry_tx.set_tag("outcome", outcome)
                sentry_tx.set_data("duration_seconds", duration)
                sentry_tx.__exit__(None, None, None)
            except Exception as e:
                logger.debug("Failed to close root Sentry maintenance transaction: %s", e)

        logger.info(
            "Maintenance Run [%s] (%s) completed in %0.3fs with outcome=%s",
            run_id, clean_repo_type, duration, outcome
        )


@contextmanager
def agent_stage_span(
    stage_name: str,
    op: Optional[str] = None,
    run_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    extra_tags: Optional[Dict[str, Any]] = None,
    extra_data: Optional[Dict[str, Any]] = None,
):
    """Context manager for tracing an agent stage using a distinct child Sentry span.

    Supports required span ops:
      - quietpatch.initialize
      - quietpatch.repository_discovery
      - quietpatch.deterministic_scan
      - quietpatch.gemma_analysis
      - quietpatch.prioritization
      - quietpatch.patch_generation
      - quietpatch.verification
      - quietpatch.report_generation
    """
    start_time = time.time()

    # Determine effective op: if explicit op is passed, prioritize it;
    # otherwise map stage_name, or default to 'agent.stage'
    if op:
        effective_op = op
    elif stage_name.startswith("quietpatch."):
        effective_op = stage_name
        stage_name = stage_name.replace("quietpatch.", "").replace("_", " ").title()
    else:
        effective_op = STAGE_NAME_TO_OP.get(stage_name, "agent.stage")

    span_description = f"QuietPatch: {stage_name}"

    tags = {
        "stage": stage_name,
        "stage_name": stage_name,
        "op": effective_op,
        "run_id": run_id or "unassigned",
        "provider": provider or "unknown",
        "model": model or "unknown",
    }
    if extra_tags:
        tags.update(extra_tags)

    clean_tags = sanitize_telemetry_tags(tags)

    sentry_span = None
    if _sentry_initialized:
        try:
            import sentry_sdk

            sentry_span = sentry_sdk.start_span(
                op=effective_op,
                name=span_description,
            )
            for k, v in clean_tags.items():
                sentry_span.set_tag(k, v)
            if extra_data:
                for dk, dv in extra_data.items():
                    if str(dk).lower() in SAFE_TAG_KEYS:
                        sentry_span.set_data(dk, sanitize_scalar(dv))
            sentry_span.__enter__()
        except Exception as e:
            logger.debug("Failed to enter Sentry child span [%s]: %s", effective_op, e)
            sentry_span = None

    ctx = SpanContext(sentry_span)
    try:
        yield ctx
    except Exception as exc:
        ctx.set_outcome("failure")
        ctx.error_category = type(exc).__name__
        if sentry_span:
            try:
                sentry_span.set_status("internal_error")
                sentry_span.set_tag("outcome", "failure")
                sentry_span.set_tag("error_category", type(exc).__name__)
            except Exception:
                pass
        raise
    finally:
        duration = round(time.time() - start_time, 3)
        if sentry_span:
            try:
                if ctx.outcome == "success" and ctx.stage_status == "completed":
                    sentry_span.set_status("ok")
                elif ctx.stage_status == "failed" or ctx.outcome == "failure":
                    sentry_span.set_status("internal_error")

                sentry_span.set_tag("outcome", ctx.outcome)
                sentry_span.set_tag("stage_status", ctx.stage_status)
                sentry_span.set_data("duration_seconds", duration)
                if ctx.details:
                    sentry_span.set_data("details", ctx.details)
                sentry_span.__exit__(None, None, None)
            except Exception as e:
                logger.debug("Failed to exit Sentry child span [%s]: %s", effective_op, e)

        logger.info(
            "Agent Stage [%s] (%s) completed in %0.3fs with outcome=%s",
            stage_name, effective_op, duration, ctx.outcome
        )
