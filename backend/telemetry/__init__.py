"""Telemetry package for QuietPatch."""
from .tracing import (
    STAGE_NAME_TO_OP,
    RunTracer,
    SpanContext,
    agent_stage_span,
    get_observability_status,
    init_sentry,
    is_sentry_active,
    maintenance_run_trace,
    sanitize_scalar,
    sanitize_telemetry_tags,
)

__all__ = [
    "init_sentry",
    "agent_stage_span",
    "maintenance_run_trace",
    "is_sentry_active",
    "get_observability_status",
    "RunTracer",
    "SpanContext",
    "STAGE_NAME_TO_OP",
    "sanitize_scalar",
    "sanitize_telemetry_tags",
]
