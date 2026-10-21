"""Data models and Pydantic schemas for QuietPatch."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class FindingCategory(str, Enum):
    TODO = "todo"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    GENERAL = "general"

class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PatchStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    UNDONE = "undone"

class Finding(BaseModel):
    id: str
    category: FindingCategory
    severity: Severity = Severity.MEDIUM
    title: str
    file_path: str
    line_number: Optional[int] = None
    evidence: str
    explanation: str
    suggested_action: str
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    ai_assisted: bool = False
    is_demo: bool = False
    ai_model: Optional[str] = None
    estimated_effort_mins: int = 15
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ProposedPatch(BaseModel):
    id: str
    finding_id: str
    file_path: str
    change_summary: str
    original_content: str
    proposed_content: str
    diff: str
    explanation: str
    risks_assumptions: str
    suggested_verification: str
    status: PatchStatus = PatchStatus.PENDING
    backup_path: Optional[str] = None
    is_demo: bool = False
    ai_model: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    applied_at: Optional[str] = None

class StageRecord(BaseModel):
    name: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    duration_seconds: float = 0.0
    details: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class MaintenanceRun(BaseModel):
    id: str
    repository_name: str
    repository_path: str
    repository_type: str = "local"  # 'local' | 'sample' | 'github'
    status: str = "discovering"  # discovering, scanning, analyzing, prioritizing, proposing, completed, failed
    stages: Dict[str, StageRecord] = Field(default_factory=dict)
    files_scanned: int = 0
    findings: List[Finding] = Field(default_factory=list)
    patches: List[ProposedPatch] = Field(default_factory=list)
    ai_provider: str = "demo"
    ai_model: str = "none"
    duration_seconds: float = 0.0
    time_saved_minutes: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None

class TimeSavedSummary(BaseModel):
    total_minutes: int
    total_hours: float
    breakdown_by_category: Dict[str, int]
    breakdown_by_status: Dict[str, int]
    assumptions_note: str

class MaintenanceReport(BaseModel):
    run_id: str
    repository_name: str
    repository_path: str
    repository_type: str
    generated_at: str
    ai_provider: str
    ai_model: str
    total_duration_seconds: float
    files_scanned: int
    total_findings: int
    findings_by_severity: Dict[str, int]
    findings_by_category: Dict[str, int]
    fixes_proposed: int
    fixes_applied: int
    checks_passed: int
    checks_failed: int
    checks_skipped: int
    time_saved: TimeSavedSummary
    remaining_work: List[Dict[str, Any]]
    stage_durations: Dict[str, float]
