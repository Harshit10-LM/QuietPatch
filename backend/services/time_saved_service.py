"""Time saved calculation and lazy payoff accounting."""
from typing import Dict, List
from backend.models import Finding, ProposedPatch, TimeSavedSummary

# Documented baseline estimates (minutes) per maintenance category
CATEGORY_EFFORT_MINUTES = {
    "documentation": 10,
    "configuration": 15,
    "general": 12,
    "python": 20,
    "javascript": 15,
    "todo": 20,
}

def calculate_time_saved(
    findings: List[Finding],
    patches: List[ProposedPatch],
) -> TimeSavedSummary:
    """Calculates estimated developer time saved with full transparency.

    Every finding has an explicit effort estimate, and applied/proposed patches
    represent immediate actionable time saved.
    """
    category_totals: Dict[str, int] = {}
    status_totals: Dict[str, int] = {
        "identified": 0,
        "patched": 0,
        "applied": 0,
    }

    # Time identified across all confirmed findings
    for f in findings:
        cat = f.category.value
        effort = f.estimated_effort_mins or CATEGORY_EFFORT_MINUTES.get(cat, 15)
        category_totals[cat] = category_totals.get(cat, 0) + effort
        status_totals["identified"] += effort

    # Time saved by proposed and applied fixes
    for p in patches:
        # Each proposed minimal fix saves an estimated 15 minutes of authoring and diff review
        status_totals["patched"] += 15
        if p.status.value == "applied":
            status_totals["applied"] += 20

    total_mins = status_totals["identified"]
    total_hours = round(total_mins / 60.0, 1)

    assumptions_note = (
        "Estimates are based on standard manual research, reproduction, and remediation baselines: "
        "Documentation link fixes (~10m), Environment configuration parity (~15m), "
        "Packaging & manifest hygiene (~15-20m), Maintenance marker resolution (~20m). "
        "All figures are documented estimates to illustrate lazy payoff."
    )

    return TimeSavedSummary(
        total_minutes=total_mins,
        total_hours=total_hours,
        breakdown_by_category=category_totals,
        breakdown_by_status=status_totals,
        assumptions_note=assumptions_note,
    )
