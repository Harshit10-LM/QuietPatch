"""Abstract base class for QuietPatch AI providers."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.models import Finding, ProposedPatch

class BaseAIProvider(ABC):
    name: str = "base"

    @abstractmethod
    def check_availability(self) -> Dict[str, Any]:
        """Checks whether the provider is reachable and ready."""
        pass

    @abstractmethod
    def analyze_and_prioritize(
        self,
        findings: List[Finding],
        run_id: Optional[str] = None
    ) -> List[Finding]:
        """Uses Gemma to review, explain, and prioritize confirmed deterministic findings."""
        pass

    @abstractmethod
    def propose_patch(
        self,
        finding: Finding,
        file_content: str,
        repo_path: Path,
        run_id: Optional[str] = None
    ) -> Optional[ProposedPatch]:
        """Proposes a minimal, targeted patch for a confirmed finding."""
        pass
