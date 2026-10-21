"""Base scanner interface and repository inspection utilities."""
import os
import re
from pathlib import Path
from typing import List, Optional, Set
from backend.models import Finding

# Standard ignored directories
DEFAULT_IGNORED_DIRS: Set[str] = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "out",
    ".next",
    ".nuxt",
    ".output",
    "target",
    ".quietpatch",
    "coverage",
    ".turbo",
    ".cache",
}

DEFAULT_IGNORED_EXTENSIONS: Set[str] = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib",
    ".exe", ".bin", ".tar", ".gz", ".zip", ".png",
    ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot", ".lock",
    ".map", ".min.js", ".min.css"
}

class BaseScanner:
    category: str = "general"

    def scan(self, repo_path: Path, files: List[Path]) -> List[Finding]:
        """Scans the repository and returns a list of findings."""
        raise NotImplementedError

    def read_text_safe(self, file_path: Path, max_bytes: int = 500_000) -> Optional[str]:
        """Safely reads text content with size limits and utf-8 fallback handling."""
        try:
            if not file_path.is_file():
                return None
            stat = file_path.stat()
            if stat.st_size > max_bytes:
                return None
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return None

    def get_relative_path(self, repo_path: Path, file_path: Path) -> str:
        """Returns relative path string from repository root."""
        try:
            return str(file_path.relative_to(repo_path))
        except ValueError:
            return str(file_path)
