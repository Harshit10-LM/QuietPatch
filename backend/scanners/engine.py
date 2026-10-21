"""Scanning engine orchestrator for QuietPatch."""
import fnmatch
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from backend.config import Config
from backend.models import Finding
from backend.scanners.base import DEFAULT_IGNORED_DIRS, DEFAULT_IGNORED_EXTENSIONS
from backend.scanners.config_scanner import ConfigScanner
from backend.scanners.doc_scanner import DocScanner
from backend.scanners.general_scanner import GeneralScanner
from backend.scanners.javascript_scanner import JavaScriptScanner
from backend.scanners.python_scanner import PythonScanner
from backend.scanners.todo_scanner import TodoScanner
from backend.telemetry.tracing import agent_stage_span

class ScanEngine:
    def __init__(
        self,
        max_files: int = Config.MAX_REPO_FILES,
        max_bytes: int = Config.MAX_REPO_BYTES,
        max_seconds: int = Config.MAX_SCAN_SECONDS,
    ):
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.max_seconds = max_seconds
        
        self.scanners = [
            GeneralScanner(),
            DocScanner(),
            ConfigScanner(),
            TodoScanner(),
            PythonScanner(),
            JavaScriptScanner(),
        ]

    def load_ignore_patterns(self, repo_path: Path) -> List[str]:
        """Loads custom patterns from .gitignore and .quietpatchignore if present."""
        patterns: List[str] = []
        for ignore_filename in [".gitignore", ".quietpatchignore"]:
            ignore_file = repo_path / ignore_filename
            if ignore_file.is_file():
                try:
                    with open(ignore_file, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            clean = line.strip()
                            if clean and not clean.startswith("#"):
                                patterns.append(clean)
                except Exception:
                    pass
        return patterns

    def should_ignore(self, path: Path, repo_path: Path, ignore_patterns: List[str]) -> bool:
        """Determines if a file or directory should be ignored."""
        try:
            rel = path.relative_to(repo_path)
            parts = rel.parts
        except ValueError:
            return True

        # Check against default ignored directories in any part of path
        for part in parts:
            if part in DEFAULT_IGNORED_DIRS:
                return True
            if part.startswith(".") and part not in {".env.example", ".env.sample", ".env.template", ".github"}:
                # Skip hidden directories like .idea, .vscode, etc.
                if path.is_dir() or len(parts) > 1:
                    return True

        if path.is_file():
            if path.suffix.lower() in DEFAULT_IGNORED_EXTENSIONS:
                return True

        # Match against gitignore / quietpatchignore patterns
        rel_str = str(rel)
        for pattern in ignore_patterns:
            clean_pat = pattern.rstrip("/")
            if fnmatch.fnmatch(rel_str, clean_pat) or fnmatch.fnmatch(path.name, clean_pat):
                return True
            if any(fnmatch.fnmatch(p, clean_pat) for p in parts):
                return True

        return False

    def discover_files(self, repo_path: Path) -> Tuple[List[Path], int]:
        """Discovers eligible files up to resource limits."""
        ignore_patterns = self.load_ignore_patterns(repo_path)
        discovered: List[Path] = []
        total_bytes = 0

        for root, dirs, files in os.walk(repo_path, followlinks=False):
            # Prune ignored directories in-place
            dirs[:] = [
                d for d in dirs
                if not self.should_ignore(Path(root) / d, repo_path, ignore_patterns)
            ]

            for file_name in files:
                file_path = Path(root) / file_name
                if self.should_ignore(file_path, repo_path, ignore_patterns):
                    continue

                try:
                    stat = file_path.stat()
                    # Skip symlinks that escape repo or circular
                    if file_path.is_symlink():
                        continue
                    
                    file_size = stat.st_size
                    if total_bytes + file_size > self.max_bytes:
                        continue
                    
                    total_bytes += file_size
                    discovered.append(file_path)

                    if len(discovered) >= self.max_files:
                        return discovered, total_bytes
                except Exception:
                    continue

        return discovered, total_bytes

    def run_scanners(
        self,
        repo_path: Path,
        files: List[Path],
        run_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Tuple[List[Finding], Dict[str, float]]:
        """Runs the deterministic scanner suite across discovered files with quietpatch.deterministic_scan span."""
        start_time = time.time()
        stage_durations: Dict[str, float] = {}
        all_findings: List[Finding] = []

        with agent_stage_span(
            "Deterministic Scanner Suite",
            op="quietpatch.deterministic_scan",
            run_id=run_id,
            provider=provider,
            model=model,
            extra_tags={"files_count": len(files)},
        ) as scan_span_ctx:
            for scanner in self.scanners:
                if time.time() - start_time > self.max_seconds:
                    break

                scanner_name = scanner.__class__.__name__
                t_scan = time.time()
                try:
                    findings = scanner.scan(repo_path, files)
                    all_findings.extend(findings)
                except Exception:
                    pass
                stage_durations[scanner.category] = round(time.time() - t_scan, 3)

            scan_span_ctx.set_tag("findings_count", len(all_findings))
            scan_span_ctx.set_tag("files_count", len(files))
            scan_span_ctx.set_data("files_scanned", len(files))
            scan_span_ctx.set_data("findings_detected", len(all_findings))

        return all_findings, stage_durations

    def scan_repository(
        self,
        repo_path: Path,
        run_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Tuple[List[Finding], int, Dict[str, float]]:
        """Executes discovery and all registered scanners with stage tracing and timeout safety."""
        stage_durations: Dict[str, float] = {}

        # 1. Discover stage (quietpatch.repository_discovery)
        with agent_stage_span(
            "Repository Discovery",
            op="quietpatch.repository_discovery",
            run_id=run_id,
            provider=provider,
            model=model,
        ) as disc_span_ctx:
            t0 = time.time()
            files, total_bytes = self.discover_files(repo_path)
            stage_durations["discovery"] = round(time.time() - t0, 3)
            disc_span_ctx.set_tag("files_count", len(files))
            disc_span_ctx.set_data("total_bytes", total_bytes)

        # 2. Scanner stages (quietpatch.deterministic_scan)
        all_findings, scan_durations = self.run_scanners(
            repo_path=repo_path,
            files=files,
            run_id=run_id,
            provider=provider,
            model=model,
        )
        stage_durations.update(scan_durations)

        return all_findings, len(files), stage_durations
