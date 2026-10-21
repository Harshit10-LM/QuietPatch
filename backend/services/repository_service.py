"""Repository management service with strict path containment and SSRF defenses."""
import io
import ipaddress
import os
import re
import socket
import tempfile
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

from backend.config import Config

class SecurityError(Exception):
    """Raised when an operation violates security boundaries."""
    pass

class RepositoryService:
    def __init__(self, allowed_root: Path = Config.QUIETPATCH_ALLOWED_ROOT):
        self.allowed_root = allowed_root.resolve()
        self.sample_repo_path = Config.SAMPLE_REPO_PATH.resolve()

    def validate_local_path(self, raw_path: str) -> Path:
        """Validates that a local repository path is strictly contained within allowed root or sample repo."""
        if not raw_path:
            raise SecurityError("Repository path cannot be empty.")

        # Check if requesting sample repo directly
        if raw_path in {"sample-repo", "sample", "demo"}:
            if self.sample_repo_path.exists():
                return self.sample_repo_path
            raise SecurityError(f"Sample repository not found at {self.sample_repo_path}")

        try:
            target_path = Path(raw_path).expanduser().resolve()
        except Exception as e:
            raise SecurityError(f"Invalid path syntax: {e}")

        # Allow exact sample repo path
        if target_path == self.sample_repo_path or self.sample_repo_path in target_path.parents:
            if not target_path.exists() or not target_path.is_dir():
                raise SecurityError(f"Directory does not exist: {target_path}")
            return target_path

        # Check containment within allowed_root
        try:
            target_path.relative_to(self.allowed_root)
        except ValueError:
            raise SecurityError(
                f"Path '{target_path}' is outside the permitted repository root: '{self.allowed_root}'. "
                "Update QUIETPATCH_ALLOWED_ROOT to inspect this directory."
            )

        if not target_path.exists():
            raise SecurityError(f"Directory does not exist: {target_path}")
        if not target_path.is_dir():
            raise SecurityError(f"Path is not a directory: {target_path}")

        return target_path

    def validate_github_url(self, url: str) -> Tuple[str, str]:
        """Validates a public GitHub repository URL against SSRF and unauthorized schemes.

        Returns (owner, repo).
        """
        if not url:
            raise SecurityError("GitHub URL cannot be empty.")

        parsed = urllib.parse.urlparse(url.strip())
        if parsed.scheme not in {"http", "https"}:
            raise SecurityError("Only HTTPS/HTTP repository URLs are supported.")

        hostname = (parsed.hostname or "").lower()
        if hostname not in {"github.com", "www.github.com"}:
            raise SecurityError("Only public repositories from github.com are permitted.")

        # Check for IP address literals or localhost
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                raise SecurityError("Private and loopback addresses are prohibited.")
        except ValueError:
            pass  # It's a hostname string, not an IP literal

        # Extract /owner/repo
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(path_parts) < 2:
            raise SecurityError("URL must contain repository owner and name (e.g. https://github.com/owner/repo).")

        owner = path_parts[0]
        repo = path_parts[1].removesuffix(".git")

        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", owner) or not re.match(r"^[a-zA-Z0-9_\-\.]+$", repo):
            raise SecurityError("Invalid characters in repository owner or name.")

        return owner, repo

    def download_public_github_repo(
        self,
        url: str,
        max_bytes: int = Config.MAX_REPO_BYTES,
        max_files: int = Config.MAX_REPO_FILES,
    ) -> Path:
        """Safely downloads a public GitHub repository archive into a temporary sandbox.

        Includes strict zip-slip and file limit defenses.
        """
        owner, repo = self.validate_github_url(url)
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"

        try:
            res = requests.get(zip_url, timeout=15.0, stream=True)
            if res.status_code == 404:
                # Try 'master' branch fallback
                zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"
                res = requests.get(zip_url, timeout=15.0, stream=True)
            res.raise_for_status()
        except Exception as e:
            raise SecurityError(f"Failed to fetch repository archive: {e}")

        # Stream into memory with size limit
        buffer = bytearray()
        for chunk in res.iter_content(chunk_size=65536):
            buffer.extend(chunk)
            if len(buffer) > max_bytes:
                raise SecurityError(f"Repository archive exceeds maximum allowed size ({max_bytes} bytes).")

        # Create temporary extraction directory
        temp_dir = Path(tempfile.mkdtemp(prefix=f"quietpatch_gh_{owner}_{repo}_"))
        
        try:
            with zipfile.ZipFile(io.BytesIO(buffer)) as zf:
                file_count = 0
                for member in zf.infolist():
                    # Zip slip defense: ensure target does not escape temp_dir
                    target_file = (temp_dir / member.filename).resolve()
                    if not str(target_file).startswith(str(temp_dir.resolve())):
                        raise SecurityError("Malicious archive path traversal detected in repository zip.")

                    # Skip symlinks
                    if (member.external_attr >> 16) & 0o120000 == 0o120000:
                        continue

                    if not member.is_dir():
                        file_count += 1
                        if file_count > max_files:
                            raise SecurityError(f"Archive exceeds maximum file count limit ({max_files} files).")

                # Extract safely
                zf.extractall(temp_dir)

            # GitHub zips have top-level dir: repo-main/
            extracted_subdirs = [d for d in temp_dir.iterdir() if d.is_dir()]
            if extracted_subdirs:
                return extracted_subdirs[0]
            return temp_dir

        except Exception as e:
            raise SecurityError(f"Failed to extract repository archive safely: {e}")

    def list_recent_repositories(self) -> List[Dict[str, Any]]:
        """Returns sample repository and local root options for quick selection in UI."""
        repos = []
        if self.sample_repo_path.exists():
            repos.append({
                "name": "NovaTask (Sample Project)",
                "path": str(self.sample_repo_path),
                "type": "sample",
                "description": "Bundled sample repository with realistic reproducible maintenance issues.",
                "languages": ["Python", "TypeScript", "Markdown"],
            })
        
        repos.append({
            "name": f"Current Workspace ({self.allowed_root.name})",
            "path": str(self.allowed_root),
            "type": "local",
            "description": f"Configured local repository root at {self.allowed_root}",
            "languages": ["Auto-detected"],
        })

        return repos
