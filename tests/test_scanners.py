"""Unit tests for deterministic scanners."""
import tempfile
from pathlib import Path
import pytest
from backend.models import FindingCategory, Severity
from backend.scanners.config_scanner import ConfigScanner
from backend.scanners.doc_scanner import DocScanner
from backend.scanners.engine import ScanEngine
from backend.scanners.general_scanner import GeneralScanner
from backend.scanners.javascript_scanner import JavaScriptScanner
from backend.scanners.python_scanner import PythonScanner
from backend.scanners.todo_scanner import TodoScanner

def test_todo_scanner_finds_markers(tmp_path: Path):
    py_file = tmp_path / "app.py"
    py_file.write_text(
        "# Normal comment\n"
        "# TODO: Refactor auth handler\n"
        "# FIXME: Database connection leak on timeout\n"
        "# XXX: Temporary hack for batching\n"
        "# TODO(security): Redact authorization tokens\n"
    )

    scanner = TodoScanner()
    findings = scanner.scan(tmp_path, [py_file])

    assert len(findings) == 4
    categories = [f.category for f in findings]
    assert all(c == FindingCategory.TODO for c in categories)

    severities = {f.title.split(":")[0]: f.severity for f in findings}
    assert severities["FIXME"] == Severity.HIGH
    assert severities["TODO(security)"] == Severity.HIGH
    assert severities["XXX"] == Severity.MEDIUM
    assert severities["TODO"] == Severity.LOW

def test_todo_scanner_deduplication(tmp_path: Path):
    py_file = tmp_path / "app.py"
    py_file.write_text(
        "# TODO: Duplicate comment\n"
        "# TODO: Duplicate comment\n"
    )

    scanner = TodoScanner()
    findings = scanner.scan(tmp_path, [py_file])
    assert len(findings) == 1

def test_doc_scanner_broken_relative_links(tmp_path: Path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    readme = tmp_path / "README.md"
    existing_file = doc_dir / "valid.md"
    existing_file.write_text("# Valid")

    readme.write_text(
        "# My Project\n\n"
        "- [Valid Link](./docs/valid.md)\n"
        "- [Broken Link](./docs/missing.md)\n"
        "- [External Link](https://github.com/quietpatch)\n"
        "- [Anchor](#my-project)\n"
    )

    scanner = DocScanner()
    findings = scanner.scan(tmp_path, [readme, existing_file])

    broken_findings = [f for f in findings if "Broken relative link" in f.title]
    assert len(broken_findings) == 1
    assert "missing.md" in broken_findings[0].title
    assert broken_findings[0].category == FindingCategory.DOCUMENTATION

def test_config_scanner_detects_undocumented_vars(tmp_path: Path):
    env_example = tmp_path / ".env.example"
    env_example.write_text("PORT=3000\nDATABASE_URL=postgres://localhost\n")

    code_file = tmp_path / "server.py"
    code_file.write_text(
        "import os\n"
        "port = os.getenv('PORT')\n"
        "db = os.environ.get('DATABASE_URL')\n"
        "secret = os.environ['SUPER_SECRET_TOKEN']\n"
    )

    scanner = ConfigScanner()
    findings = scanner.scan(tmp_path, [env_example, code_file])

    missing_env = [f for f in findings if "SUPER_SECRET_TOKEN" in f.title]
    assert len(missing_env) == 1
    assert missing_env[0].severity == Severity.HIGH
    assert missing_env[0].category == FindingCategory.CONFIGURATION
    assert "SUPER_SECRET_TOKEN" in missing_env[0].title

def test_python_scanner_unpinned_deps(tmp_path: Path):
    py_file = tmp_path / "main.py"
    py_file.write_text("print('hello')")
    req = tmp_path / "requirements.txt"
    req.write_text("flask>=3.0.0\nrequests\npydantic\n")

    scanner = PythonScanner()
    findings = scanner.scan(tmp_path, [py_file, req])

    unpinned = [f for f in findings if "Unpinned" in f.title]
    assert len(unpinned) == 1
    assert "requests" in unpinned[0].evidence

def test_javascript_scanner_detects_missing_test_script(tmp_path: Path):
    pkg = tmp_path / "package.json"
    pkg.write_text('{"name": "test-pkg", "scripts": {"build": "tsc"}}')

    scanner = JavaScriptScanner()
    findings = scanner.scan(tmp_path, [pkg])

    no_test = [f for f in findings if "No 'test' script" in f.title]
    assert len(no_test) == 1

def test_general_scanner_missing_license_and_gitignore(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text("# Readme")

    scanner = GeneralScanner()
    findings = scanner.scan(tmp_path, [readme])

    titles = [f.title for f in findings]
    assert any("Missing LICENSE" in t for t in titles)
    assert any("Missing .gitignore" in t for t in titles)

def test_scan_engine_respects_limits_and_ignores(tmp_path: Path):
    # Create git dir, node_modules, and standard files
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("dummy git")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("dummy pkg")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "index.py").write_text("# TODO: Real work\n")

    engine = ScanEngine(max_files=10)
    findings, count, durations = engine.scan_repository(tmp_path)

    # .git and node_modules must be excluded from discovery
    assert count == 1
    assert any("TODO" in f.title for f in findings)
