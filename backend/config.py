"""QuietPatch Application Configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    # Execution Mode: 'local' | 'hosted'
    APP_MODE = os.getenv("APP_MODE", "local").lower().strip()

    # AI Provider: 'ollama' | 'gemma_api' | 'demo'
    AI_PROVIDER = os.getenv("AI_PROVIDER", "demo" if APP_MODE == "hosted" and not os.getenv("GEMMA_API_KEY") else "ollama").lower().strip()

    # Local Ollama Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    try:
        OLLAMA_REQUEST_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "240.0"))
    except ValueError:
        OLLAMA_REQUEST_TIMEOUT_SECONDS = 240.0
    OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "15m").strip()
    try:
        OLLAMA_BATCH_SIZE = int(os.getenv("OLLAMA_BATCH_SIZE", "3"))
    except ValueError:
        OLLAMA_BATCH_SIZE = 3

    # Hosted Gemma Settings (Google Gen AI)
    GEMMA_API_KEY = os.getenv("GEMMA_API_KEY", "").strip()
    GEMMA_API_MODEL = os.getenv("GEMMA_API_MODEL", "gemma-4-26b-a4b-it")
    try:
        GEMMA_API_MAX_RETRIES = int(os.getenv("GEMMA_API_MAX_RETRIES", "2"))
    except ValueError:
        GEMMA_API_MAX_RETRIES = 2
    try:
        GEMMA_API_BACKOFF_FACTOR = float(os.getenv("GEMMA_API_BACKOFF_FACTOR", "0.5"))
    except ValueError:
        GEMMA_API_BACKOFF_FACTOR = 0.5
    try:
        GEMMA_API_MAX_OUTPUT_TOKENS = int(os.getenv("GEMMA_API_MAX_OUTPUT_TOKENS", "2048"))
    except ValueError:
        GEMMA_API_MAX_OUTPUT_TOKENS = 2048
    try:
        GEMMA_API_RECOVERY_MAX_TOKENS = int(os.getenv("GEMMA_API_RECOVERY_MAX_TOKENS", "3072"))
    except ValueError:
        GEMMA_API_RECOVERY_MAX_TOKENS = 3072

    # Security Boundaries
    default_allowed_root = str(BASE_DIR.resolve())
    QUIETPATCH_ALLOWED_ROOT = Path(os.getenv("QUIETPATCH_ALLOWED_ROOT", default_allowed_root)).resolve()

    # Sentry Configuration
    SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
    try:
        SENTRY_TRACES_SAMPLE_RATE = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
    except ValueError:
        SENTRY_TRACES_SAMPLE_RATE = 0.1
    SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "development")

    # Resource Limits
    MAX_REPO_FILES = int(os.getenv("MAX_REPO_FILES", "500"))
    MAX_REPO_BYTES = int(os.getenv("MAX_REPO_BYTES", str(10 * 1024 * 1024)))  # 10 MB
    MAX_SCAN_SECONDS = int(os.getenv("MAX_SCAN_SECONDS", "60"))

    # Bundled sample repo
    SAMPLE_REPO_PATH = (BASE_DIR / "sample-repo").resolve()

    # Frontend build directory
    FRONTEND_DIST = (BASE_DIR / "frontend" / "dist").resolve()

    @classmethod
    def get_sanitized_public_config(cls):
        """Returns safe configuration metadata for UI consumption (NO secrets)."""
        return {
            "app_mode": cls.APP_MODE,
            "ai_provider": cls.AI_PROVIDER,
            "ollama_base_url": cls.OLLAMA_BASE_URL,
            "ollama_model": cls.OLLAMA_MODEL,
            "ollama_request_timeout_seconds": cls.OLLAMA_REQUEST_TIMEOUT_SECONDS,
            "ollama_keep_alive": cls.OLLAMA_KEEP_ALIVE,
            "ollama_batch_size": cls.OLLAMA_BATCH_SIZE,
            "gemma_api_model": cls.GEMMA_API_MODEL,
            "has_gemma_api_key": bool(cls.GEMMA_API_KEY),
            "gemma_api_max_retries": cls.GEMMA_API_MAX_RETRIES,
            "gemma_api_backoff_factor": cls.GEMMA_API_BACKOFF_FACTOR,
            "gemma_api_max_output_tokens": cls.GEMMA_API_MAX_OUTPUT_TOKENS,
            "gemma_api_recovery_max_tokens": cls.GEMMA_API_RECOVERY_MAX_TOKENS,
            "allowed_root": str(cls.QUIETPATCH_ALLOWED_ROOT),
            "sentry_configured": bool(cls.SENTRY_DSN),
            "sentry_environment": cls.SENTRY_ENVIRONMENT,
            "sentry_sample_rate": cls.SENTRY_TRACES_SAMPLE_RATE,
            "max_repo_files": cls.MAX_REPO_FILES,
            "max_repo_bytes": cls.MAX_REPO_BYTES,
            "max_scan_seconds": cls.MAX_SCAN_SECONDS,
            "sample_repo_available": cls.SAMPLE_REPO_PATH.exists(),
        }
