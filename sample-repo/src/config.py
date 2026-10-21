"""NovaTask Runtime Configuration."""
import os

PORT = int(os.getenv("PORT", "8000"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/novatask")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# Undocumented environment variables missing from .env.example
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))

# TODO(security): Validate API_SECRET_KEY entropy and enforce non-empty value in production
# FIXME: Implement automatic database connection reconnection pool timeout
def get_config():
    return {
        "port": PORT,
        "database_url": DATABASE_URL,
        "metrics_port": METRICS_PORT,
    }
