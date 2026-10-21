# =============================================================================
# Stage 1: Build React Frontend
# =============================================================================
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# =============================================================================
# Stage 2: Python Backend Runtime
# =============================================================================
FROM python:3.12-slim AS runner
WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python backend dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend codebase
COPY backend/ ./backend/
COPY sample-repo/ ./sample-repo/
COPY pytest.ini ./

# Copy built frontend assets from builder stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Default environment configuration for production container
ENV PYTHONUNBUFFERED=1 \
    APP_MODE=hosted \
    AI_PROVIDER=demo \
    PORT=5001

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5001}/api/health || exit 1

# Production WSGI startup (1 worker + 8 threads preserves shared in-process state on single-instance Render)
CMD exec gunicorn --bind 0.0.0.0:${PORT:-5001} --workers 1 --threads 8 --timeout 120 "backend.app:create_app()"
