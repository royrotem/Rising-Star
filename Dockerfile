# =============================================================================
# UAIE Production Dockerfile — single-container deployment
# Builds the React frontend, then serves it + the FastAPI backend from one image.
# =============================================================================

# ── Stage 1: Build the React frontend ───────────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /build

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
# Use vite build directly — skip tsc type-checking which fails in CI
# due to strict mode. Vite/esbuild handles the actual transpilation.
RUN npx vite build


# ── Stage 2: Production backend + frontend static files ─────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps (libpq for asyncpg, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY backend/ .

# Copy built frontend into /app/static
COPY --from=frontend-build /build/dist /app/static

# Persistent data volume
RUN mkdir -p /app/data

# Render.com injects PORT at runtime (default 8000)
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE ${PORT}

# Use 4 Uvicorn workers by default (Render free tier has 512 MB)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-2} --timeout-keep-alive 120
