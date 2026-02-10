#!/usr/bin/env bash
set -euo pipefail

# UAIE - Run with Docker Compose

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================"
echo "   UAIE - Docker Deployment"
echo "========================================"
echo ""

cd "$PROJECT_DIR"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed!${NC}"
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    echo -e "${RED}ERROR: Docker Compose is not available!${NC}"
    echo "Install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "Building and starting containers..."
docker compose up --build -d

echo ""
echo "========================================"
echo -e "   ${GREEN}UAIE is running!${NC}"
echo "========================================"
echo ""
echo "  Frontend:  http://localhost:3001"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Postgres:  localhost:5432"
echo "  Redis:     localhost:6379"
echo ""
echo "  Logs:       docker compose logs -f"
echo "  Stop:       ./scripts/stop-docker.sh"
echo ""
