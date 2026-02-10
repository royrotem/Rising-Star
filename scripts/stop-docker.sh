#!/usr/bin/env bash
set -euo pipefail

# UAIE - Stop Docker containers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "   UAIE - Stopping Docker"
echo "========================================"
echo ""

cd "$PROJECT_DIR"
docker compose down

echo ""
echo "All containers stopped."
echo ""
