#!/usr/bin/env bash
set -euo pipefail

# UAIE - Pull latest changes and update dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GREEN='\033[0;32m'
NC='\033[0m'

echo "========================================"
echo "   UAIE - Update"
echo "========================================"
echo ""

cd "$PROJECT_DIR"

echo "Pulling latest changes..."
git pull

echo ""
echo "Updating backend dependencies..."
cd backend
if [ -d "venv" ]; then
    source venv/bin/activate
    pip install -r requirements.txt --quiet
fi
cd "$PROJECT_DIR"

echo "Updating frontend dependencies..."
cd frontend
npm install --silent
cd "$PROJECT_DIR"

echo ""
echo "========================================"
echo -e "   ${GREEN}Update Complete!${NC}"
echo "========================================"
echo ""
