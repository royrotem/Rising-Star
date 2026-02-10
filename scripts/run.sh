#!/usr/bin/env bash
set -euo pipefail

# UAIE - Run locally (backend + frontend)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "   UAIE - Local Development"
echo "========================================"
echo ""

cd "$PROJECT_DIR"

# Load .env if present
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo -e "${GREEN}Loaded .env${NC}"
fi

# Ensure data directory exists
mkdir -p backend/data

# Start Backend
echo "Starting Backend on http://localhost:8000 ..."
cd backend

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Run install.sh first.${NC}"
    exit 1
fi

source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

cd "$PROJECT_DIR"

# Start Frontend
echo "Starting Frontend on http://localhost:5173 ..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

cd "$PROJECT_DIR"

echo ""
echo "========================================"
echo -e "   ${GREEN}UAIE is running!${NC}"
echo "========================================"
echo ""
echo "  Frontend:  http://localhost:5173"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop all services"
echo ""

# Trap Ctrl+C to stop both processes
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID 2>/dev/null || true
    wait $FRONTEND_PID 2>/dev/null || true
    echo "All services stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for either process to exit
wait
