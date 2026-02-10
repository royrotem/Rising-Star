#!/usr/bin/env bash
set -euo pipefail

# UAIE - Universal Autonomous Insight Engine
# Cross-platform install script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "   UAIE - Full Installation"
echo "========================================"
echo ""

cd "$PROJECT_DIR"

# Check Python
echo -e "[1/6] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed!${NC}"
    echo "Install Python 3.11+:"
    echo "  macOS:  brew install python@3.11"
    echo "  Ubuntu: sudo apt install python3.11 python3.11-venv"
    exit 1
fi
python3 --version
echo -e "${GREEN}Python OK!${NC}"
echo ""

# Check Node.js
echo "[2/6] Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo -e "${RED}ERROR: Node.js is not installed!${NC}"
    echo "Install Node.js 18+:"
    echo "  macOS:  brew install node"
    echo "  Ubuntu: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install -y nodejs"
    exit 1
fi
node --version
echo -e "${GREEN}Node.js OK!${NC}"
echo ""

# Check npm
echo "[3/6] Checking npm..."
if ! command -v npm &> /dev/null; then
    echo -e "${RED}ERROR: npm is not installed!${NC}"
    exit 1
fi
npm --version
echo -e "${GREEN}npm OK!${NC}"
echo ""

# Setup Backend
echo "[4/6] Setting up Backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to install Python dependencies${NC}"
    exit 1
fi

cd "$PROJECT_DIR"
echo -e "${GREEN}Backend setup complete!${NC}"
echo ""

# Setup Frontend
echo "[5/6] Setting up Frontend..."
cd frontend

echo "Installing Node.js dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to install Node.js dependencies${NC}"
    exit 1
fi

cd "$PROJECT_DIR"
echo -e "${GREEN}Frontend setup complete!${NC}"
echo ""

# Create .env file
echo "[6/6] Creating environment file..."
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "Created .env file from template"
fi
echo ""

echo "========================================"
echo -e "   ${GREEN}Installation Complete!${NC}"
echo "========================================"
echo ""
echo "You can now run the application using:"
echo "  ./scripts/run.sh        (local development)"
echo "  ./scripts/run-docker.sh (Docker - recommended)"
echo ""
