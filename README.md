# Rising-Star

## Product Vision: The Universal Autonomous Insight Engine (UAIE)

### The "Tesla Standard" – Democratized as a Service (SaaS)

---

## Table of Contents

- [Product Vision](#the-challenge)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
  - [Windows](#windows-installation)
  - [macOS](#macos-installation)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Development](#development)

---

## The Challenge

Complex hardware companies (aerospace, robotics, MedTech, automotive) struggle with a massive disconnect between their physical products and the petabytes of data they generate. Currently, achieving the level of data maturity found in companies like Tesla requires years of internal development and a dedicated team of 50+ data scientists.

## The Solution

The UAIE is a domain-agnostic SaaS platform that deploys a workforce of AI Agents to ingest raw, unstructured system data and transform it into actionable engineering intelligence. It bridges the gap between hardware physics and software logic, allowing any engineering team to achieve "AI Readiness" from Day One.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/royrotem/Rising-Star.git
cd Rising-Star

# Switch to the development branch (contains all the code)
git fetch origin
git checkout claude/refine-saas-description-3Bzwc

# Run with Docker (recommended)
docker-compose up

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

> **Note:** The main branch contains only the README. All source code is on the `claude/refine-saas-description-3Bzwc` branch.

---

## Windows Quick Start (One-Click Scripts)

For Windows users, we provide easy-to-use batch scripts in the `scripts/` folder:

| Script | Description |
|--------|-------------|
| `START.bat` | Main menu - choose what to do |
| `install.bat` | Install all dependencies (Python + Node.js packages) |
| `run.bat` | Run the application locally (opens Backend + Frontend) |
| `run-docker.bat` | Run with Docker (recommended) |
| `update.bat` | Pull latest updates from Git |
| `stop-docker.bat` | Stop Docker containers |

### Usage

1. Clone and switch to branch (first time only):
   ```cmd
   git clone https://github.com/royrotem/Rising-Star.git
   cd Rising-Star
   git fetch origin
   git checkout claude/refine-saas-description-3Bzwc
   ```

2. Double-click `scripts\START.bat` and choose an option!

---

## Prerequisites

### Option 1: Docker (Recommended)

| Software | Version | Download Link |
|----------|---------|---------------|
| Docker Desktop | 4.0+ | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) |
| Git | 2.30+ | [git-scm.com](https://git-scm.com/) |

### Option 2: Local Development

| Software | Version | Download Link |
|----------|---------|---------------|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 20+ | [nodejs.org](https://nodejs.org/) |
| PostgreSQL | 15+ | [postgresql.org](https://www.postgresql.org/download/) |
| Redis | 7+ | [redis.io](https://redis.io/download/) |
| Git | 2.30+ | [git-scm.com](https://git-scm.com/) |

---

## Installation Guide

### Windows Installation

#### Step 1: Install Git

1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Run the installer with default settings
3. Open **Command Prompt** or **PowerShell** and verify:
   ```cmd
   git --version
   ```

#### Step 2: Clone the Repository

```cmd
# Open Command Prompt or PowerShell
cd %USERPROFILE%\Documents
git clone https://github.com/royrotem/Rising-Star.git
cd Rising-Star

# Switch to the development branch
git fetch origin
git checkout claude/refine-saas-description-3Bzwc

# Verify files are present
dir
```

> You should see: `backend/`, `frontend/`, `docker-compose.yml`, etc.

#### Step 3A: Run with Docker (Recommended)

1. Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Start Docker Desktop and wait for it to initialize
3. Run the application:
   ```cmd
   docker-compose up
   ```

#### Step 3B: Run Locally (Without Docker)

**Install Python:**
1. Download Python 3.11+ from [python.org](https://www.python.org/downloads/)
2. During installation, check **"Add Python to PATH"**
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

**Install Node.js:**
1. Download Node.js 20+ from [nodejs.org](https://nodejs.org/)
2. Run the installer
3. Verify installation:
   ```cmd
   node --version
   npm --version
   ```

**Install PostgreSQL:**
1. Download from [postgresql.org/download/windows](https://www.postgresql.org/download/windows/)
2. Run the installer, set password to `postgres`
3. Create the database:
   ```cmd
   psql -U postgres
   CREATE DATABASE uaie;
   \q
   ```

**Install Redis:**
1. Download from [github.com/microsoftarchive/redis/releases](https://github.com/microsoftarchive/redis/releases)
2. Run `redis-server.exe`

**Setup Backend:**
```cmd
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
copy ..\.env.example .env

# Run the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Setup Frontend (new terminal):**
```cmd
cd frontend

# Install dependencies
npm install

# Run the frontend
npm run dev
```

---

### macOS Installation

#### Step 1: Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Step 2: Install Git

```bash
brew install git
git --version
```

#### Step 3: Clone the Repository

```bash
cd ~/Documents
git clone https://github.com/royrotem/Rising-Star.git
cd Rising-Star

# Switch to the development branch
git fetch origin
git checkout claude/refine-saas-description-3Bzwc

# Verify files are present
ls -la
```

> You should see: `backend/`, `frontend/`, `docker-compose.yml`, etc.

#### Step 4A: Run with Docker (Recommended)

1. Download and install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
2. Start Docker Desktop from Applications
3. Run the application:
   ```bash
   docker-compose up
   ```

#### Step 4B: Run Locally (Without Docker)

**Install dependencies:**
```bash
# Install Python
brew install python@3.11

# Install Node.js
brew install node@20

# Install PostgreSQL
brew install postgresql@15
brew services start postgresql@15

# Install Redis
brew install redis
brew services start redis
```

**Setup Database:**
```bash
createdb uaie
```

**Setup Backend:**
```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp ../.env.example .env

# Run the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Setup Frontend (new terminal):**
```bash
cd frontend

# Install dependencies
npm install

# Run the frontend
npm run dev
```

---

## Running the Application

### With Docker

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild after changes
docker-compose up --build
```

### Access Points

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

---

## Project Structure

```
Rising-Star/
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── agents/            # AI Agents framework
│   │   │   ├── base.py        # Base agent classes
│   │   │   └── orchestrator.py # Agent coordination
│   │   ├── api/               # REST API endpoints
│   │   │   └── systems.py     # System management APIs
│   │   ├── core/              # Configuration
│   │   │   └── config.py      # App settings
│   │   ├── models/            # SQLAlchemy models
│   │   │   ├── anomaly.py     # Anomaly detection models
│   │   │   ├── insight.py     # Insight models
│   │   │   ├── system.py      # System models
│   │   │   └── user.py        # User models
│   │   ├── services/          # Business logic
│   │   │   ├── anomaly_detection.py
│   │   │   ├── ingestion.py   # Data ingestion
│   │   │   └── root_cause.py  # Root cause analysis
│   │   └── main.py            # FastAPI app entry
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                   # React TypeScript frontend
│   ├── src/
│   │   ├── components/        # Reusable components
│   │   │   └── Layout.tsx
│   │   ├── pages/             # Page components
│   │   │   ├── Dashboard.tsx
│   │   │   ├── SystemDetail.tsx
│   │   │   ├── DataIngestion.tsx
│   │   │   └── Conversation.tsx
│   │   ├── services/          # API client
│   │   │   └── api.ts
│   │   ├── types/             # TypeScript types
│   │   │   └── index.ts
│   │   └── styles/            # CSS styles
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml          # Docker orchestration
├── .env.example               # Environment template
└── README.md
```

---

## API Documentation

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/systems/` | Create a new system |
| `GET` | `/api/v1/systems/` | List all systems |
| `GET` | `/api/v1/systems/{id}` | Get system details |
| `POST` | `/api/v1/systems/{id}/ingest` | Ingest data file |
| `POST` | `/api/v1/systems/{id}/confirm-fields` | Confirm schema fields |
| `POST` | `/api/v1/systems/{id}/analyze` | Run analysis |
| `POST` | `/api/v1/systems/{id}/query` | Natural language query |
| `GET` | `/api/v1/systems/{id}/impact-radar` | Get 80/20 impact data |
| `GET` | `/api/v1/systems/{id}/next-gen-specs` | Get next-gen recommendations |

### Example: Create a System

```bash
curl -X POST http://localhost:8000/api/v1/systems/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Vehicle Alpha", "system_type": "vehicle"}'
```

### Example: Upload Data

```bash
curl -X POST http://localhost:8000/api/v1/systems/{system_id}/ingest \
  -F "file=@data.csv" \
  -F "source_name=telemetry"
```

---

## Development

### Backend Development

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run with auto-reload
uvicorn app.main:app --reload

# Run tests
pytest

# Format code
black app/
```

### Frontend Development

```bash
cd frontend

# Run development server
npm run dev

# Build for production
npm run build

# Lint code
npm run lint
```

---

## Core Modules & Capabilities

### 1. Zero-Knowledge Ingestion & Adaptive Discovery

Unlike traditional tools that require months of manual tagging and schema definition, UAIE starts "blind" and learns fast.

- **Agentic Mapping:** AI agents ingest raw logs (CAN bus, JSON, CSV, binary) and autonomously map the system's DNA.
- **Human-in-the-Loop Calibration:** The system presents its mapped understanding to the engineer for verification.

### 2. Physics-Aware Anomaly Detection

Moving beyond simple threshold alerts, the system understands the physical context of the machine.

- **Behavioral Deviation:** Detects when a system is behaving "differently," even if no red lines are crossed.
- **Margin & Robustness Analysis:** Calculates "Engineering Margins" in real-time.

### 3. Root Cause Reasoning (The "Why")

The platform replaces cryptic error codes with plain-language narratives.

- **Cross-Domain Correlation:** Connects disparate dots across time and systems.
- **Natural Language Explanations:** AI explains the "why" behind anomalies.

### 4. The "AI Ready" Lifecycle

The system doesn't just monitor the current fleet; it actively designs the next one.

- **Blind Spot Detection:** Identifies data gaps.
- **Next-Gen Specs:** Generates data requirements for future products.

---

## License

MIT License - see LICENSE file for details.
