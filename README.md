# Rising-Star

## UAIE - Universal Autonomous Insight Engine

### The "Tesla Standard" - Democratized as a Service

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Docker (Recommended)](#docker-recommended)
  - [Local Development](#local-development)
  - [Windows Quick Start](#windows-quick-start)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Core Features](#core-features)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Development](#development)
- [License](#license)

---

## Overview

Complex hardware companies (aerospace, robotics, MedTech, automotive) generate petabytes of data from their physical products but lack the tools to extract intelligence from it. Achieving Tesla-level data maturity typically requires years and 50+ data scientists.

**UAIE** is a domain-agnostic SaaS platform that deploys **13 specialized AI agents** to ingest raw, unstructured system data and transform it into actionable engineering intelligence. It bridges hardware physics and software logic, enabling any engineering team to achieve AI readiness from day one.

### Key Capabilities

- **Zero-Knowledge Ingestion** - Upload raw data (16+ formats), the system learns the structure autonomously
- **Physics-Aware Anomaly Detection** - 6-layer detection engine that understands physical context
- **13 AI Agent Swarm** - Specialized agents (statistical, domain, temporal, safety, etc.) powered by Claude
- **Root Cause Analysis** - Natural language explanations of why anomalies occur
- **Engineering Margins** - Real-time safety margin tracking with projected breach dates
- **Conversational AI** - Chat with your data using natural language
- **Watchdog Mode** - Scheduled auto-analysis at configurable intervals
- **PDF Reports** - Export analysis results for stakeholder communication

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/royrotem/Rising-Star.git
cd Rising-Star

# Run with Docker (recommended)
docker compose up --build

# Access the application
# Frontend: http://localhost:3001
# Backend:  http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## Prerequisites

### Docker (Recommended)

| Software | Version | Link |
|----------|---------|------|
| Docker Desktop | 4.0+ | [docker.com](https://www.docker.com/products/docker-desktop) |
| Git | 2.30+ | [git-scm.com](https://git-scm.com/) |

### Local Development

| Software | Version | Link |
|----------|---------|------|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| Git | 2.30+ | [git-scm.com](https://git-scm.com/) |

> **Note:** PostgreSQL and Redis are optional. The system uses file-based storage by default and works without them.

---

## Installation

### Docker (Recommended)

```bash
git clone https://github.com/royrotem/Rising-Star.git
cd Rising-Star

# Copy environment template
cp .env.example .env

# Edit .env and add your Anthropic API key (optional, enables AI agents)
# ANTHROPIC_API_KEY=sk-ant-...

# Start all services
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3001 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |

```bash
# View logs
docker compose logs -f

# Stop
docker compose down

# Rebuild after code changes
docker compose up --build
```

### Local Development

#### Linux / macOS

```bash
git clone https://github.com/royrotem/Rising-Star.git
cd Rising-Star

# Run the install script
./scripts/install.sh

# Start the application
./scripts/run.sh
```

Or manually:

```bash
# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

#### macOS (Homebrew)

```bash
brew install python@3.11 node
```

Then follow the Linux/macOS steps above.

### Windows Quick Start

Use the batch scripts in `scripts/`:

| Script | Description |
|--------|-------------|
| `START.bat` | Interactive menu |
| `install.bat` | Install all dependencies |
| `run.bat` | Run locally (backend + frontend) |
| `run-docker.bat` | Run with Docker |
| `update.bat` | Pull latest updates |
| `stop-docker.bat` | Stop Docker containers |

```cmd
cd Rising-Star
scripts\START.bat
```

---

## Cloud Deployment (Free Domain)

Deploy UAIE to the internet with a free `.onrender.com` domain and automatic Git-based deployments.

### One-Click Deploy on Render.com

1. **Push** this repository to GitHub
2. Go to [dashboard.render.com/blueprints](https://dashboard.render.com/blueprints)
3. Click **New Blueprint Instance** and connect your GitHub repo
4. Set the `ANTHROPIC_API_KEY` environment variable (optional)
5. Click **Apply**

Render will:
- Build the production Docker image (frontend + backend in one container)
- Deploy to `https://<your-app-name>.onrender.com`
- **Auto-deploy** on every push to the main branch
- Run health checks at `/health`
- Provide 1 GB persistent disk for data storage

> The `render.yaml` blueprint in this repo defines the full infrastructure as code.

### Manual Deploy on Render

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New Web Service**
2. Connect your GitHub repository
3. Settings:
   - **Runtime:** Docker
   - **Dockerfile Path:** `./Dockerfile`
   - **Docker Context:** `.`
   - **Plan:** Free
4. Environment variables:
   - `ANTHROPIC_API_KEY` — your API key (optional)
   - `CORS_ORIGINS` — `https://<your-app>.onrender.com`
   - `DEBUG` — `false`
5. Add a **Disk**: mount path `/app/data`, size 1 GB
6. Click **Create Web Service**

### CI/CD Pipeline

The repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on every push to `main`:

1. **Backend checks** — Python syntax validation across all modules
2. **Frontend checks** — `npm ci` + `npm run build` (TypeScript + Vite)
3. **Docker build** — Builds the production image and runs a `/health` smoke test

Render auto-deploys from the main branch after CI passes.

### Other Free Hosting Options

| Platform | How | Notes |
|----------|-----|-------|
| [Render.com](https://render.com) | `render.yaml` included | Recommended. Free tier with auto-deploy |
| [Railway.app](https://railway.app) | Docker deploy | $5/month free credit |
| [Fly.io](https://fly.io) | `fly launch` with Dockerfile | Free tier with 3 VMs |

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │     Frontend (React/TypeScript)  │
                    │   Dashboard, Wizard, Chat, Explorer │
                    └──────────────┬──────────────────┘
                                   │ REST + SSE
                    ┌──────────────▼──────────────────┐
                    │      Backend (FastAPI/Python)    │
                    │                                  │
                    │  ┌────────────────────────────┐  │
                    │  │    API Layer (8 routers)    │  │
                    │  │  systems, chat, reports,    │  │
                    │  │  streaming, feedback,       │  │
                    │  │  baselines, schedules,      │  │
                    │  │  settings                   │  │
                    │  └─────────────┬──────────────┘  │
                    │                │                  │
                    │  ┌─────────────▼──────────────┐  │
                    │  │    Service Layer            │  │
                    │  │  IngestionService (16 fmt)  │  │
                    │  │  AnalysisEngine (6 layers)  │  │
                    │  │  AnomalyDetection           │  │
                    │  │  RootCauseService            │  │
                    │  │  ChatService (Claude)        │  │
                    │  │  ReportGenerator (PDF)       │  │
                    │  │  Scheduler (Watchdog)        │  │
                    │  └─────────────┬──────────────┘  │
                    │                │                  │
                    │  ┌─────────────▼──────────────┐  │
                    │  │  AI Agent Swarm (13 agents) │  │
                    │  │  Statistical, Domain,       │  │
                    │  │  Pattern, Safety, Temporal,  │  │
                    │  │  Predictive, Reliability,    │  │
                    │  │  Compliance, Efficiency...   │  │
                    │  └────────────────────────────┘  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   File-Based Storage (/data)     │
                    │   (PostgreSQL + Redis optional)   │
                    └──────────────────────────────────┘
```

### Data Flow

1. **Upload** - User uploads raw data files (CSV, JSON, Parquet, Excel, CAN, binary, etc.)
2. **Discover** - AI agents autonomously learn data structure, infer types and physical units
3. **Confirm** - Engineer verifies the system's understanding (human-in-the-loop)
4. **Analyze** - 6-layer rule engine + 13 AI agents analyze in parallel via SSE streaming
5. **Act** - Anomalies, root causes, margins, blind spots, and recommendations are presented
6. **Chat** - Engineer can ask questions in natural language about findings
7. **Monitor** - Watchdog mode runs periodic analysis automatically

---

## Project Structure

```
Rising-Star/
├── backend/
│   ├── app/
│   │   ├── agents/                 # AI agent framework
│   │   │   ├── base.py             # BaseAgent, agent types
│   │   │   └── orchestrator.py     # Multi-agent coordination
│   │   ├── api/                    # REST API endpoints
│   │   │   ├── systems.py          # System CRUD + ingestion + analysis
│   │   │   ├── streaming.py        # SSE real-time analysis progress
│   │   │   ├── chat.py             # Conversational AI
│   │   │   ├── reports.py          # PDF report generation
│   │   │   ├── baselines.py        # Historical baseline tracking
│   │   │   ├── schedules.py        # Watchdog mode management
│   │   │   ├── feedback.py         # Anomaly feedback loop
│   │   │   ├── app_settings.py     # API key & AI configuration
│   │   │   └── schemas.py          # Pydantic request/response models
│   │   ├── core/
│   │   │   └── config.py           # Application settings (env vars)
│   │   ├── models/                 # SQLAlchemy ORM models
│   │   │   ├── system.py           # System, DataSource
│   │   │   ├── anomaly.py          # Anomaly, EngineeringMargin
│   │   │   ├── insight.py          # Insight, DataGap
│   │   │   ├── analysis.py         # RootCause, Correlation
│   │   │   ├── data.py             # Data models
│   │   │   └── user.py             # Organization, User, Conversation
│   │   ├── services/               # Business logic
│   │   │   ├── ingestion.py        # 16+ format parsing, schema discovery
│   │   │   ├── analysis_engine.py  # 6-layer anomaly detection engine
│   │   │   ├── anomaly_detection.py# Physics-aware detection + margins
│   │   │   ├── ai_agents.py        # 13 specialized AI agents
│   │   │   ├── root_cause.py       # Correlation & root cause analysis
│   │   │   ├── chat_service.py     # Claude-powered conversations
│   │   │   ├── data_store.py       # File-based persistence layer
│   │   │   ├── report_generator.py # PDF report synthesis
│   │   │   ├── scheduler.py        # Watchdog background scheduler
│   │   │   ├── baseline_store.py   # Baseline tracking & comparison
│   │   │   ├── feedback_store.py   # Anomaly feedback persistence
│   │   │   └── recommendation.py   # System type detection & naming
│   │   ├── utils.py                # Utility functions
│   │   └── main.py                 # FastAPI app entry point
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx       # Fleet overview, impact radar
│   │   │   ├── Systems.tsx         # System list/grid
│   │   │   ├── NewSystemWizard.tsx # 5-step creation wizard
│   │   │   ├── SystemDetail.tsx    # Analysis results, margins, insights
│   │   │   ├── DataIngestion.tsx   # Upload additional data
│   │   │   ├── Conversation.tsx    # AI chat interface
│   │   │   ├── AnomalyExplorer.tsx # Interactive anomaly browser
│   │   │   └── Settings.tsx        # API keys, watchdog config
│   │   ├── components/
│   │   │   ├── Layout.tsx          # Sidebar navigation
│   │   │   ├── AnalysisStreamPanel.tsx # Real-time progress
│   │   │   ├── AnomalyFeedback.tsx # Feedback buttons
│   │   │   ├── BaselinePanel.tsx   # Baseline visualization
│   │   │   ├── WatchdogPanel.tsx   # Schedule status
│   │   │   └── OnboardingGuide.tsx # Getting started checklist
│   │   ├── hooks/
│   │   │   └── useAnalysisStream.ts # SSE streaming hook
│   │   ├── services/               # API clients (axios)
│   │   ├── types/                  # TypeScript definitions
│   │   ├── utils/                  # Color & formatting utilities
│   │   └── styles/                 # Tailwind base styles
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
│
├── scripts/                        # Helper scripts
│   ├── install.sh                  # Linux/Mac: install dependencies
│   ├── run.sh                      # Linux/Mac: run locally
│   ├── run-docker.sh               # Linux/Mac: run with Docker
│   ├── stop-docker.sh              # Linux/Mac: stop Docker
│   ├── update.sh                   # Linux/Mac: pull & update deps
│   ├── START.bat                   # Windows: interactive menu
│   ├── install.bat                 # Windows: install dependencies
│   ├── run.bat                     # Windows: run locally
│   ├── run-docker.bat              # Windows: run with Docker
│   ├── stop-docker.bat             # Windows: stop Docker
│   └── update.bat                  # Windows: pull & update deps
│
├── .github/workflows/ci.yml        # CI pipeline (GitHub Actions)
├── Dockerfile                      # Production image (frontend + backend)
├── .dockerignore                   # Docker build exclusions
├── docker-compose.yml              # Full stack orchestration (development)
├── render.yaml                     # Render.com one-click deploy blueprint
├── .env.example                    # Environment variables template
├── .gitignore
└── README.md
```

---

## Core Features

### 1. Zero-Knowledge Ingestion

Upload raw data and the system autonomously discovers the structure.

**Supported formats (16+):**

| Category | Formats |
|----------|---------|
| Tabular | CSV, TSV, Parquet, Feather, Excel (XLSX/XLS) |
| Structured | JSON, JSONL/NDJSON, XML, YAML |
| Binary | CAN bus (.can), generic binary (.bin) |
| Logs | TXT, LOG, DAT |

**Discovery capabilities:**
- Field type inference (numeric, categorical, timestamp, binary)
- Physical unit detection (temperature, voltage, pressure, speed, RPM, etc.)
- Relationship discovery (correlations, causation, derived fields)
- Metadata extraction and statistical profiling

### 2. Physics-Aware Anomaly Detection (6 Layers)

| Layer | Method |
|-------|--------|
| 1 | Statistical outlier detection (Z-score, Isolation Forest) |
| 2 | Threshold breach detection (design spec violations) |
| 3 | Trend change detection (time-series trend analysis) |
| 4 | Correlation break detection (expected relationships missing) |
| 5 | Pattern anomaly detection (deviation from learned patterns) |
| 6 | Rate-of-change analysis (derivative-based detection) |

### 3. AI Agent Swarm (13 Specialized Agents)

Each agent provides a unique perspective on the data:

| # | Agent | Focus |
|---|-------|-------|
| 1 | Statistical Analyst | Distributions, outliers, significance |
| 2 | Domain Expert | Domain-specific engineering knowledge |
| 3 | Pattern Detective | Hidden patterns, unusual correlations |
| 4 | Root Cause Investigator | Deep causal reasoning |
| 5 | Safety Auditor | Safety margins, risk assessment |
| 6 | Temporal Analyst | Time-series, seasonality, change-points |
| 7 | Data Quality Inspector | Sensor drift, corruption, integrity |
| 8 | Predictive Forecaster | Trend extrapolation, failure prediction |
| 9 | Operational Profiler | Operating modes, regime transitions |
| 10 | Efficiency Analyst | Energy waste, optimization |
| 11 | Compliance Checker | Regulatory limits, standards |
| 12 | Reliability Engineer | MTBF, wear-out, degradation |
| 13 | Environmental Correlator | Cross-parameter effects |

> Agents are powered by Claude (Anthropic). Without an API key, the system falls back to rule-based analysis.

### 4. Engineering Margins

Real-time tracking of distance from design limits:
- Margin percentage calculation per component
- Trend analysis (degrading / stable / improving)
- Projected breach dates
- Safety-critical classification

### 5. Conversational AI (Chat)

Chat with your data in natural language. The AI assistant has full context of:
- System metadata and configuration
- Ingested data statistics
- Analysis results and anomalies
- Conversation history (persistent)

### 6. Watchdog Mode

Scheduled automatic analysis:
- Configurable intervals: 1h, 6h, 12h, 24h, 7d
- Background execution with status tracking
- Results saved automatically

### 7. PDF Reports

Export analysis results as PDF reports including:
- System overview and health score
- Anomaly details with severity
- Engineering margins
- Blind spots and recommendations

---

## API Reference

### System Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/systems/` | Create a new system |
| `GET` | `/api/v1/systems/` | List all systems |
| `GET` | `/api/v1/systems/{id}` | Get system details |
| `PUT` | `/api/v1/systems/{id}` | Update a system |
| `DELETE` | `/api/v1/systems/{id}` | Delete a system |

### Data Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/systems/analyze-files` | Analyze uploaded files, discover schema |
| `POST` | `/api/v1/systems/{id}/ingest` | Ingest a data file |
| `POST` | `/api/v1/systems/{id}/confirm-fields` | Confirm discovered schema |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/systems/{id}/analyze` | Trigger analysis (blocking) |
| `GET` | `/api/v1/systems/{id}/analyze-stream` | SSE streaming analysis with progress |
| `GET` | `/api/v1/systems/{id}/analysis` | Retrieve saved analysis results |
| `GET` | `/api/v1/systems/{id}/impact-radar` | Get 80/20 impact prioritization |
| `GET` | `/api/v1/systems/{id}/next-gen-specs` | Get next-gen recommendations |
| `POST` | `/api/v1/systems/{id}/query` | Natural language query |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat/systems/{id}` | Send a message |
| `GET` | `/api/v1/chat/systems/{id}/history` | Get conversation history |
| `DELETE` | `/api/v1/chat/systems/{id}/history` | Clear conversation |

### Reports

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/reports/systems/{id}/pdf` | Download PDF report |
| `POST` | `/api/v1/reports/systems/{id}/analyze-and-report` | Analyze & generate report |

### Schedules (Watchdog)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/schedules/` | List all schedules |
| `GET` | `/api/v1/schedules/{id}` | Get schedule for system |
| `POST` | `/api/v1/schedules/{id}` | Create/update schedule |
| `DELETE` | `/api/v1/schedules/{id}` | Delete schedule |

### Feedback

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/feedback/systems/{id}` | Submit anomaly feedback |
| `GET` | `/api/v1/feedback/systems/{id}/summary` | Get feedback summary |

### Baselines

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/baselines/{id}` | Get baseline data |
| `POST` | `/api/v1/baselines/{id}` | Set baseline |

### Settings

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/settings/` | Get application settings |
| `PUT` | `/api/v1/settings/` | Update settings (API key, AI config) |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/agents/status` | AI agents status |

### Examples

```bash
# Create a system
curl -X POST http://localhost:8000/api/v1/systems/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Vehicle Alpha", "system_type": "vehicle"}'

# Upload data
curl -X POST http://localhost:8000/api/v1/systems/{id}/ingest?source_name=telemetry \
  -F "file=@data.csv"

# Run streaming analysis
curl -N http://localhost:8000/api/v1/systems/{id}/analyze-stream

# Chat with your data
curl -X POST http://localhost:8000/api/v1/chat/systems/{id} \
  -H "Content-Type: application/json" \
  -d '{"message": "What anomalies did you find?"}'

# Download PDF report
curl -O http://localhost:8000/api/v1/reports/systems/{id}/pdf
```

Full interactive documentation available at http://localhost:8000/docs when the server is running.

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | UAIE | Application name |
| `APP_VERSION` | 0.1.0 | Application version |
| `DEBUG` | false | Enable debug mode |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `CORS_ORIGINS` | localhost:3000,3001,5173 | Allowed CORS origins (comma-separated) |
| `DATABASE_URL` | postgresql+asyncpg://... | PostgreSQL connection (optional) |
| `DATABASE_POOL_SIZE` | 20 | DB connection pool size |
| `REDIS_URL` | redis://localhost:6379/0 | Redis connection (optional) |
| `SECRET_KEY` | (change in production) | JWT signing key |
| `ANTHROPIC_API_KEY` | (none) | Anthropic API key for AI agents |
| `MAX_FILE_SIZE_MB` | 500 | Max upload file size |
| `ANOMALY_THRESHOLD` | 0.95 | Anomaly detection sensitivity |
| `DETECTION_WINDOW_HOURS` | 24 | Detection window |

### API Key Setup

For AI-powered features (13 agents, chat, recommendations):

1. Get an API key from [console.anthropic.com](https://console.anthropic.com/)
2. Either:
   - Set `ANTHROPIC_API_KEY` in `.env`
   - Or configure it in the UI: Settings page

Without an API key, the system operates with rule-based analysis only (6-layer detection engine still works).

---

## Development

### Backend

```bash
cd backend
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Run with auto-reload
uvicorn app.main:app --reload

# Run tests
pytest

# Type checking
mypy app/
```

### Frontend

```bash
cd frontend

# Development server
npm run dev

# Production build
npm run build

# Lint
npm run lint
```

### Tech Stack

**Backend:**
- FastAPI 0.109 (Python 3.11+)
- SQLAlchemy 2.0 (ORM, async)
- Anthropic SDK (Claude AI)
- pandas, numpy, scipy, scikit-learn (data analysis)
- fpdf2 (PDF generation)

**Frontend:**
- React 18 + TypeScript
- Vite 5 (build tool)
- Tailwind CSS 3 (styling)
- React Router 6 (routing)
- Axios (HTTP client)
- Recharts (charts)
- Lucide React (icons)

**Infrastructure:**
- Docker + Docker Compose
- Nginx (production frontend)
- PostgreSQL 15 (optional)
- Redis 7 (optional)

---

## License

MIT License - see LICENSE file for details.
