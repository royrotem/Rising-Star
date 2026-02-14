# API Versioning Strategy

## Current Version

All API endpoints are served under the `/api/v1` prefix, configured via `API_PREFIX` in `backend/app/core/config.py`.

## Versioning Approach

UAIE uses **URL path versioning** (`/api/v1/`, `/api/v2/`, etc.).

### Why URL Path Versioning?

- Simple to understand and implement
- Easy to route at load balancer / reverse proxy level
- Clients can see the version in every request
- No header manipulation required

## Rules for Version Bumps

| Change Type | Action |
|---|---|
| New endpoint added | No version bump needed |
| New optional field in response | No version bump needed |
| New optional query parameter | No version bump needed |
| Required field removed from response | **Major version bump** (v1 -> v2) |
| Endpoint path changed | **Major version bump** |
| Request body schema breaking change | **Major version bump** |
| Behavior change for existing endpoint | **Major version bump** |

## Migration Strategy

When a new major version is introduced:

1. **Create a new router module** (e.g., `app/api/v2/systems.py`)
2. **Register with new prefix** (`/api/v2`)
3. **Keep the old version running** for a deprecation period
4. **Add `Deprecation` header** to old version responses
5. **Document migration guide** for each breaking change

## Current Endpoints (v1)

| Router | Prefix | Module |
|---|---|---|
| Systems | `/api/v1/systems` | `app.api.systems` |
| Streaming | `/api/v1/systems` | `app.api.streaming` |
| Chat | `/api/v1/chat` | `app.api.chat` |
| Reports | `/api/v1/reports` | `app.api.reports` |
| Baselines | `/api/v1/baselines` | `app.api.baselines` |
| Schedules | `/api/v1/schedules` | `app.api.schedules` |
| Feedback | `/api/v1/feedback` | `app.api.feedback` |
| Settings | `/api/v1/settings` | `app.api.app_settings` |
| Agents Status | `/api/v1/agents/status` | `app.main` (inline) |

## Non-Versioned Endpoints

| Endpoint | Purpose |
|---|---|
| `/health` | Health check (always unversioned) |
| `/docs` | OpenAPI documentation |
| `/` | SPA root / API info |
