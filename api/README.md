# ThriveSight API

Django + Django REST Framework backend for authentication, workspace-scoped data, live conversation, graph queries, analytics, imports, and supporting analysis flows.

## Architecture

The backend is organized around a few clear layers:

- `thrivemind_api/`: Django project configuration and root URL wiring
- `events_api/urls.py`: endpoint registration under `/api/`
- `events_api/views.py`: HTTP transport layer for request validation and response shaping
- `events_api/live_conversation_service.py`: application service for the live conversation workflow
- `events_api/*.py`: domain and infrastructure modules for analysis, graph work, imports, and exports

Current persistence model:

- SQL: `Workspace`, `WorkspaceMembership`, `Event`, `PatternRun`, `Pattern`
- Graph: Neo4j / Neomodel-backed knowledge graph reads and writes

## Key Modules

- `events_api/auth_views.py`: signup, login, logout, session, export, and deletion endpoints
- `events_api/workspaces.py`: active workspace resolution and membership-aware scoping
- `events_api/services.py`: deterministic evidence-backed event and pattern services
- `events_api/conversation.py`: transcript analysis pipeline
- `events_api/live_conversation_service.py`: prompt assembly, reply generation, graph persistence, signal linking, and pipeline tracing
- `events_api/llm_client.py`: shared Anthropic integration and LLM helpers
- `events_api/graph_tests.py`: analytics payload builder for the Graph Tests UI
- `events_api/live_graph.py`, `events_api/graph_sync.py`, `events_api/neo4j_client.py`: graph persistence and query helpers

## Authentication And Workspace Model

- Authentication uses JWT bearer tokens.
- Refresh tokens rotate and are blacklisted on logout.
- The default workspace is the user's personal workspace.
- Requests can target another accessible workspace by sending `X-Workspace-Id` or `workspace_id`.
- Workspace access is enforced through `WorkspaceMembership` or workspace ownership.

## Local Development

From `apps/api`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 manage.py migrate
python3 manage.py runserver
```

API base URL:

- `http://127.0.0.1:8000/api/`

## Environment Variables

Common settings:

- `SECRET_KEY`
- `DEBUG`
- `ALLOWED_HOSTS`
- `DATABASE_URL`
- `CORS_ALLOWED_ORIGINS`
- `CORS_ALLOW_ALL_ORIGINS`
- `THRIVESIGHT_LLM_MODEL`
- `THRIVESIGHT_LLM_MAX_RETRIES`

Service integrations:

- `ANTHROPIC_API_KEY`
- `ELEVENLABS_API_KEY`
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`

Legacy / fallback graph configuration is also supported through `NEOMODEL_NEO4J_BOLT_URL`.

## Endpoint Families

Auth and account:

- `/api/auth/me/`
- `/api/auth/signup/`
- `/api/auth/login/`
- `/api/auth/refresh/`
- `/api/auth/logout/`
- `/api/auth/data/`
- `/api/auth/account/`
- `/api/auth/export/conversations/`
- `/api/auth/export/memory-graph/`

Workspace data and analysis:

- `/api/events/`
- `/api/patterns/`
- `/api/patterns/recompute/`
- `/api/ask/`
- `/api/analysis/`
- `/api/analysis/ask/`

Conversation and graph surfaces:

- `/api/conversation/`
- `/api/graph/query/`
- `/api/graph-tests/`

Additional workflows:

- `/api/import/conversations/`
- `/api/audio/realtime-token/`
- `/api/audio/transcribe/`
- `/api/reflection/start/`
- `/api/reflection/next/`
- `/api/profiles/`
- `/api/screenplay/*`

## Testing

Primary regression commands:

```bash
python3 manage.py test events_api.tests
python3 manage.py test events_api.tests.test_integration
```

The test suite is designed to run without a live Neo4j instance; graph failures are tolerated in several degraded-mode paths.
