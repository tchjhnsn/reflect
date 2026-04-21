# Reflect

**Reflect is an open-source memory-graph-based self-reflection engine and backend.** It helps users build a persistent, transparent representation of their beliefs, observations, reflections, and conversations, and supports AI-assisted insight generation on top of that graph.

This repository contains the author's continuation of work originally begun during the ThriveSight hackathon, now released under a unified name and an open-source license.

## Origin and scope

Reflect began as **ThriveSight**, a hackathon project co-built with a collaborator. When the hackathon ended, the collaborator retained the **ThriveSight** name and continued independent development of the front-end experience under that brand. This repository collects the author-owned portions of the work — specifically:

- The **reasoning engine** (signal generation, clustering, coordinate system, context assembly, graph agent, insight engine, LLM prompt templates, persona configuration)
- The **backend architecture** (authentication, workspace-scoped data, live conversation service, Neo4j graph persistence, analytics)
- The **research corpus** (memory hypotheses, signal address architecture, federated graph architecture, reflect graph contract family)

and releases them under Apache License 2.0 as a reference implementation and a foundation others can build on.

The **ThriveSight** name and front-end implementation remain with the original collaborator and are **not** part of this repository.

## Current repository structure

```
reflect/
├── LICENSE                   Apache License 2.0
├── README.md                 this file
├── pyproject.toml            Python package definition
├── thrivesight_core/         ← Python reasoning engine (package name retained for internal imports)
│   ├── signal_engine.py
│   ├── cluster_engine.py
│   ├── coordinate_system.py
│   ├── context_assembly.py
│   ├── graph_agent.py
│   ├── insight_engine.py
│   ├── llm_prompts.py
│   └── persona_config.py
└── _ideas/                   idea notes (will migrate to docs/ideas/)
```

The internal Python package is still named `thrivesight_core` for import stability. Renaming it to `reflect_core` (or similar) is tracked in a migration issue. Until then, `from thrivesight_core.signal_engine import …` is the canonical import.

## Planned consolidation

Reflect is being consolidated into a single repository from what were previously separate codebases. The following folders are expected to land here over time via `git subtree` merges:

- `api/` — the Django REST backend (currently at [tchjhnsn/reflect-api](https://github.com/tchjhnsn/reflect-api), archived)
- `docs/` — architecture, research, and thesis documents (currently in the author's vault; see the migration tracking issue)
- `packages/` — TypeScript domain packages (values ontology, scoring, number-theory framework) from what was previously [tchjhnsn/ai-lab](https://github.com/tchjhnsn/ai-lab)

Track the consolidation progress in the [issues](https://github.com/tchjhnsn/reflect/issues).

## Install (Python engine)

```bash
pip install git+https://github.com/tchjhnsn/reflect.git
```

Or for local development:

```bash
git clone https://github.com/tchjhnsn/reflect.git
cd reflect
pip install -e .
```

The engine requires Python 3.11+ and declares its own dependencies (`anthropic`, `neomodel`, `neo4j`, `jsonschema`) via `pyproject.toml`.

## Status

- **Python engine:** released. Not actively deployed; used as a reference implementation.
- **Django backend:** not yet merged; lives at [tchjhnsn/reflect-api](https://github.com/tchjhnsn/reflect-api) (archived).
- **Docs corpus:** not yet migrated into this repo.

Reflect is maintained at a best-effort cadence by a solo author. Pull requests are welcome but may take time to review. Forks and derivative projects are explicitly encouraged — that's the point of releasing this.

## Credits

- **Original co-author** (hackathon): thanks for the collaboration. ThriveSight as a product and a name remains yours.
- **Author of this repository:** Torian Johnson ([@tchjhnsn](https://github.com/tchjhnsn))

## License

[Apache License 2.0](./LICENSE). TL;DR: you can use, modify, and distribute this code for any purpose, including commercial. You must preserve the license notice and state any significant modifications. There is no warranty.
