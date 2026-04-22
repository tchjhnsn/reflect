# Reflect

**Reflect is an open-source memory-graph-based self-reflection engine and backend.** It helps a user build a persistent, transparent representation of their beliefs, observations, reflections, and conversations, and supports AI-assisted insight generation on top of that graph.

This repository is my continuation of work I originally began during the **Data Portability Hackathon** — sponsored by the AI Collective, AITX, the Data Transfer Initiative, and the University of Texas at Austin School of Law and Innovation — now consolidated under a unified name and released under an open-source license.

## Origin

Going into the hackathon, I was already pursuing ideas in this domain through prior lines of work I called **NTS** (Number Theory of the Soul) and **Journey**. I was grateful for the opportunity to build something with collaborators under the high-pressure, time-constrained format of a competition, and the event gave me a reason to materialize those ideas into a working system quickly.

What I built was largely based on my intuitions rather than on formal studies of machine learning. Instead of designing a neural network from first principles to achieve the project's goals, I set out to build a **harness around existing models** — manipulating the system so that, given specified inputs from the user, it would produce outputs shaped the way I wanted, and then relationally storing both the user's inputs and the harness's outputs in a persistent knowledge graph.

That is, honestly, what this repository is: a working prototype of the harness-and-graph approach. In my own words, a **bootleg version** of what I'm ultimately trying to accomplish. I'm now devoting my studies to machine learning and attention mechanisms so that future iterations can replace the harness with a purpose-designed architecture that achieves the same goals natively, with fewer compromises.

I'm releasing this work openly in the spirit of the hackathon that hosted it — data portability, transparency, and the right to inspect and modify the systems that mediate one's own thinking. If it's useful to you, take it.

## What my partner retained

I built this with a collaborator, [@clarkmyfancy](https://github.com/clarkmyfancy). After the hackathon, my partner kept the **ThriveSight** name and the front-end they built, and has continued independent development under that brand. Everything in this repository is my portion of the work:

- the **reasoning engine** (signal generation, clustering, coordinate system, context assembly, graph agent, insight engine, LLM prompts, persona configuration)
- the **Django REST backend** that exposes it (authentication, workspace-scoped data, live conversation, graph persistence, analytics)
- the **research corpus** (memory hypotheses, signal address architecture, federated graph architecture, reflect graph contract family)

The **ThriveSight** name and the front-end belong to my partner and are not part of this repository.

## Current repository structure

```
reflect/
├── LICENSE                   Apache License 2.0
├── README.md                 this file
├── pyproject.toml            Python package definition
├── thrivesight_core/         ← Python reasoning engine (internal name retained for import stability)
│   ├── signal_engine.py
│   ├── cluster_engine.py
│   ├── coordinate_system.py
│   ├── context_assembly.py
│   ├── graph_agent.py
│   ├── insight_engine.py
│   ├── llm_prompts.py
│   └── persona_config.py
├── api/                      ← Django REST backend (squash-imported 2026-04-22 from tchjhnsn/reflect-api)
│   ├── manage.py
│   ├── requirements.txt
│   ├── reflect_api/          (Django project configuration + root URL wiring)
│   ├── events_api/           (HTTP transport, services, live-conversation, graph helpers, analytics)
│   ├── datasets/
│   ├── scripts/
│   └── …
└── _ideas/                   idea notes (will migrate to docs/ideas/)
```

The internal Python package is still named `thrivesight_core` for import stability. Renaming it to `reflect_core` is tracked as an issue. The `api/` subfolder was imported from the (now-archived) standalone `tchjhnsn/reflect-api` repo via a squash import — full per-commit history of the Django backend remains preserved at that archived repo.

## Planned consolidation

Reflect is being consolidated into a single repository from previously separate codebases.

- ✅ **`api/`** — Django REST backend. Imported 2026-04-22 from the archived [tchjhnsn/reflect-api](https://github.com/tchjhnsn/reflect-api).
- ⏳ **`docs/`** — architecture, research, and thesis documents (currently in my vault; see the migration tracking issue).
- ⏳ **`packages/`** — TypeScript domain packages (values ontology, scoring, number-theory framework) from the archived [tchjhnsn/ai-lab](https://github.com/tchjhnsn/ai-lab).

Track the remaining consolidation progress in the [issues](https://github.com/tchjhnsn/reflect/issues).

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

The engine requires Python 3.11+ and declares its dependencies (`anthropic`, `neomodel`, `neo4j`, `jsonschema`) via `pyproject.toml`.

## Status

- **Python engine:** released. Reference implementation; not actively deployed.
- **Django backend:** not yet merged; lives at [tchjhnsn/reflect-api](https://github.com/tchjhnsn/reflect-api) (archived).
- **Docs corpus:** not yet migrated into this repo.

I maintain this solo, at a best-effort cadence. Pull requests are welcome but may take time to review. Forks and derivative projects are explicitly encouraged — that's the point of releasing this.

## Credits

- **Hackathon context:** the **Data Portability Hackathon**, sponsored by the AI Collective, AITX, the Data Transfer Initiative, and the University of Texas at Austin School of Law and Innovation. Thanks to the organizers for a format that made fast, honest building possible.
- **Collaborator:** [@clarkmyfancy](https://github.com/clarkmyfancy). Thank you for building alongside me. **ThriveSight** as a product and as a name is yours; the front-end you built continues its life under that brand, independent of this repository.
- **Author:** Torian Johnson ([@tchjhnsn](https://github.com/tchjhnsn)).

## License

[Apache License 2.0](./LICENSE). In short: you can use, modify, and distribute this code for any purpose, including commercial. You must preserve the license notice and state any significant modifications. There is no warranty.
