# AGENTS.md: reflect

The map of this repo, loaded by every agent. Keep it to about a page; details belong in `docs/` and folder READMEs.

## What this is

Reflect, an open-source reference project. Vault: `projects/reflect/`.

**Reflect is an open-source memory-graph-based self-reflection engine and backend.** It helps a user build a persistent, transparent representation of their beliefs, observations, reflections, and conversations, and supports AI-assisted insight generation on top of that graph.

## Where things are

| Path | Holds |
|---|---|
| `thrivesight_core/` |  |
| `README.md` | overview |
| `pyproject.toml` | dependencies and scripts |

## Build and test

- Python project: see `pyproject.toml` / `requirements.txt` for dependencies.

## Rules

- Never read or print `.env*` files. Refer to secrets by variable name only.
- Governing decisions are cited by id (`G-NNN` workspace, `D-NNN` Polity), not by vault path, because this repo may be cloned alone.
- Torian merges. Agents open draft PRs and do not merge.

*Drafted 2026-10-26 from the README and layout (workspace stage 3, G-027). Correct it the next time you work here.*
