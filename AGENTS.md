# AGENTS.md: reflect

The map of this repo, loaded by every agent. Keep it to about a page; details belong in `docs/` and folder READMEs.

## What this is

Reflect, an open-source reference project. Vault: `projects/reflect/`.

**Reflect is an open-source memory-graph-based self-reflection engine and backend.** It helps a user build a persistent, transparent representation of their beliefs, observations, reflections, and conversations, and supports AI-assisted insight generation on top of that graph.

## Where things are

| Path | Holds |
|---|---|
| `thrivesight_core/` | the hackathon reasoning engine (Python) |
| `packages/cli/` | the `reflect` command over an instance in a folder (`init`, `write`, `add`, `close`, `list`, `check`); see `packages/cli/README.md` |
| `action/` | the GitHub Action an instance repository runs: refuses a public repository, checks the vault and records hand-written changes |
| `packages/core/` | the headless core: the person's stores, journal and packs, in plain TypeScript that runs in Node with no database; see `packages/core/README.md` |
| `web/` | the Reflect web app (Next.js), one interface over the core; see `web/README.md` |
| `docs/design/` | designs; `HEADLESS_REFLECT_2026-11-10.md` is the plan for one core with two hosts; `LIFE_AND_MIND_DOMAIN_2026-11-11.md` and `LIFE_AND_MIND_SYSTEM_2026-11-11.md` propose the life and mind layers, ideas from micro to published, and the blog (not built) |
| `README.md` | overview |
| `pyproject.toml` | dependencies and scripts |

## Build and test

- Python project: see `pyproject.toml` / `requirements.txt` for dependencies.
- `cd web && pnpm test` runs the core's, the CLI's and the web app's tests (the Neon ones only with `DATABASE_URL`); `pnpm typecheck` type-checks them.
- The Number Theory of the Soul (the element table, its Lean model and the labeling reader) left Reflect on 2026-11-11 for its own repository. Nothing here imports it.

## Rules

- Never read or print `.env*` files. Refer to secrets by variable name only.
- Governing decisions are cited by id (`G-NNN` workspace, `D-NNN` Polity), not by vault path, because this repo may be cloned alone.
- Agents open draft PRs and merge their own on green CI (Torian, 2026-11-07).

*Drafted 2026-10-26 from the README and layout (workspace stage 3, G-027). Correct it the next time you work here.*
