# AGENTS.md: reflect

The map of this repo, loaded by every agent. Keep it to about a page; details belong in `docs/` and folder READMEs.

## What this is

Reflect, an open-source reference project. Vault: `projects/reflect/`.

**Reflect is an open-source memory-graph-based self-reflection engine and backend.** It helps a user build a persistent, transparent representation of their beliefs, observations, reflections, and conversations, and supports AI-assisted insight generation on top of that graph.

## Where things are

| Path | Holds |
|---|---|
| `thrivesight_core/` | the reasoning engine; `thrivesight_core/nts/` is the element model of the Number Theory of the Soul (see its README) |
| `packages/cli/` | the `reflect` command over an instance in a folder (`init`, `write`, `add`, `close`, `list`, `check`); see `packages/cli/README.md` |
| `action/` | the GitHub Action an instance repository runs: check, keep merged drafts, open a pull request with the reader's guesses |
| `packages/core/` | the headless core: element model, the person's stores, journal and packs, in plain TypeScript that runs in Node with no database; see `packages/core/README.md` |
| `web/` | the Reflect web app (Next.js), one interface over the core; see `web/README.md` |
| `docs/design/` | designs; `HEADLESS_REFLECT_2026-11-10.md` is the plan for one core with two hosts |
| `lean/` | Lean model of the soul's scopes, model outputs, frames, the person's stores and onboarding |
| `README.md` | overview |
| `pyproject.toml` | dependencies and scripts |

## Build and test

- Python project: see `pyproject.toml` / `requirements.txt` for dependencies.
- `python -m unittest thrivesight_core.nts.test_nts thrivesight_core.nts.test_investigate thrivesight_core.nts.test_person thrivesight_core.nts.test_onboarding` tests the element model.
- `cd lean && lake build` checks the Lean model (toolchain in `lean/lean-toolchain`, no Mathlib); `lake exe nts-export > ../thrivesight_core/nts/structure.json` refreshes the export.

## Rules

- Never read or print `.env*` files. Refer to secrets by variable name only.
- Governing decisions are cited by id (`G-NNN` workspace, `D-NNN` Polity), not by vault path, because this repo may be cloned alone.
- Torian merges. Agents open draft PRs and do not merge.

*Drafted 2026-10-26 from the README and layout (workspace stage 3, G-027). Correct it the next time you work here.*
