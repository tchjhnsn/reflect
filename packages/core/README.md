# Reflect core

The headless core of Reflect, in plain TypeScript with no database and no Next.js. The web app imports it as `@reflect/core/<module>`. The plan for it is `docs/design/HEADLESS_REFLECT_2026-11-10.md`: one core, with a git repository or our Postgres as the host of a person's records.

## Where things are

| Path | Holds |
|---|---|
| `src/structure.ts`, `src/structure.json` | The Lean export (`../../lean`) parsed into typed values. `structure.json` is a copy of `../../thrivesight_core/nts/structure.json`; `pnpm sync` in `web/` refreshes it. |
| `src/state.ts` | The 16 elements, their categories, salience readings and `decide`. |
| `src/person.ts`, `src/onboarding.ts` | The person's stores as pure functions over `PersonState`, and the onboarding reducer Me still uses. Mirror `../../lean/NTS/Person.lean` and `Onboarding.lean`. |
| `src/journal.ts`, `src/packs.ts`, `src/reflect.ts` | Entries and prompts, prompt packs, and reflecting with frames. |
| `src/store.ts` | The store interface, `Journal`: one person's journal, append-only, whichever host keeps it. |
| `src/files-journal.ts` | The folder host: entries as `people/<id>/entries/<date>-<id>.md` (front matter optional, so a hand-written file is an entry), reflections and labelings as `people/<id>/reflections/<entry>.json`. |
| `src/memory.ts` | Memory: `memory.jsonl`, one hash-chained line per write (`verifyMemory` finds a changed or removed line). |
| `src/journal-contract.ts` | The behaviour every host must have. The folder host runs it in `files-journal.test.ts`, and the Postgres host (`postgresJournal` in `../../web/src/server/journal-store.ts`) in its own tests against Neon. |
| `src/theme.ts` | The app's themes. |
| `src/salience-prompt.ts` | Generated from `../../thrivesight_core/nts/prompts/salience.md`. Do not edit. |

Modules import each other by their `.ts` names and import JSON with `with { type: "json" }`, so Node runs them directly (`node -e 'import("./src/state.ts")'`, Node 22.18 or later).

## Tests

`pnpm test` in `web/` runs `src/*.test.ts` with vitest. The CLI, the GitHub Action, and export and import between hosts are not built yet. The app still calls the Postgres store's functions directly.
