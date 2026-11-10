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
| `src/layout.ts` | Where records sit in an instance folder: a vault (`journal/`, `reflections/`, `concepts/`, `library/`, `ventures/<venture of a lifetime>/`, `telos/`, `intents/` at the root) or the older `people/<id>/` layout. |
| `src/files-journal.ts` | The folder host: entries as `journal/<date>-<id>.md` (front matter optional, so a hand-written file is an entry; front matter the person adds is kept on later writes), reflections and labelings as `reflections/<entry>.json`, sparks with their concept notes in `concepts/`. |
| `src/spark.ts` | Object writing: the forms (1, 5, 10, 15 minutes or one sentence), concrete nouns to start from, the prompt, and a spark's front matter and concept note. |
| `src/memory.ts` | Memory: `memory.jsonl`, one hash-chained line per write (`verifyMemory` finds a changed or removed line). |
| `src/journal-contract.ts` | The behaviour every host must have. The folder host runs it in `files-journal.test.ts`, and the Postgres host (`postgresJournal` in `../../web/src/server/journal-store.ts`) in its own tests against Neon. |
| `src/instance.ts` | An instance in a folder: `initInstance` (writes the vault and `template/reflect.yml` as the instance's workflow), `readConfig`, `checkInstance`, `adopt`. |
| `src/reader.ts` | The reader: Jev's element odds and the basis of each self-term, gpt-oss to find self-terms. The web app adds the Vercel OIDC token. |
| `src/draft.ts` | A draft reflection (`makeDraft`) and what keeping it records (`keepDraft`). |
| `src/bundle.ts` | A bundle: one person's journal as one JSON document, the same from either host. Moving hosts is an export from one and an import into the other's empty journal. |
| `src/bindings.ts` | Concepts and bindings. A binding ties one element, at a weight, to one or more concepts the entry names: an entity (a noun: a person, place, thing, group or idea), an event (a verb: something that happened or was done), a quality (an adjective or adverb: how something was) or a time. Each attachment keeps the entry's own words for the concept ("left" for the concept "leave") and, when given, its role (agent, patient, place, time, manner, cause, goal). A reading is the bindings kept together at one time; the first is as felt then and later ones never replace it. `bindingProblems` checks a reading, `conceptKey` is a concept's identity (kind, sort, name), `asFeltThen` and `asOf` pick a reading, and `presentMeaning` fades a concept's history by element (the half-lives are parameters, not findings). The Postgres store is `../../web/src/server/binding-store.ts`; the folder host does not keep bindings yet (not built). |
| `src/theme.ts` | The app's themes. |
| `src/salience-prompt.ts` | Generated from `../../thrivesight_core/nts/prompts/salience.md`. Do not edit. |

Modules import each other by their `.ts` names and import JSON with `with { type: "json" }`, so Node runs them directly (`node -e 'import("./src/state.ts")'`, Node 22.18 or later).

## Tests

`pnpm test` in `web/` runs `src/*.test.ts` with vitest. The app still calls the Postgres store's functions directly.
