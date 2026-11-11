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
| `src/layout.ts` | Where records sit in an instance folder: a vault (the lifetime folders, `journal/`, `reflections/`, `concepts/`, `documents/`, `telos/`, `intents/` at the root; `work/` ventures left to Naive) or the older `people/<id>/` layout. |
| `src/lifetime.ts` | The Ventures of a Lifetime as folders: each folder and its guide, the legacy templates, the eight needs and how each is met (`keepNeed`), the regions and their states with dated passages and the state rebuilt from them (`keepPassage`, `lifeState`), the legacy prompts, and their checks. Folders are regions and associations, never states. |
| `src/files-journal.ts` | The folder host: entries as `journal/<date>-<id>.md` (front matter optional, so a hand-written file is an entry; front matter the person adds is kept on later writes), reflections and labelings as `reflections/<entry>.json`, sparks with their concept notes in `concepts/`. |
| `src/learning.ts` | What the person knows, in `learn/`, with books at the center: books (`learn/books/`, status and dates, the subjects they cover), subjects (`learn/subjects/`, nodes of the graph with `requires`, `part_of` and `same_as`, aspire or need, and dated grasp lines the person gives: teach, explain, gist or forgot), and resources (`learn/resources/`, other links). `learnMap` derives known from the latest grasp, the subjects of a read book not yet rated, and the fringe. Nothing is known until the person rates it. `learnProblems` checks it. |
| `src/openlibrary.ts` | Looking a book up in Open Library: `searchBooks` by title or author, `lookupIsbn` for one edition with its work's subjects and table of contents, both over an injected fetch. The catalog is editable by anyone, so its entry is a suggestion saved beside the book note (`learn/books/<book>.openlibrary.json`); the book note changes only by the person. |
| `src/transcript.ts` | A transcript as courses: `draftTranscript` asks a model for rows and `anchorCourses` keeps only rows whose title (and grade) are in the text; `keepCourse` writes a confirmed row to `learn/courses/` linked to a subject. The grade is the school's; grasp is the person's. |
| `src/credentials.ts` | Credentials (`learn/credentials/`), with `checkBadge` reading an Open Badges assertion; vouches as dated lines on a subject (`keepVouches`); recommendations kept whole (`learn/vouches/`). None set grasp. |
| `src/linkedin.ts` | Reads a LinkedIn data export: endorsements, recommendations, certifications and listed skills, columns found by name (`parseCsv`, `linkedInDate`, `readLinkedInExport`). |
| `src/documents.ts` | Official documents as a custody checklist: the catalog of document types (grouped after FEMA's Emergency Financial First Aid Kit), how each is normally kept as of 2026-11 in the United States and what we recommend, custody cards (`documents/<type>.md`), connections (`connections/<id>.json`, a service and the types it holds, never credentials) and their checks. |
| `src/record-io.ts` | Writes one record and logs it in `memory.jsonl`. |
| `src/release.ts` | The release plan (`legacy/release.json`): trustees, confirmations, the wait, and gifts, with the hard rule that only `legacy/` and entries marked `legacy: true` can be given (`giftRefusal`), and `previewRelease`, what any release would hand each person. The Reflect app carries a release out (`../../web/src/server/legacy-store.ts`); a person's AI sends the plan with `keep_legacy_plan`. |
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
