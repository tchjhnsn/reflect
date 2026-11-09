# Reflect web

The Reflect web app: Next.js (App Router) with strict TypeScript.

## Where things are

| Path | Holds |
|---|---|
| `src/core/structure.ts` | Parses `structure.json`, the Lean export, into typed values. A missing or malformed field throws, so a drift between Lean and the app fails the tests. |
| `src/core/structure.json` | A copy of `../thrivesight_core/nts/structure.json`. `npm run sync` refreshes it, and `pnpm test` fails when it differs. |
| `src/core/person.ts` | The person's stores as pure functions over `PersonState`: dated facts, relationships, and puzzles. Mirrors `../lean/NTS/Person.lean`. |
| `src/core/onboarding.ts` | Onboarding as a reducer (`answer`, `skip`, `back`, `goto`) and `finish`, which returns the records to write. Mirrors `../lean/NTS/Onboarding.lean`. |
| `src/app/` | The pages. |

## Commands

- `pnpm install`
- `pnpm test` checks the structure copy and runs the core tests.
- `pnpm typecheck`
- `pnpm dev`, then open http://localhost:3000.
- `pnpm build`
