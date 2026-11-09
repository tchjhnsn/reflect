# Reflect web

The Reflect web app: Next.js (App Router) with strict TypeScript.

## Where things are

| Path | Holds |
|---|---|
| `src/core/structure.ts` | Parses `structure.json`, the Lean export, into typed values. A missing or malformed field throws, so a drift between Lean and the app fails the tests. |
| `src/core/structure.json` | A copy of `../thrivesight_core/nts/structure.json`. `npm run sync` refreshes it, and `pnpm test` fails when it differs. |
| `src/core/person.ts` | The person's stores as pure functions over `PersonState`: dated facts and relationships. Mirrors `../lean/NTS/Person.lean`. |
| `src/core/onboarding.ts` | Onboarding as a reducer (`answer`, `skip`, `back`, `goto`) and `finish`, which returns the records to write. Mirrors `../lean/NTS/Onboarding.lean`. |
| `src/app/welcome/` | Onboarding: one question per screen, a quiet Skip, and a review where each answer can be edited or removed before anything is kept. `copy.ts` holds the fixed wording. Answers are kept only on "Keep these". |
| `src/server/` | Server only. `auth.ts` is Better Auth (email and password) on Neon Postgres. `person-store.ts` writes onboarding in one transaction, once per person. `crypto.ts` seals every answer with AES-256-GCM under a key derived per person. |
| `src/db/migrations/0001_person.sql` | The person's tables. Append-only: a trigger refuses DELETE and any UPDATE except setting `valid_to` once. |
| `src/app/sign-in/`, `src/app/journal/` | Sign up or sign in, and the journal page onboarding leads to. |
| `e2e/onboarding.spec.ts` | Playwright runs of the flow on desktop and phone, with axe-core checks. `SHOTS_DIR=<dir>` saves screenshots. |

## Environment

Set by name only, never committed: `DATABASE_URL` (the Neon project `reflect`, pooled), `BETTER_AUTH_SECRET`, `REFLECT_DATA_KEY` (32 random bytes, base64), and `BETTER_AUTH_URL` in deployments. `REFLECT_RATE_LIMIT=off` turns the sign-in rate limit off for the browser tests only.

## Commands

- `pnpm install`
- `pnpm migrate` creates the Better Auth tables and the person's tables.
- `pnpm test` checks the structure copy and runs the core tests.
- `pnpm typecheck`
- `pnpm dev`, then open http://localhost:3000.
- `pnpm build`
- `pnpm build && pnpm e2e` runs the browser tests (Playwright 1.63 with Chromium).
