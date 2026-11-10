# Reflect web

The Reflect web app: Next.js (App Router) with strict TypeScript.

## Where things are

| Path | Holds |
|---|---|
| `../packages/core/` | The headless core the app imports as `@reflect/core/<module>`: the element model, the person's stores, the journal and packs. See [its README](../packages/core/README.md). `npm run sync` refreshes its copy of the Lean export, and `pnpm test` fails when it differs and runs its tests. |
| `src/server/` | Server only. `auth.ts` is Better Auth (email and password) on Neon Postgres. `person-store.ts` writes onboarding in one transaction, once per person. `crypto.ts` seals every answer with AES-256-GCM under a key derived per person. |
| `src/db/migrations/0001_person.sql` | The person's tables. Append-only: a trigger refuses DELETE and any UPDATE except setting `valid_to` once. |
| `src/app/sign-in/`, `src/app/journal/` | Sign up or sign in, and the journal page onboarding leads to. |
| `e2e/` | Playwright runs of the journal and Me on desktop and phone, with axe-core checks. `SHOTS_DIR=<dir>` saves screenshots. |

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
