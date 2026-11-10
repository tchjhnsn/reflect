# Reflect web

The Reflect web app: Next.js (App Router) with strict TypeScript.

## Where things are

| Path | Holds |
|---|---|
| `../packages/core/` | The headless core the app imports as `@reflect/core/<module>`: the element model, the person's stores, the journal and packs. See [its README](../packages/core/README.md). `npm run sync` refreshes its copy of the Lean export, and `pnpm test` fails when it differs and runs its tests. |
| `src/server/` | Server only. `auth.ts` is Better Auth (email and password) on Neon Postgres, and also an OAuth server for a person's own AI (the `@better-auth/mcp` plugin: clients register themselves, the person signs in and allows them on `/consent`, and tokens are bound to `/api/mcp`). `site.ts` fixes the app's public address. `mcp-tools.ts` is what that AI can do: `reflect_guide`, `list_entries`, `write_entry` (today's entry, in the person's words), `get_entry`, `keep_reading` (only what the person said yes to; anchored in the entry's words) and `concept_history`, each a plain function of the person. `person-store.ts` writes onboarding in one transaction, once per person. `crypto.ts` seals every answer with AES-256-GCM under a key derived per person, and `tag` gives a concept a keyed hash so the same person, act or quality is found again without storing it in the clear. `binding-store.ts` keeps readings (`keepReading`, one transaction, write-once tables `concept`, `binding`, `binding_concept` from migration 0007), lists an entry's readings and a concept's history across entries (`conceptHistory`, `onThisConcept`). The app does not show bindings yet (not built). |
| `src/db/migrations/0001_person.sql` | The person's tables. Append-only: a trigger refuses DELETE and any UPDATE except setting `valid_to` once. |
| `src/app/sign-in/`, `src/app/journal/` | Sign up or sign in, and the journal page onboarding leads to. |
| `src/app/api/mcp/`, `src/app/consent/`, `src/app/.well-known/` | The MCP endpoint a person's own AI (Claude, ChatGPT or any MCP client) connects to, the page where the person allows it, and the OAuth discovery documents clients look for. To connect, add `<app address>/api/mcp` as a custom connector in the AI app. |
| `e2e/` | Playwright runs of the journal and Me on desktop and phone, with axe-core checks, and the whole MCP connection (`mcp.spec.ts`: register, sign in, allow, token, tools). `SHOTS_DIR=<dir>` saves screenshots. |

## Environment

Set by name only, never committed: `DATABASE_URL` (the Neon project `reflect`, pooled), `BETTER_AUTH_SECRET`, `REFLECT_DATA_KEY` (32 random bytes, base64), and `BETTER_AUTH_URL` in deployments (also the address MCP tokens are bound to; `REFLECT_URL` overrides it). `REFLECT_RATE_LIMIT=off` turns the sign-in rate limit off for the browser tests only.

## Commands

- `pnpm install`
- `pnpm migrate` creates the Better Auth tables (including the OAuth tables for MCP clients) and the person's tables.
- `pnpm test` checks the structure copy and runs the app's, the core's and the CLI's tests.
- `pnpm bundle export --user <id> --out <file>` and `pnpm bundle import <file> --user <id>` move one person's journal between Postgres and a bundle file (see `../packages/core/src/bundle.ts`), against whichever branch `DATABASE_URL` names. Import goes only into an empty journal.
- `pnpm typecheck`
- `pnpm dev`, then open http://localhost:3000.
- `pnpm build`
- `pnpm build && pnpm e2e` runs the browser tests (Playwright 1.63 with Chromium).
