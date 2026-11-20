# Reflect

Reflect is a private place for one person's life: their journal, calendar, money, and reading and learning, with an assistant that can act across all of it. It is headless. The core is plain TypeScript that runs in Node over a vault folder, with no database. A `reflect` command line and an MCP server work on that folder, and a web app is one host over the same core. When the person chooses to publish, Reflect writes files to a `published/` folder that their own site reads. It never edits a site's code.

Its ontology follows Aristotle. A person's works are sorted by their end. Making has a product as its end: code, products, art and finished writing. Knowing has truth as its end: research, study and reading. Doing has the action itself as its end: service, care, civic work, talks and events. What a person says is kept as quotes, and claims and stances are tags on those quotes. Reflect records what can be observed, in words, deeds and things made. It does not model the inner mind.

Reflect is in active development and its author uses it every day. The command line and the core are the most stable parts. The design documents in `docs/design/` say what is built and what is not.

## For agents and contributors

`AGENTS.md` is the map every agent loads. The main folders:

| Path | Holds |
|---|---|
| `packages/core/` | the headless core: the person's stores, journal and packs, and `published/`, the files a site reads (Reflect writes files, never a site's code), in plain TypeScript that runs in Node with no database; see `packages/core/README.md` |
| `packages/cli/` | the `reflect` command over an instance in a folder (`init`, `write`, `add`, `close`, `list`, `check`, `present` for the presentation gate, `quote`, `tag` and `claim` for quotes and their tags, `publish`, `unpublish`, `approve` and `decline` for the `published/` folder a site reads, and `actions`, `act` and `mcp` over the core's action registry); see `packages/cli/README.md` |
| `web/` | the Reflect web app (Next.js), one interface over the core; see `web/README.md` |
| `action/` | the GitHub Action an instance repository runs: refuses a public repository, checks the vault and records hand-written changes |
| `packages/virtu/` | the Virtu design system's editor and check, and the personal site's design system (`site/`); see `packages/virtu/README.md` |
| `lean/` | Lean 4 proofs with no Mathlib: every ontology noun has one category, serves links form no cycle, every choice reaches a telos, the communities nest, every source of a value but stated starts proposed; CI builds them; see `lean/README.md` |
| `docs/design/` | designs; `HEADLESS_REFLECT_2026-11-10.md` is the plan for one core with two hosts, `REFLECT_FRAMEWORK_2026-11-19.md` the plan for Reflect as a framework for a life, with Aristotle's Categories as its ontology, and `REFLECT_QUOTES_2026-11-20.md` the design for claims and beliefs as tags on quotes |
| `thrivesight_core/` | the hackathon reasoning engine (Python) |
| `api/` | the hackathon Django REST backend |

### Build and test

```bash
cd web && pnpm install
pnpm test        # the core's, the CLI's and the web app's tests (the Neon ones only with DATABASE_URL)
pnpm typecheck   # type-checks them
cd ../lean && lake build   # the Lean proofs
```

Pull requests are welcome. The author maintains Reflect alone, so reviews may take time.

## History

Reflect began as a prototype built for the Data Portability Hackathon, sponsored by the AI Collective, AITX, the Data Transfer Initiative and the University of Texas at Austin School of Law and Innovation. The Python reasoning engine in `thrivesight_core/` and the Django backend in `api/` are from that prototype; the package keeps its old name for import stability. The prototype was built with [@clarkmyfancy](https://github.com/clarkmyfancy), who kept the ThriveSight name and its front end, which are not part of this repository.

## License

[Apache License 2.0](./LICENSE). Author: Torian Johnson ([@tchjhnsn](https://github.com/tchjhnsn)).
