# reflect CLI

The command line over one person's instance in a folder, usually a private git repository the person owns. Node 22.18 or later runs it as is: `node packages/cli/reflect.ts <command>`. The plan is `docs/design/HEADLESS_REFLECT_2026-11-10.md`.

| Command | Does |
|---|---|
| `init --name <name>` | Starts an instance: `reflect.json` (who, which host), `memory.jsonl`, and `people/<id>/` with `person.json`, `entries/`, `reflections/`, `intents/`, `telos/`. Refuses a folder that already is one. |
| `write [text] [--date YYYY-MM-DD]` | Writes an entry, from the argument or stdin, and prints its id. |
| `add <entry> [text]`, `close <entry>` | Adds to an open entry, or closes it. |
| `list` | The entries, newest first. |
| `check [--adopt]` | Fails on a broken `memory.jsonl` chain, a `reflect.json` that is not an instance config, or a grant that names `entries`, `reflections` or `_private`. Lists records written outside reflect (by hand, say in the GitHub app); `--adopt` records them in memory as they are. |

Every command takes `--dir <instance>` (default `.`). Writes are refused once `reflect.json` has `moved_to`.

`reflect.test.ts` tests it; `pnpm test` in `web/` runs it. Not built yet: `read` and `keep` (the reader), the GitHub Action, and `export` and `import`.
