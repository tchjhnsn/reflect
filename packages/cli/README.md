# reflect CLI

The command line over one person's instance in a folder, usually a private git repository the person owns. Node 22.18 or later runs it as is: `node packages/cli/reflect.ts <command>`. The plan is `docs/design/HEADLESS_REFLECT_2026-11-10.md`.

| Command | Does |
|---|---|
| `init --name <name>` | Starts an instance: `reflect.json` (who, which host), `memory.jsonl`, `people/<id>/` with `person.json`, `entries/`, `reflections/`, `intents/`, `telos/`, and `.github/workflows/reflect.yml`, which runs [the Action](../../action/action.yml). Refuses a folder that already is one. |
| `write [text] [--date YYYY-MM-DD]` | Writes an entry, from the argument or stdin, and prints its id. |
| `add <entry> [text]`, `close <entry>` | Adds to an open entry, or closes it. |
| `list` | The entries, newest first. |
| `read [entry] [--pending]` | Asks the reader (Jev, with gpt-oss for self-terms) for its guesses and writes `people/<id>/reflections/<entry>.draft.json`. A draft is never a record and is not in memory. `--pending` reads every entry with no labeling and no draft. Needs `TYPESAFEAI_API_KEY` or `AI_GATEWAY_API_KEY` (and `OLLAMA_API_KEY` to find self-terms without a gateway key). |
| `keep [entry] [--all]` | Keeps a draft as the person left it: `chosen` is the elements they say were part of the entry, and each self-term's `basis` and `keep` are theirs. Each decision is recorded against the reader's guess (confirmed, added, rejected, corrected), then the draft is removed. |
| `check [--adopt]` | Fails on a broken `memory.jsonl` chain, a `reflect.json` that is not an instance config, or a grant that names `entries`, `reflections` or `_private`. Lists records written outside reflect (by hand, say in the GitHub app); `--adopt` records them in memory as they are. |

Every command takes `--dir <instance>` (default `.`). Writes are refused once `reflect.json` has `moved_to`.

`reflect.test.ts` tests it; `pnpm test` in `web/` runs it. Not built yet: `export` and `import`, and `encrypt: "age"`.

## In a GitHub repository

Push the instance to a **private** repository and add `TYPESAFEAI_API_KEY` (or `AI_GATEWAY_API_KEY`) and `OLLAMA_API_KEY` as repository secrets. On each push to `main` the workflow runs `action/action.yml`:

1. It refuses to run in a public repository.
2. `check`.
3. `keep --all` for drafts that reached `main` (a merged pull request), then `check --adopt` for entries written by hand, committed to `main`.
4. `read --pending` onto the branch `reflect/drafts`, and a pull request "Reflect on new entries" if none is open. The pull request is the hold gate: edit the drafts in it and merge to keep them, or close it to keep nothing.
