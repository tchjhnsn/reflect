# reflect CLI

The command line over one person's instance in a folder, usually a private git repository the person owns. Node 22.18 or later runs it as is: `node packages/cli/reflect.ts <command>`. The plan is `docs/design/HEADLESS_REFLECT_2026-11-10.md`.

| Command | Does |
|---|---|
| `init --name <name>` | Starts a vault any editor can open (Obsidian, VS Code), laid out as the Ventures of a Lifetime: `life/`, `needs/`, `learn/`, `work/` (each venture a Naive venture, left to Naive's own checks), `belonging/` (`friends/`, `groups/`, `standing/`), `line/`, `household/` (`members/`, `home/`, `money/`), `community/`, `polis/` and `legacy/` (`wishes.md`, `bucket-list.md`, `letters/`), each with a README that explains it. Beside them: `reflect.json` (`"layout": "vault"`), `memory.jsonl`, `person.json`, `journal/`, `reflections/`, `concepts/`, `documents/`, `telos/`, `intents/`, a `.gitignore` for Obsidian's workspace files, and `.github/workflows/reflect.yml`, which runs [the Action](../../action/action.yml). Refuses a folder that already is one. An instance made before the vault (records under `people/<id>/`) still works. |
| `upgrade` | Moves a vault made on 2026-11-10 (`ventures/household`, `work`, `community`, `friendship` and `library/`) onto the lifetime folders (`friendship` goes to `belonging/friends`, `library` to `learn`), never over a file already there, adopts the moved files in memory, and adds the missing folder guides. |
| `need <need> <how> [--provider <who>] [--runway 30d] [--note <text>]` | Keeps how one of the eight needs (food, water, shelter, health, transport, income, knowledge, safety) is met now: `self-provided`, `stocked` (with a runway), `supplied` or `cared-for`, by `self`, `market`, `nature`, `polis` or `kin`. Writes `needs/<need>.md`; each change adds a line and earlier ones stay. |
| `passage <region> <state> [--on YYYY-MM[-DD]] [--note <text>]` | Keeps a move in your life, in one of the statechart's regions: `provide` (learning, working), `belonging` (affiliating, gaining-status) or `line` (seeking-a-mate, pair-bond, raising-children, caring-for-kin). Appends to `life/passages.jsonl` and rebuilds `life/state.json`, the latest in each region by when it happened. Only the person keeps a passage. |
| `legacy [prompt] [text]` | With no prompt, offers one for what you want to survive you (after the Dignity Therapy questions and the ethical will: your life story, what you are proud of, what you want known, what you learned, your hopes for the people you love, how you want to be remembered, and more). With a prompt and text (argument or stdin), keeps it as an open journal entry with `legacy: true` and the prompt in its front matter. |
| `spark [--minutes 1\|5\|10\|15 \| --sentence]` | Object writing (Pat Pattison): prints a concrete noun, how long to write (5 minutes unless told), and to stay with the senses. |
| `spark <noun> [text] [--minutes N \| --sentence] [--date YYYY-MM-DD]` | Keeps what was written on that noun, from the argument or stdin, as a closed entry with `spark` and `concept: "[[noun]]"` in its front matter, and makes `concepts/<noun>.md` the first time, so Obsidian's backlinks list every entry on the noun. |
| `write [text] [--date YYYY-MM-DD]` | Writes an entry, from the argument or stdin, and prints its id. |
| `documents` | The custody checklist for official documents, by group (identity, education, household, money, health, legal): for each one not yet checked, how it is normally kept today and what we recommend (keep the paper safe, store it digitally, or connect the service that holds it), and which connections hold it. The documents themselves need not be in the vault. |
| `documents <type> <custody> [--where <text>] [--connection <id>]` | Checks one off: `original`, `digital-copy`, `connection` (with a connection that holds it), `missing` or `not-applicable`. Writes `documents/<type>.md`: front matter for the current state and one line per check-off, so earlier ones stay. |
| `connect <id> --name <name> --kind vault\|issuer\|portal --holds <type,type>` | Records a service that holds documents for you, such as a secure vault or a school's credential service, in `connections/<id>.json`. No credentials are stored, and Reflect never copies a document from it. Running it again adds to what it holds. |
| `add <entry> [text]`, `close <entry>` | Adds to an open entry, or closes it. |
| `list` | The entries, newest first. |
| `read [entry] [--pending]` | Asks the reader (Jev, with gpt-oss for self-terms) for its guesses and writes `reflections/<entry>.draft.json`. A draft is never a record and is not in memory. `--pending` reads every entry with no labeling and no draft. Needs `TYPESAFEAI_API_KEY` or `AI_GATEWAY_API_KEY` (and `OLLAMA_API_KEY` to find self-terms without a gateway key). |
| `keep [entry] [--all]` | Keeps a draft as the person left it: `chosen` is the elements they say were part of the entry, and each self-term's `basis` and `keep` are theirs. Each decision is recorded against the reader's guess (confirmed, added, rejected, corrected), then the draft is removed. |
| `export [--out <file>] [--moving-to <where>]` | The whole journal as a bundle (`packages/core/src/bundle.ts`), with times and decisions. `--moving-to` then sets `moved_to` in `reflect.json`, so writes here are refused. |
| `import <bundle>` | Imports a bundle into this instance's empty journal. |
| `check [--adopt]` | Fails on a broken `memory.jsonl` chain, a `reflect.json` that is not an instance config, or a grant that names `journal`, `entries`, `reflections` or `_private`. Lists records written outside reflect (any file in the vault's folders, never `.obsidian/` or `.github/`; by hand, say in Obsidian or the GitHub app); `--adopt` records them in memory as they are. |

Every command takes `--dir <instance>` (default `.`). Writes are refused once `reflect.json` has `moved_to`.

`reflect.test.ts` tests it; `pnpm test` in `web/` runs it. Not built yet: `encrypt: "age"`. A bundle holds the journal only; Me's facts and holds are not in it yet.

## In a GitHub repository

Push the instance to a **private** repository and add `TYPESAFEAI_API_KEY` (or `AI_GATEWAY_API_KEY`) and `OLLAMA_API_KEY` as repository secrets. On each push to `main` the workflow runs `action/action.yml`:

1. It refuses to run in a public repository.
2. `check`.
3. `keep --all` for drafts that reached `main` (a merged pull request), then `check --adopt` for entries written by hand, committed to `main`.
4. `read --pending` onto the branch `reflect/drafts`, and a pull request "Reflect on new entries" if none is open. The pull request is the hold gate: edit the drafts in it and merge to keep them, or close it to keep nothing.
