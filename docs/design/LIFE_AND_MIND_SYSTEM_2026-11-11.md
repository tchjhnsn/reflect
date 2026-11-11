# Life and mind: system design

Status: design, not built. This runs `LIFE_AND_MIND_DOMAIN_2026-11-11.md`. It changes the vault, the core and the web app, which becomes the person's blog. It assumes the domain's open questions are answered as proposed there: one note per idea that grows, and the works library in its own project.

## What stays

The life layer is already built and does not change: `life/` with passages and `state.json`, `needs/`, `learn/`, `belonging/`, `line/`, `household/`, `community/`, `polis/`, `legacy/` (`packages/core/src/lifetime.ts`). Ideas are already kept in `ideas/<slug>.md` as dated lines with `from`, `maybe from` and recommendations (`packages/core/src/ideas.ts`).

## Vault layout

```
<vault>/
  polis/  community/  household/        life layer: real state, memberships (dated, plural)
  life/  needs/  learn/  belonging/  line/  legacy/
  telos/  intents/                      the bridge: ideals, and intents that close a gap
  ideas/<slug>.md                       mind layer: one note per idea, from one line up
  ideas/<slug>/research.md              when it becomes an inquiry: question, plan, sources
  ideas/<slug>/drafts/<n>.md            drafts of the work
  published/<slug>@<sha12>.md           approved copies only, what the blog shows
  published/_log.jsonl                  every approval and withdrawal, hash-chained
```

`published/` is written only by `reflect publish`. A hand edit there fails `reflect check`.

## An idea note

The head is rebuilt from the lines, as now. The body is still the person's own. New lines join the existing grammar, each dated and never rewritten:

```
2026-11-11 stated Children learn conflict from how their parents repair it.
2026-11-11 about line, household
2026-11-12 state developing
2026-11-12 mode inquiry
2026-11-12 from [[learn/books/seven-principles]] the repair attempts chapter
2026-11-20 adopted from "Ephesians 6:4" <https://www.esv.org/Ephesians+6/> with changes
2026-12-01 merged into [[ideas/repair-before-rules]]
```

| Line | Meaning |
|---|---|
| `stated` | The person's words. The first one makes it a micro idea. |
| `about <areas>` | Areas of life, from the vault's folders. |
| `state <s>` | captured, developing, parked, merged or retired. A note with no state line is captured. |
| `mode inquiry \| venture` | Can appear twice for both. |
| `origin founded \| inherited` | Venture only. Inherited takes `from <predecessor>` and `learned by <how>`. |
| `adopted from <target> [with changes]` | Cites the source passage. |
| `merged into <idea>` | The target idea gets a `stated` line copied with `(merged from [[ideas/x]])`. |
| `retired <reason>` | Never deleted. |

The fold that already decides inputs (`foldIdea`) also decides state and mode: the latest line of each kind wins. `ideaProblems` gains checks for these:

- a state outside the five;
- a `merged into` that points at itself, at a missing note, or forms a cycle;
- `origin` on an idea that is not a venture;
- `about` naming an area that is not a vault folder.

## Merging

`reflect idea "<words>"` compares the new line with existing ideas before writing. It uses the same index `index.md` is built from and simple word overlap; no model is needed. If one looks the same, it says so and offers to restate that idea instead of starting a new one. The person decides. A person's own AI can do the same through MCP with a `find_idea` tool.

## Research

`ideas/<slug>/research.md` holds:

- the question;
- a plan (what would settle it);
- sources, each a vault note or a work in the shared library;
- findings as dated lines.

`reflect idea research <slug>` creates it from the latest statement and moves the idea to inquiry. `reflect check` reports an inquiry with no research file, and a research file with no sources after 30 days.

## Publishing

`reflect publish <slug> [--micro | --work <draft>]` checks the publication rules first. The person then reads the copy and confirms it by typing the slug.

Rules checked before the person is asked:

- The copy comes from `ideas/`. A journal entry can appear only as a quote that the person marked.
- It names no other person except people who appear in the person's own published works or in the shared library.
- Every adopted passage cites its source.
- There are no health details or exact money figures, the same rule the wiki uses.

On confirmation:

1. It writes `published/<slug>@<sha12>.md`, where the hash is of the approved text.
2. It appends an `approved` line to `published/_log.jsonl`.
3. It adds a `published <sha12>` line to the idea.

`reflect withdraw <slug>` appends a `withdrawn` line. The file stays, and the blog stops showing it.

## The blog (the web app)

The web app becomes the person's public site. It reads only `published/` and `_log.jsonl`. It shows:

- a stream of micro posts;
- a library of works, grouped by area of life;
- the profile, built from published works only.

The private app pages (journal, Me, Learn, Ideas) move behind sign-in as the workspace, or are dropped in favor of the vault and the CLI. That choice is still open.

For a vault in a git repository, the GitHub Action builds the static blog from `published/` on push. For the hosted app, the same files are kept in Postgres in tables shaped like them. That second path is not built.

## Shared works library

This is its own project, the way NTS is, so a person's vault never holds the canon. It is shaped like this:

```
works/<id>.md          title, authors, year, areas, a short note
relations.jsonl        {from, to, relation, scope?, by, on}
clusters.json          named clusters of works, kept coarse
```

The relations are depends on, extends, complements and opposes. Reflect reads the library by pinned version and never writes to it. A person's idea links to a work as `[[works/politics]]`. A published work by a person can be added to the library as a work, with the person's approval, so others can adopt from it.

## Reconcile

Intents already exist as a folder. An intent gains:

- `serves <idea or telos>`;
- `closes <region>`, the life region it means to move.

When a passage moves that region, `reflect check` reports which intents it may have satisfied, and the person marks them. Nothing is marked automatically.

## Build order

Each step is its own pull request.

1. Idea state, mode, about, origin, adopted and merged lines, with the fold and the checks.
2. Merge suggestion on `reflect idea`, and `find_idea` over MCP.
3. Research files.
4. `reflect publish` and `withdraw`, with `published/` and its log.
5. The blog from `published/`, built by the Action, plus the app's public pages.
6. The shared works library as its own repository, and links into it.
7. Intents that serve ideas and close regions.

## Not built

All of the above.
