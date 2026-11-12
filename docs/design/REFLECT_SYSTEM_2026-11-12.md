# Reflect: system design

Status: system design in progress, nothing built. Started 2026-11-12, after the domain design (`REFLECT_DOMAIN_2026-11-12.md`) was settled. Each point is marked **Decided** (Torian said it) or **Proposed** (Claude suggested it, not yet confirmed).

Order, by what depends on what: (1) where records live, (2) record layout, (3) the state machine, (4) standards and connections, (5) what leaves (export, counsel preparation, publishing).

## 1. Where records live

**Decided.**
- Hosted is the record: the website holds every vault (private, joint, household, village), and any vault can be exported as a plain file system at any time.
- Records sync both ways between hosted and local, and the same holds for a mobile app if one is made.
- The core keeps running on plain files, so a local copy is a working vault, not only a backup.
- Every change is an event: dated, with who made it and on which device, added to the item's history and never edited. An item's current state is computed from its events. Sync exchanges the events each side is missing.
- When two devices edit the same field of the same item before syncing, Reflect keeps both versions and asks the person which stands. Nothing is silently dropped.

**Proposed.**
- Events carry the domain's "history only added to, never overwritten" into storage; two devices adding events never conflict, only edits to the same field can.
- Local and mobile copies work offline and sync when back online.
- Shared vaults sync to each member's devices; a member sees only the vaults they belong to.

## 2. Record layout

**Decided.** A hand edit to a note (in any editor, such as Obsidian) becomes an event on the next sync, so hand edits count. Notes are not read-only views. (Reflect's GitHub Action already records hand-written changes this way.)

**Proposed.** Build on the vault Reflect already writes (packages/core: life/, needs/, learn/, ideas/, work/, belonging/, line/, journal/), which already keeps notes as markdown with front matter and dated history lines, and rebuilds `life/state.json` from `passages.jsonl`.
- Each vault is its own folder with its own `reflect.json` naming its kind (private, joint, household, village) and members. Locally: one folder per vault the person belongs to.
- Inside a vault, folders follow domains and subdomains (the existing folders, plus ones the domain design adds).
- An item is a markdown note: front matter holds its kind, holders with citations, origins, relations and computed state; the body holds the person's words. Beside it, `<item>.events.jsonl` holds its events, append-only. The note is rebuilt from the events, the same pattern as `state.json` from `passages.jsonl`.

## 3. The state machine

**Proposed.**
- An item's state is a fold over its events, checked against the item statechart in the domain design. An event that makes a move the statechart does not allow is refused with the reason.
- State events: stated, reflected (held, revised, unsure, dropped), put under examination, examined (what is believed x how it stands), planned, scheduled, started, done, missed, examined again.
- Unstated items are computed, never stored: a catalog of expected questions per subdomain, minus the ones answered.
- A domain's or subdomain's status is computed: the count of its items in each state.
- Life stages: the person's move into a stage is an event (Reflect's existing `passages.jsonl`). Each gate's evidence is computed from items (income against the livable wage). Soft gates only order what is shown; hard gates are consent events (a partner joins the joint vault, a household admits a member, parents hand over a child's records).
