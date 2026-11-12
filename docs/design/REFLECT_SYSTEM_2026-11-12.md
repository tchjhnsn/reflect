# Reflect: system design

Status: system design in progress, nothing built. Started 2026-11-12, after the domain design (`REFLECT_DOMAIN_2026-11-12.md`) was settled. Each point is marked **Decided** (Torian said it) or **Proposed** (Claude suggested it, not yet confirmed).

Order, by what depends on what: (1) where records live, (2) record layout, (3) the state machine, (4) standards and connections, (5) what leaves (export, counsel preparation, publishing).

## 1. Where records live

**Decided.**
- Hosted is the record: the website holds every vault (private, joint, household, village), and any vault can be exported as a plain file system at any time.
- Records sync both ways between hosted and local, and the same holds for a mobile app if one is made.
- The core keeps running on plain files, so a local copy is a working vault, not only a backup.

**Proposed.**
- Every change is an event: dated, with who made it and on which device, added to the item's history and never edited. An item's current state is computed from its events. This is the domain's "history only added to, never overwritten" carried into storage.
- Sync exchanges the events each side is missing. Two devices adding events never conflict; only two edits to the same field of the same item can, and those are kept side by side for the person to choose, never silently dropped.
- Local and mobile copies work offline and sync when back online.
- Shared vaults sync to each member's devices; a member sees only the vaults they belong to.
