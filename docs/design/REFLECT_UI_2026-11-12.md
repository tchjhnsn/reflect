# Reflect: interface design

Status: started 2026-11-12, after the system design (`REFLECT_SYSTEM_2026-11-12.md`). Nothing built. Each point is marked **Decided** (Torian said it) or **Proposed**. The working canvas is a private design artifact; this file is the record.

## 1. Home screen

**Decided.** The style follows a reference Torian chose: wordless objects on a quiet white canvas, a thin black bar, monospace labels. The design itself is original, and only the principles are taken from the reference.
- A grid of objects, one per area of life, each a solid glyph with a small label.
- A MINE / OURS toggle at the top center switches between the private vault and the shared spaces.
- An object's tone shows its state: black when something in it is in action, grey when it is only in thought, light grey when nothing is stated.
- The journal is a fixed place in the top bar, not a tile.

## 2. Tiles are apps

**Decided.** The pipeline (Observe, Reflect, Examine, Plan, Act) is not what sits behind a tile; that belongs in the journal. Each tile is its own app, where the person interacts with the thing itself. Everything the person does in a tile is sent to the journal as an action, without writing, and the person reflects over those actions later in the journal. Tiles carry Act and Observe; the journal carries Reflect, Examine and Plan.

**Proposed.** One app is built well at a time; the others stay plain until their turn.

## 3. Which tiles

**Decided.** A tile is one app, so two areas merge when the person does the same kinds of things in both.

MINE (9): Learn; Work (with income, unpaid work counts); Body (health, fitness, rest, sleep); Home (food, water, shelter, transport, safety); Friends; Mate (its own private journal); Kin; Polis (with standing); Legacy. Ideas live in the journal.

OURS (4): Marriage; Household (with home and budget); Children; Village.

They follow Aristotle's three kinds of goods: of the body (Body, Home), external (Work, Friends, Mate, Kin, Polis, Legacy and the shared spaces) and of the soul (Learn).

## 4. Learn, the first app

**Decided.** Learn comes first: curriculum built from YouTube, books, courses and other things a person might want to learn. The person says what they want to learn; Reflect drafts a curriculum and the person edits it.

**Proposed.** Watching happens inside Reflect, with videos embedded, so that it can count as an action: YouTube does not give apps a person's watch history (to be checked against YouTube's current terms before building). Credentials earned in Learn feed Work, per the domain design.
