# Reflect: interface design

Status: started 2026-11-12, after the system design (`REFLECT_SYSTEM_2026-11-12.md`). Nothing built. Each point is marked **Decided** (Torian said it) or **Proposed**. The working canvas is a private design artifact; this file is the record.

## 1. Home screen

**Decided.** The style follows a reference Torian chose: wordless objects on a quiet white canvas, a thin black bar, monospace labels. The design itself is original, and only the principles are taken from the reference.
- A grid of objects, one per area of life, each a solid glyph with a small label.
- A MINE / OURS toggle at the top center switches between the private vault and the shared spaces.
- An object's tone shows its state: black when something in it is in action, grey when it is only in thought, light grey when nothing is stated.
- The journal is the folder icon at the top right (06:29Z): it is the vault's files, and export lives inside it. The journal also has a tile (see 3).

## 2. Tiles are apps

**Decided.** The pipeline (Observe, Reflect, Examine, Plan, Act) is not what sits behind a tile; that belongs in the journal. Each tile is its own app, where the person interacts with the thing itself. Everything the person does in a tile is sent to the journal as an action, without writing, and the person reflects over those actions later in the journal. Tiles carry Act and Observe; the journal carries Reflect, Examine and Plan.

**Proposed.** One app is built well at a time; the others stay plain until their turn.

## 3. Which tiles

**Decided.** A tile is one app, so two areas merge when the person does the same kinds of things in both.

MINE (9): Learn; Work (with income, unpaid work counts); Body (health, fitness, rest, sleep); Home (food, water, shelter, transport, safety); Friends; Mate (its own private journal); Kin; Polis (with standing); Legacy. Ideas live in the journal.

OURS (4): Marriage; Household (with home and budget); Children; Village.

**Decided (06:28Z).** The screen keeps the fuller grid: the apps are condensed, the tiles are not. MINE shows 18 tiles (Learn, Work, Income, Food, Water, Shelter, Health, Fitness, Transport, Safety, Friends, Standing, Mate, Kin, Polis, Legacy, Ideas, Journal) and OURS shows 6 (Marriage, Household, Home, Budget, Children, Village). Several tiles open the same app: Income opens Work; Food, Water, Shelter, Transport and Safety open Home; Health and Fitness open Body; Standing opens Polis; Ideas and Journal open the journal; Home and Budget open Household.

The apps follow Aristotle's three kinds of goods: of the body (Body, Home), external (Work, Friends, Mate, Kin, Polis, Legacy and the shared spaces) and of the soul (Learn).

## 4. Learn, the first app

**Decided.** Learn comes first: curriculum built from YouTube, books, courses and other things a person might want to learn. The person says what they want to learn; Reflect drafts a curriculum and the person edits it.

**Proposed.** Watching happens inside Reflect, with videos embedded, so that it can count as an action: YouTube does not give apps a person's watch history (to be checked against YouTube's current terms before building). Credentials earned in Learn feed Work, per the domain design.

## 5. The plus button opens a chat

**Decided (06:40Z, 06:47Z).** The plus button opens a chat with the AI, and the chat fetches from the apps rather than answering from memory. Torian pointed to Vercel's json-render with Chat SDK agents (vercel.com/changelog/json-render-and-chat-sdk-agents) as the way to do it.

**Proposed.**
- Each app exposes tools to the chat: reads (what is in Learn, what the budget says) and proposals (draft a curriculum, add a budget line). The chat calls the tools; it holds no copy of the apps' data.
- Each app also publishes a small catalog of components (a curriculum card, a budget table, an item with its holders). The model can only render what is in the catalog, and json-render checks every spec against it, so a card shows the app's real records in the app's own shape.
- A proposal renders as a card with Accept, the one tap that lets it change anything. Nothing the AI guesses is saved without that tap; facts about the person are never inferred.
- The conversation is kept in the journal; only accepted proposals become items.
- A card can open its app ("open in Learn"), so the work itself stays in the tiles.
- Chat SDK's adapters can post the same cards to Slack or Teams. Reflect would use that only under the rules in the system design's "What leaves": nothing from a private or shared vault goes to an outside platform unless the person sends it.
