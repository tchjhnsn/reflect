# Reflect: interface design

Status: started 2026-11-12, after the system design (`REFLECT_SYSTEM_2026-11-12.md`). Nothing built. Each point is marked **Decided** (Torian said it) or **Proposed**. The working canvas is a private design artifact; this file is the record.

## 1. Home screen

**Decided.** The style follows a reference Torian chose: wordless objects on a quiet white canvas, a thin black bar, monospace labels. The design itself is original, and only the principles are taken from the reference. **Superseded in part (08:51Z):** the look is now shadcn's (section 8); the wordless objects and the quiet canvas stay.
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

**Decided (06:40Z, 06:47Z).** The plus button opens a chat with the AI, and the chat fetches from the apps rather than answering from memory. Torian pointed to Vercel's json-render with Chat SDK agents (vercel.com/changelog/json-render-and-chat-sdk-agents) as the way to do it. **Decided (06:55Z):** it is used inside Reflect's own chat only, for rendering app cards in the conversation; it is not used to post to Slack, Teams or any other outside channel.

**Proposed.**
- Each app exposes tools to the chat: reads (what is in Learn, what the budget says) and proposals (draft a curriculum, add a budget line). The chat calls the tools; it holds no copy of the apps' data.
- Each app also publishes a small catalog of components (a curriculum card, a budget table, an item with its holders). The model can only render what is in the catalog, and json-render checks every spec against it, so a card shows the app's real records in the app's own shape.
- A proposal renders as a card with Accept, the one tap that lets it change anything. Nothing the AI guesses is saved without that tap; facts about the person are never inferred.
- The conversation is kept in the chat history (09:22Z), not in the journal; only accepted proposals become items.
- A card can open its app ("open in Learn"), so the work itself stays in the tiles.

**Template (08:08Z).** Torian asked for a Vercel template for the chat. Of Vercel's Chatbot template (Next.js, AI SDK, Auth.js, Drizzle, Blob) and json-render's own chat example (vercel-labs/json-render, `examples/chat`: AI SDK `useChat`, tools first, cards streamed as json-render spec parts), the chat follows the json-render example, since it already uses json-render and needs nothing that clashes with Reflect's Better Auth and Neon. One difference stays on purpose: in the example the model writes the spec; here it never does.

**Built (Learn only).** The model does not render specs itself. It talks and calls tools naming what to show (shelves, claims, a library search, a library class, or a claim to propose), streamed with the AI SDK. Reflect fetches each item from Learn and Open Library and builds the json-render spec from those records, so a card can only show what the app holds, and the model never sees more of the person than titles and names. Library books carry Add to my shelves; a proposed claim carries Accept. The person's own stance is the stance; they are never listed as a holder. Not built: cards do not open their app, and only Learn answers.

## 6. Learn catalogs what a person knows

**Decided (07:04Z).** Learn does not start with proving that something was learned. It is a tool for a person to categorize and catalog what they know, what they believe and what they want to learn, among other things. The wording is not settled; Torian was describing the idea, not naming it.

**Decided (07:08Z).** Learn keeps two kinds of entry, linked: subjects (electrical wiring, the book of Romans) and claims (marriage is a covenant). A subject can hold the claims the person holds about it. A claim names who holds it, with a citation, as items do.

**Decided (07:11Z).** Rejection is included, and Learn uses established terms, not coined ones.

**Proposed: the standards.**
- Claims take the three doxastic attitudes of epistemology: belief, disbelief and suspension of judgment (Stanford Encyclopedia of Philosophy, "Belief"; Friedman, "Suspended Judgment", 2013). A claim is stored in the shape of schema.org `Claim`.
- What a person does with a subject is recorded with Experience API (xAPI) verbs from the ADL vocabulary, such as registered, experienced, progressed, completed and mastered. A subject's state is read from its latest verb; there is no separate status field.
- Kinds of knowing, offered for sorting and never required: Ryle's knowing that and knowing how, and Aristotle's episteme (knowing why), techne (skill), phronesis (practical wisdom) and doxa (opinion).
- Sources are typed as schema.org creative works (`Book`, `VideoObject`, `Course`) and cited in CSL, the citation format reference managers use.
- Each term goes through the term engine before it is used in code.

## 7. Learn's shelves

**Decided (07:17Z).**
- Each subject carries a Wikidata identifier (CC0; labels in every language; maps to library systems).
- Subjects are arranged on shelves by the Library of Congress Classification's 21 main classes (public domain). Dewey is not used (OCLC owns it).
- Claims sit under the subject they are about.
- Only shelves with something on them show; a quiet "browse all" opens the rest.

## 8. Built so far

**Decided (08:00Z).** Torian: the old pages clash with the new design, so they are archived (`web/archive/2026-11-12-old-app/`), not deleted. The app now holds only the new design: home, Learn and the chat. Sign-in is off on the local dev server and on Vercel previews (which sit behind the Vercel team's own sign-in), and stays on in production. Work is previewed on a dev server in the session, not by pushing.


- `/`: the home screen, MINE with 18 tiles and tones from the person's records; only Learn opens; the plus button opens the chat. The other apps and OURS are not built in this design.
- `/learn`, in the new design: MY SHELVES (Library of Congress shelves, books and subjects toned known, learning or registered, a book's side panel to set where you are with it), LIBRARY (Open Library by class or search, Add to shelves) and CLAIMS (belief, disbelief or suspension of judgment, holders cited, stances kept).
- `/chat`: the plus button's chat over Learn (section 5). **Decided (09:12Z):** each chat is kept, message by message, words only; accepted cards are already items. **Revised (09:22Z):** Torian leans to a separate chat history, with the journal holding everything else (reflecting, Examine and Plan, the actions tiles send in). Chats now show only in the chat history (`/chat/history`), read-only; underneath they are still stored with the journal's records, marked as chats, so they can move to their own store if that is decided.
- `/journal` (09:05Z): the folder icon, and the Journal and Ideas tiles. Write an entry, add to it while open, close it, export the whole journal. Not built: reflecting on entries (Reflect, Examine, Plan), and the actions each tile sends in (section 2).
- **Decided (08:48Z).** Torian prefers shadcn's look to the mono one for the chat. `/chat` is now built from Vercel's AI Elements on shadcn; the mono chat is archived. At 08:51Z he chose the shadcn look for the whole app: home, Learn, sign-in and consent now use shadcn components on its neutral theme with Inter. The mono `.nd` look (IBM Plex Mono, square corners) is archived; the shapes, tones and layout of section 1 to 4 stand.
- **Decided (09:30Z): log, journal, chat history.** Three kinds of record, kept apart. The log is what the person did, kept by the tiles, never written by them (an act is chosen and done, Aristotle's prohairesis; putting a book on the shelves is an intention and shows as one). The chat history is what they said with Reflect. The journal is what they write and think: entries, and reflecting, examining and planning, which can point at a log item. The log is a view over the apps' own records, not a copy.
- `/log` (09:40Z): Learn's acts by day. **Decided (09:44Z):** every book status change is kept (`learn_book_status`, migration 0013, append-only), so putting a book down is in the log; migrated on dev, preview and production. Not built: other tiles.
- **Reflecting (10:04Z, on Torian's "Continue").** Built from the domain design's pipeline and item statechart and system design section 3, with nothing new decided. In the journal, "Reflect on this" on an entry, "Reflect" on a log act, or "Observe something" states an item. Its page shows where it sits (Observe, Reflect, Examine, Plan, Act), only the moves its state allows, and its whole history. Reflect asks "do you still hold it?" (held, revised, unsure, dropped). Examine offers the four proposed steps in any order and settles on the two decided axes; a withdrawn belief leaves the pipeline like a dropped one (**Proposed**). Plan, then a day, start, done or missed; missed goes back to Observe only by writing why. Examine may follow Reflect or be skipped, since the statechart draws Reflected to Planned (the open question stays open). Items are events in `item_event` (migration 0014), applied on dev, then on preview and production at 10:45Z with Torian's approval.
- **Decided (18:23Z to 18:48Z): the journal is a writing editor.** Torian on the reflecting screens: "This is a form, not a journal. It's uninviting and feels like work. A journal should actually be a journal." Not a page of the day either: the journal is a container, a library of the person's works, with a Create button to start a new one, and "This essentially needs to become a writing editor." It includes a prompt store: templates for kinds of thinking. Built: `/journal` lists works, Create opens the prompt library, a work opens in a full-page editor that saves versions on its own (migration 0015). The reflecting screens are archived; the item statechart stays underneath. **Decided, not built:** beliefs a person adopts in any domain are compiled into the journal, which becomes the book of their life; importing works (Torian's message was cut off there). **Open:** what is imported, and whether the book is a live view or compiled in versions.
