# The mono design (archived 2026-11-12)

Reflect's first new design: wordless objects on a white canvas, a thin black bar, one monospace face
(IBM Plex Mono), square corners, black and white with greys for tone. Torian liked it and asked for it to be
kept when he chose shadcn's look for the app on 2026-11-12 (08:51Z).

## See it

- `shots/`: home, shelves, a book's panel, the library, claims and the chat, as it last looked.
- To run it whole, check out commit `f2d4ee9` on reflect#48 (the last commit with home and Learn in this
  look; the chat moved to shadcn one commit earlier, at `4fd8c82` the mono chat is still on `/chat`).

## What is here

| Path | Holds |
|---|---|
| `app/page.tsx` | The home screen (MINE and OURS, 18 tiles, tones). |
| `app/learn/page.tsx` | Learn: MY SHELVES, LIBRARY, CLAIMS, the book panel. |
| `app/chat/` | The chat (`page.tsx`, `chat.tsx`, `registry.tsx`) with the cards drawn in the mono look. |
| `app/fonts.ts` | IBM Plex Mono. |
| `app/mono.css` | The `.home`, `.nd` and `.chat` styles, cut from the old `globals.css`. |

Nothing here is routed, built or type-checked. The pages import modules by their old paths, so bringing
the look back means copying them into `src/app/`, importing `mono.css`, and fixing the paths.
