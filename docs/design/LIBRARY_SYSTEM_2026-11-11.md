# The library: system design

Status: design, not built. This runs `LIBRARY_DOMAIN_2026-11-11.md`. It assumes the library is its own repository (open question 3 in the life and mind domain design), read by Reflect at a pinned version, the way Mentura reads Naive.

## Repository layout

```
library/
  works/<id>.md              title, authors, year, areas, edition, public domain or not
  excerpts/<work>/<ref>.md   one passage: standard reference, text (public domain) or a short quote
                             and a link, areas, a one-line gloss
  forms/<form>.md            one template per form
  relations.jsonl            {from, to, relation, scope?, by, on}
  clusters.json              named clusters of works, kept coarse
  encyclopedia/<concept>.md  the library's own entries: concept, works and excerpts that treat it
```

A model may draft relations, clusters and encyclopedia entries. Each one lands only after a person on our side accepts it, using the same hold-gate pattern as Naive.

## A form file

```
---
form: essay
answers_to: inquiry
gathers: []
length: 1,500 to 5,000 words
---

## Parts
1. The question or claim, stated in the first paragraph.
2. The argument, with each step's grounds.
3. The strongest objection, and the answer to it.
4. What follows if it holds.

## Must contain
- A thesis sentence.
- At least one cited source for each claim of fact.

## How a reader judges it
Is the claim clear, are the grounds sound, was the objection a real one.
```

`answers_to` is one of inquiry, life, commitment, reference, address or imagination. `gathers` lists the forms it can collect: a rule of life gathers house codes, a commonplace book gathers excerpts.

## How Reflect uses it

- `reflect forms` lists the forms, grouped by what they answer to.
- `reflect idea draft <slug> --form essay` creates `ideas/<slug>/drafts/<n>.md` from the template's parts. The parts become headings the person fills in. It adds a `form essay` line to the idea.
- `reflect publish` reads the form's "must contain" list and checks it. A research paper with no sources is refused, and so is a thesis with no thesis sentence. A check the code cannot make, such as "the strongest objection", is shown to the person as a question. It is never decided by a model.
- `reflect library <area>` lists the works and excerpts for an area of life.
- `reflect adopt <excerpt> [--changes "<what changed>"]` writes a new idea whose first lines are `adopted from [[library/excerpts/...]]` and the excerpt's gloss in the person's words. That idea then joins the person's encyclopedia.

## Subscribing

The idea note gains one line, the person's stance:

```
2026-11-11 holds adopts
```

The stance is one of wrote, adopts, adopts-with-changes, studies or rejects. An idea with no `holds` line and no `adopted from` line counts as wrote.

## A person's encyclopedia

This is a view, not a separate store. Every idea whose stance is wrote, adopts or adopts-with-changes, grouped by its `about` areas, each with its source. `reflect encyclopedia` prints it. Publishing it is `reflect publish --encyclopedia`: it gathers the published copies of those ideas into one work, and every idea in it must pass the gate on its own first.

## The works library in Reflect's own checks

`reflect check` flags any of these:

- a `[[library/...]]` link that does not exist at the pinned version;
- an `adopted from` with no excerpt reference;
- a draft whose `form` is unknown.

## Build order

This follows build steps 1 to 4 of the life and mind system design.

1. The library repository with works, excerpts and forms, starting with public-domain works for household, line and polis.
2. `reflect forms` and `reflect idea draft --form`.
3. `holds` lines, `reflect adopt` and `reflect encyclopedia`.
4. Form checks in `reflect publish`.
5. Relations, clusters and the library's encyclopedia.

## Not built

All of the above.
