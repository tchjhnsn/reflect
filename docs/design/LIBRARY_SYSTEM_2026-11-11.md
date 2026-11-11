# The library: system design

Status: design, not built. This runs `LIBRARY_DOMAIN_2026-11-11.md`. It assumes the library is its own repository (open question 3 in the life and mind domain design), read by Reflect at a pinned version, the way Mentura reads Naive.

## Repository layout

```
library/
  works/<wikidata-id>.md     title, authors, year, areas, editions by ISBN, public domain or not
  excerpts/<cts-or-osis>.md  one passage by CTS URN or OSIS reference: text (public domain) or a short quote
                             and a link, areas, a one-line gloss
  schemes/<scheme>.json      SKOS concept schemes: areas, forms, fields, topics, instruments of thought
  forms/<id>.md              one adopted form: its LCGFT term and matches, its definition, and a
                             structure standard when one exists
  relations.jsonl            {from, to, relation, scope?, by, on}
  clusters.json              named clusters of works, kept coarse
  encyclopedia/<concept>.md  the library's own entries: concept, works and excerpts that treat it
```

A model may draft relations, clusters and encyclopedia entries. Each one lands only after a person on our side accepts it, using the same hold-gate pattern as Naive.

## An adopted form

```
---
id: academic-theses
lcgft: Academic theses
matches: { coar: c_46ec, schema: Thesis, wikidata: <id> }
structure: ISO 7144
---

The definition, quoted from LCGFT.

## Parts (from ISO 7144)
...
```

`structure` is present only when a standard says how the form is built. `reflect idea draft --form` uses those parts as headings. With no structure standard, the draft starts empty under the form's definition.

## A scheme

A SKOS concept scheme in JSON-LD: each concept has an id, a preferred label, `broader` and `narrower`, and `exactMatch` or `closeMatch` to the standard it follows. One loader reads every scheme, so areas of life, forms, fields of study and instruments of thought are handled the same way.

## How Reflect uses it

- `reflect forms [--standard lcgft|coar|aat|fast]` lists the adopted forms, by the standard the person chooses.
- `reflect idea draft <slug> --form <id>` creates `ideas/<slug>/drafts/<n>.md`, with the parts of the form's structure standard as headings when it has one. It adds a `form <id>` line to the idea.
- `reflect publish` checks a draft against its form's structure standard when it has one, for example a research paper's sections. A check the code cannot make is shown to the person as a question. It is never decided by a model.
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

1. The library repository with schemes, works and excerpts, starting with public-domain works for household, line and polis, and the forms adopted from LCGFT with their matches.
2. `reflect forms` and `reflect idea draft --form`.
3. `holds` lines, `reflect adopt` and `reflect encyclopedia`.
4. Form checks in `reflect publish`.
5. Relations, clusters and the library's encyclopedia.

## Not built

All of the above.
