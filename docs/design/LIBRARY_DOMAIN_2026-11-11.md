# The library: domain design

Status: design, not built. Proposed 2026-11-11 from Torian's direction in the Reflect thread. It goes with `LIFE_AND_MIND_DOMAIN_2026-11-11.md`, which covers a person's own ideas. This document covers what they read, adopt and write *into*. The system design is `LIBRARY_SYSTEM_2026-11-11.md`. Forms are adopted from standards, not invented (Torian, 2026-11-11).

## What the library holds

The library is shared, not any one person's. It holds three things:

1. **Works.** Standard written works by area of life, with excerpts: the passages people actually cite and adopt.
2. **Forms.** The kinds of work a person can write (an essay, a thesis, a research paper, a biography, an encyclopedia entry and so on), adopted from standards such as the Library of Congress Genre/Form Terms.
3. **Relations.** How works relate to each other: depends on, extends, complements, opposes, each with an optional scope.

A person's vault holds their own ideas and links into the library. It never holds a copy of the canon.

## Works by area of life

The areas are the vault's life regions. A first list of standard works, from known canon and not yet verified:

| Area | Works |
|---|---|
| Life | Aristotle, *Nicomachean Ethics*; Marcus Aurelius, *Meditations*; Augustine, *Confessions*; Frankl, *Man's Search for Meaning* |
| Needs | Hesiod, *Works and Days*; Maslow 1943; Sen, *Development as Freedom*; Nussbaum, *Creating Capabilities* |
| Learn | Plato, *Meno* and *Republic* VII; Hugh of St Victor, *Didascalicon*; Dewey, *Democracy and Education*; Adler, *How to Read a Book* |
| Work | Xenophon, *Oeconomicus*; the *Rule* of Benedict; Smith, *Wealth of Nations*; Weber, *The Protestant Ethic*; Schumpeter; Drucker, *Innovation and Entrepreneurship* |
| Belonging | *Nicomachean Ethics* VIII and IX; Cicero, *De Amicitia*; Aelred, *Spiritual Friendship*; Putnam, *Bowling Alone* |
| Line | Genesis 2; Song of Songs; Plato, *Symposium*; Augustine, *On the Good of Marriage*; Lewis, *The Four Loves*; Gottman |
| Household | *Politics* I; pseudo-Aristotle, *Economics*; Ephesians 5 and 6, Colossians 3, 1 Peter 2 and 3; Proverbs 31; Berry |
| Community | Jacobs, *Death and Life of Great American Cities*; Tönnies; Nisbet, *The Quest for Community*; Ostrom, *Governing the Commons* |
| Polis | Plato, *Republic* and *Laws*; *Politics*; Augustine, *City of God*; Hobbes; Locke; *The Federalist*; Tocqueville |
| Legacy | Ecclesiastes; Cicero, *On Old Age*; Seneca, *On the Shortness of Life*; the ars moriendi; Becker, *The Denial of Death* |

Aristotle covers almost every area. That is why mapping his works is the first research project.

An **excerpt** is a passage of a work, with its standard reference (Bekker number, chapter and verse, page). Excerpts are what a person adopts: one verse of a household code, not all of Ephesians.

## Structure first: schemes and components

Torian, 2026-11-11: structure is the most important part of what we are building. Every category system is a **scheme**, written as a W3C SKOS concept scheme. A scheme gives each concept a stable id, links to broader and narrower concepts, and a match to the standard it follows. Works and passages are **components**, called by their standard codes.

| Scheme | Standard it follows |
|---|---|
| Areas of life | none; ours, citing Aristotle, *Politics* I.2 |
| Forms of work | LCGFT, with FAST, COAR, Getty AAT, schema.org and Wikidata matches beside it |
| Fields of study | UNESCO ISCED-F 2013 (the backbone), US CIP 2020 for detail, OECD Fields of Science for research |
| Topics | LCSH and Wikidata |
| Instruments of thought | LCC BC (logic) and BF (psychology), ISCED-F philosophy and psychology; frameworks filed from Moseley et al. 2005 |

Works and passages are called by code:

- **A work** follows the IFLA Library Reference Model's four levels: work, expression (a translation), manifestation (an edition) and item. A work's id is its Wikidata id. An edition hangs under it by ISBN. The International Standard Text Code, ISO 21047, was meant to identify works, but its agency closed in 2017 and ISO withdrew it in 2021, so it is not used.
- **A passage** is called by CTS URN for classical texts (*Politics* is `urn:cts:greekLit:tlg0086.tlg035`, plus a Bekker reference) and by OSIS reference for the Bible (`Eph.6.4`). Any edition resolves it.

## Forms

A **form** is a kind of work: what the work is, not what it is about. We adopt forms from standards and do not invent our own. The default is the Library of Congress Genre/Form Terms (LCGFT), which hold about 390 terms for literary works and more for other materials. Other standards sit beside it, so the library or a person can choose which one a form follows:

| Standard | Kept by | Good for |
|---|---|---|
| LCGFT | Library of Congress | the fullest list of literary and document forms; the default |
| FAST form/genre | OCLC | a simpler set derived from the LoC headings |
| COAR Resource Types | Confederation of Open Access Repositories | research outputs: bachelor, master and doctoral theses, research and review articles, reports |
| Getty AAT | Getty Research Institute | document and object types; visual and archival work such as a documentary |
| schema.org CreativeWork | schema.org | how the blog marks up what it publishes |
| Wikidata | community | the links between all of the above |

LCGFT names and defines a form but does not say how it is built. Where a structure standard exists, it is adopted with the form: ISO 7144 for presenting a thesis, ANSI/NISO Z39.18 for scientific and technical reports, and the IMRaD convention for research papers. A form with no structure standard, such as an essay or a memoir, carries only its definition. The library does not make one up. (The ISO and NISO references are from memory and are checked before the library uses them.)

Forms still relate to modes. Theses, research papers and essays are where an idea in inquiry mode ends up. A house code or a creed is where adopted ideas gather. A venture goes to Naive, though it may produce a written form such as a manifesto or a charter.

## Instruments of thought

Thinking frameworks, heuristics and metacognition need a parent category. Logic is the right parent for half of them, the normative half, which says how a person ought to reason. Heuristics and metacognition are mostly descriptive: how people do think. Kahneman and Gigerenzer on heuristics, and Flavell on metacognition, belong to psychology. Library classification splits them the same way: LCC puts logic in BC and thinking and cognition in BF.

Aristotle's name for the parent is the *Organon*. His logical works were the instrument of every science, not a branch of one. The scheme is:

```
Instruments of thought
  Logic: formal, informal, dialectic
  Rhetoric
  Methods of inquiry: scientific method, historical method
  Judgment and decision: decision theory, heuristics, biases
  Metacognition: monitoring and regulating one's own thinking
```

Each branch matches its standard home. The 42 frameworks reviewed in Moseley et al., *Frameworks for Thinking* (Cambridge, 2005) are filed under these branches. Naive's decision frames belong under judgment and decision.

## Encyclopedias

There are two, and they are different things:

- **The library's encyclopedia** is ours. It is reference: an entry per concept, linking the works and excerpts that treat it, coarse and shared.
- **A person's encyclopedia** is the ideas they subscribe to. Each entry is the person's own idea or an adopted one, says where it comes from, and is organized by area of life. Historically this is closer to a **commonplace book** (a person's own collection of passages they hold to, with their notes) or a personal confession than to Britannica.

It also differs from the wiki Reflect already has:

- **The wiki** is a private map of the person's mind, written by their own AI about them.
- **The personal encyclopedia** is what the person themselves states they subscribe to, and it can be published.

## Subscribing and adopting

A person relates to an idea or excerpt in one of these ways:

| Relation | Meaning |
|---|---|
| wrote | their own idea |
| adopts | holds it as their own, citing the source |
| adopts with changes | holds a changed version, citing the source and saying what changed |
| studies | reading it, holding no view yet |
| rejects | holds the contrary, citing what they reject |

Subscribing means wrote, adopts or adopts with changes. Those three make up the person's encyclopedia.

## Open questions

1. Which standard does each person default to when LCGFT and COAR both name a form (a doctoral thesis, for example)?
2. Should a person's encyclopedia be one work (a single form), or a view over everything they subscribe to?
3. Who may add works to the shared library: only us, or people too, with their published works?
4. How much of an excerpt can the library hold before copyright matters? Public-domain works can be held whole; others only by reference and short quotation.
5. Is "instruments of thought" the right parent, with logic as one branch beside psychology's heuristics and metacognition?
