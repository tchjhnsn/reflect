// Learn's shelves: the 21 main classes of the Library of Congress Classification (public domain; Dewey is
// not used because OCLC owns it). A book goes on the shelf its catalog's call numbers point to most often;
// a subject goes on the shelf most of its books are on. A book with no call number, or a subject with no
// shelved book, is not yet shelved: nothing is guessed. Only filled shelves show unless the person asks to
// browse all 21. Source: Library of Congress, "Library of Congress Classification Outline"
// (loc.gov/catdir/cpso/lcco).

export const LCC_CLASSES = [
  { id: "A", name: "General works" },
  { id: "B", name: "Philosophy, psychology, religion" },
  { id: "C", name: "Auxiliary sciences of history" },
  { id: "D", name: "World history" },
  { id: "E", name: "History of the Americas" },
  { id: "F", name: "Local history of the Americas" },
  { id: "G", name: "Geography, anthropology, recreation" },
  { id: "H", name: "Social sciences" },
  { id: "J", name: "Political science" },
  { id: "K", name: "Law" },
  { id: "L", name: "Education" },
  { id: "M", name: "Music" },
  { id: "N", name: "Fine arts" },
  { id: "P", name: "Language and literature" },
  { id: "Q", name: "Science" },
  { id: "R", name: "Medicine" },
  { id: "S", name: "Agriculture" },
  { id: "T", name: "Technology" },
  { id: "U", name: "Military science" },
  { id: "V", name: "Naval science" },
  { id: "Z", name: "Bibliography, library science" },
] as const;
export type LccClass = (typeof LCC_CLASSES)[number]["id"];
const IDS = new Set<string>(LCC_CLASSES.map((c) => c.id));

/** The main class of one call number, in either form: "BS2625.3 .M6" or Open Library's "BS-2625.30000000". */
export function classOf(callNumber: string): LccClass | null {
  const c = callNumber.trim().toUpperCase()[0];
  return c && IDS.has(c) ? (c as LccClass) : null;
}

/** The class most of the call numbers point to; a tie goes to the one listed first. */
export function mainClass(callNumbers: readonly string[] | undefined): LccClass | null {
  const count = new Map<LccClass, number>();
  let best: LccClass | null = null;
  for (const n of callNumbers ?? []) {
    const c = classOf(n);
    if (!c) continue;
    const k = (count.get(c) ?? 0) + 1;
    count.set(c, k);
    if (!best || k > count.get(best)!) best = c;
  }
  return best;
}

export type ShelfBook = { id: string; title: string; lcc?: readonly string[]; subjects: readonly string[] };
export type ShelfSubject = { id: string; name: string };
export type Shelf<B, S> = { id: LccClass; name: string; books: B[]; subjects: S[] };
export type Shelves<B, S> = { shelves: Shelf<B, S>[]; unshelved: { books: B[]; subjects: S[] } };

/**
 * Books and subjects on their shelves. `all` keeps every class, empty ones included, in classification
 * order; otherwise only filled shelves are kept.
 */
export function shelve<B extends ShelfBook, S extends ShelfSubject>(books: readonly B[], subjects: readonly S[], all = false): Shelves<B, S> {
  const byClass = new Map<LccClass, { books: B[]; subjects: S[] }>(LCC_CLASSES.map((c) => [c.id, { books: [], subjects: [] }]));
  const unshelved = { books: [] as B[], subjects: [] as S[] };
  const votes = new Map<string, LccClass[]>();
  for (const b of books) {
    const c = mainClass(b.lcc);
    if (!c) {
      unshelved.books.push(b);
      continue;
    }
    byClass.get(c)!.books.push(b);
    for (const s of b.subjects) votes.set(s, [...(votes.get(s) ?? []), c]);
  }
  for (const s of subjects) {
    const c = mainClass(votes.get(s.id));
    (c ? byClass.get(c)!.subjects : unshelved.subjects).push(s);
  }
  const shelves = LCC_CLASSES.map((c) => ({ id: c.id, name: c.name, ...byClass.get(c.id)! })).filter((s) => all || s.books.length || s.subjects.length);
  return { shelves, unshelved };
}
