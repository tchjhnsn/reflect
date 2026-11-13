"use server";

import { revalidatePath } from "next/cache";
import { BOOK_STATUS, type BookStatus } from "@reflect/core/learning";
import { cleanEntry, workEntry } from "@reflect/core/openlibrary";
import { importGoodreads, type ImportReport } from "@/server/goodreads-import";
import { requireUser } from "@/server/session";
import { addBook, keepPlace, setBookStatus, type BookPlace } from "@/server/learn-store";
import { keepClaim, setStance } from "@/server/claims-store";
import { addExcerpt, allExcerpts, removeExcerpt, reviseExcerpt, type Anchor } from "@/server/excerpt-store";
import { importKindle, type KindleReport } from "@/server/kindle-import";
import { publishQuote, unpublishQuote } from "@/server/published-store";
import { importWork } from "@/server/work-store";
import { excerptsMarkdown, isHighlightColor } from "@reflect/core/excerpts";
import { highlightBooks } from "@/server/highlights";
import { definitionsOf, lookupWord, type Definition } from "@reflect/core/wiktionary";

const openLibrary = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, cache: "no-store" });
const status = (v: FormDataEntryValue | null): BookStatus => ((BOOK_STATUS as readonly string[]).includes(String(v)) ? (String(v) as BookStatus) : "want");

/** Put a library book on the person's shelves. The entry comes back from the page, so it is cleaned again. */
export async function addFromLibraryAction(form: FormData): Promise<void> {
  const user = await requireUser();
  let raw: unknown;
  try {
    raw = JSON.parse(String(form.get("entry") ?? ""));
  } catch {
    throw new Error("that library entry could not be read");
  }
  let clean = cleanEntry(raw);
  if (!clean) throw new Error("that library entry has no title");
  // Some lists (trending) carry no call numbers; look the work up so the book lands on its shelf.
  if (!clean.lcc?.length) {
    const full = await workEntry(clean.work, openLibrary).catch(() => null);
    if (full) clean = { ...full, ...(clean.cover ? { cover: clean.cover } : {}) };
  }
  // No revalidation: Learn's pages render fresh on each visit, and refreshing the page the book was added from
  // would reshuffle its lists under the person.
  await addBook(user.id, { title: clean.title, authors: clean.authors, ...(clean.isbn ? { isbn: clean.isbn } : {}), status: status(form.get("status")), subjects: [], catalog: clean });
}

/** Where the person is with a book: want to read, reading, read. */
export async function setBookStatusAction(form: FormData): Promise<void> {
  const user = await requireUser();
  await setBookStatus(user.id, String(form.get("book") ?? ""), status(form.get("status")));
  revalidatePath("/learn");
}

/** Keep a claim with who holds it and the person's stance. Holders come one per line: "name | citation". */
export async function keepClaimAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const holders = String(form.get("holders") ?? "")
    .split("\n")
    .map((l) => l.split("|"))
    .map(([name, ...cite]) => ({ name: (name ?? "").trim(), citation: cite.join("|").trim() }))
    .filter((h) => h.name)
    .map((h) => (h.citation ? h : { name: h.name }));
  await keepClaim(user.id, { text: String(form.get("claim") ?? ""), holders, subject: String(form.get("subject") ?? ""), stance: String(form.get("stance") ?? "") });
  revalidatePath("/learn");
}

export async function setStanceAction(form: FormData): Promise<void> {
  const user = await requireUser();
  await setStance(user.id, String(form.get("claim") ?? ""), String(form.get("stance") ?? ""));
  revalidatePath("/learn");
}

/** Bring a Goodreads export in: the browser reads the file and sends its text. */
export async function importGoodreadsAction(text: string): Promise<{ error: string } | ImportReport> {
  const user = await requireUser();
  if (new TextEncoder().encode(text).length > 5_000_000) return { error: "That file is larger than 5 MB." };
  try {
    const r = await importGoodreads(user.id, text, openLibrary);
    revalidatePath("/learn");
    return r;
  } catch (e) {
    return { error: (e as Error).message };
  }
}


/** Keep where the person is in a book on their shelves that they are reading in Reflect. Nothing is revalidated: the reader keeps its own place. */
export async function keepPlaceAction(bookId: string, place: BookPlace): Promise<void> {
  const user = await requireUser();
  await keepPlace(user.id, bookId, { chapter: Number(place?.chapter), chapters: Number(place?.chapters), at: Number(place?.at) });
}

/** Keep an excerpt from a book: a highlight in a text read in Reflect, or words typed with a page or place. Returns its id. */
export async function addExcerptAction(e: { work: string; quote: string; anchor?: unknown; place?: string; note?: string; color?: string }): Promise<string> {
  const user = await requireUser();
  const id = await addExcerpt(user.id, { work: String(e?.work ?? ""), quote: String(e?.quote ?? ""), anchor: (e?.anchor ?? null) as Anchor | null, place: e?.place ? String(e.place) : null, note: e?.note ? String(e.note) : null, color: isHighlightColor(e?.color) ? e.color : null });
  revalidatePath(`/learn/works/${String(e.work)}`);
  return id;
}

/** Remove one of the person's excerpts. */
export async function removeExcerptAction(id: string, work: string): Promise<void> {
  const user = await requireUser();
  await removeExcerpt(user.id, String(id));
  revalidatePath(`/learn/works/${String(work)}`);
}

const wiktionary = (word: string) => fetch(`https://en.wiktionary.org/w/index.php?title=${encodeURIComponent(word)}&action=raw`, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 604800 } });

/**
 * A word looked up while reading, from Wiktionary: its English senses with their labels, the word as written
 * first and then in lower case; a word that is only a form of another ("acknowledged") also brings that word's senses.
 */
export async function defineAction(selection: string): Promise<{ word: string; definitions: Definition[]; lemma?: { word: string; definitions: Definition[] } } | null> {
  await requireUser();
  const word = lookupWord(selection);
  if (!word) return null;
  const look = async (w: string) => {
    const r = await wiktionary(w).catch(() => null);
    return r?.ok ? definitionsOf(await r.text()) : [];
  };
  let found = word;
  let defs = await look(word);
  if (!defs.length && word !== word.toLowerCase()) {
    found = word.toLowerCase();
    defs = await look(found);
  }
  if (!defs.length) return null;
  const forms = defs.flatMap((d) => d.senses).filter((s) => s.formOf).map((s) => s.formOf!);
  const lemma = forms[0] && forms[0] !== found ? { word: forms[0], definitions: await look(forms[0]) } : null;
  return { word: found, definitions: defs, ...(lemma?.definitions.length ? { lemma } : {}) };
}

/** Give an excerpt a new colour or note (an empty note removes it). Returns the excerpt's new id. */
export async function reviseExcerptAction(id: string, work: string, change: { color?: string; note?: string | null }): Promise<string> {
  const user = await requireUser();
  const next = await reviseExcerpt(user.id, String(id), { ...(isHighlightColor(change?.color) ? { color: change.color } : {}), ...(change && "note" in change ? { note: change.note == null ? null : String(change.note) } : {}) });
  revalidatePath(`/learn/works/${String(work)}`);
  revalidatePath("/learn/highlights");
  return next;
}

/** Bring Kindle highlights in: the browser reads My Clippings.txt and sends its text. */
export async function importKindleAction(text: string): Promise<{ error: string } | KindleReport> {
  const user = await requireUser();
  if (new TextEncoder().encode(String(text)).length > 10_000_000) return { error: "That file is larger than 10 MB." };
  try {
    const r = await importKindle(user.id, String(text), openLibrary);
    revalidatePath("/learn/highlights");
    return r;
  } catch (e) {
    return { error: (e as Error).message };
  }
}

/** Put all the person's highlights into their journal as one Markdown work. Returns the work's id. */
export async function highlightsToJournalAction(): Promise<string> {
  const user = await requireUser();
  const md = excerptsMarkdown(await highlightBooks(user.id, await allExcerpts(user.id)));
  const id = await importWork(user.id, "Highlights.md", md);
  revalidatePath("/journal");
  return id;
}

/** Publish one quote, with its author, book and place, at its own public page. Returns its id. */
export async function publishQuoteAction(q: { quote: string; author: string; title: string; place?: string | null; work?: string | null }): Promise<string> {
  const user = await requireUser();
  const id = await publishQuote(user.id, { quote: String(q?.quote ?? ""), author: String(q?.author ?? ""), title: String(q?.title ?? ""), place: q?.place ? String(q.place) : null, work: q?.work ? String(q.work) : null });
  revalidatePath("/learn/highlights");
  return id;
}

/** Take a published quote down. */
export async function unpublishQuoteAction(id: string): Promise<void> {
  const user = await requireUser();
  await unpublishQuote(user.id, String(id));
  revalidatePath("/learn/highlights");
}
