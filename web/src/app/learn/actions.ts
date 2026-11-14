"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { BOOK_STATUS, type BookStatus } from "@reflect/core/learning";
import { cleanEntry, workEntry } from "@reflect/core/openlibrary";
import { importGoodreads, type ImportReport } from "@/server/goodreads-import";
import { requireUser } from "@/server/session";
import { addBook, keepPlace, rankWant, setBookStatus, setFinishedOn, setGoal, type BookPlace } from "@/server/learn-store";
import { keepClaim, setStance } from "@/server/claims-store";
import { addExcerpt, allExcerpts, removeExcerpt, reviseExcerpt, type Anchor } from "@/server/excerpt-store";
import { importKindle, type KindleReport } from "@/server/kindle-import";
import { publishLesson, publishQuote, unpublishQuote } from "@/server/published-store";
import { importWork } from "@/server/work-store";
import { addItem, addItemNote, addItems, addItemSubjects, addSession, changeItem, editItem, getItem, listItems, keepItemPlace, keepUnitPlace, linkItemClaim, removeItem, removeItemNote, removeItemSubject, removeSession, restoreItem, setBookWay, setItemDates, setItemFacts, setTranscript, setUnitDone } from "@/server/ways-store";
import { mediumOf } from "@reflect/core/ways";
import { lookLink, searchMedium, type Fetch, type Found, type Searched } from "@reflect/core/lookup";
import type { Candidate } from "@reflect/core/item-imports";
import { itemListById } from "@reflect/core/item-lists";
import { excerptsMarkdown, isHighlightColor } from "@reflect/core/excerpts";
import { bestListMatch, listBookQuery, listById } from "@reflect/core/lists";
import { searchBooks } from "@reflect/core/openlibrary";
import { learnView } from "@/server/learn-store";
import { highlightBooks } from "@/server/highlights";
import { definitionsOf, lookupWord, type Definition } from "@reflect/core/wiktionary";
import { lectureOf, lecturesOf } from "@/server/courses";
import { keepReason } from "@/server/learn-reason-store";
import { courseHref } from "./courses/shell";

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
export async function keepPlaceAction(bookId: string, place: BookPlace, day?: string): Promise<void> {
  const user = await requireUser();
  await keepPlace(user.id, bookId, { chapter: Number(place?.chapter), chapters: Number(place?.chapters), at: Number(place?.at) }, typeof day === "string" ? day : undefined);
}

/** Set the day the person finished a read book. */
export async function setFinishedAction(bookId: string, day: string): Promise<void> {
  const user = await requireUser();
  await setFinishedOn(user.id, String(bookId), String(day));
  revalidatePath("/learn/reading");
}

/** Order Want to read as the person dragged it. */
export async function rankWantAction(ids: string[]): Promise<void> {
  const user = await requireUser();
  await rankWant(user.id, Array.isArray(ids) ? ids.map(String) : []);
  revalidatePath("/learn/reading");
}

/** Set or clear the person's reading goal for a year. */
export async function setGoalAction(year: number, books: number | null): Promise<void> {
  const user = await requireUser();
  await setGoal(user.id, Number(year), books === null ? null : Number(books));
  revalidatePath("/learn/reading");
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

/**
 * Put a list's books that are not on the person's shelves on Want to read, in the list's order, after the books
 * already there. Returns how many were added.
 */
export async function followListAction(id: string): Promise<number> {
  const user = await requireUser();
  const list = listById(String(id));
  if (!list) throw new Error("no such list");
  const found = await Promise.all(list.books.map((b) => searchBooks(listBookQuery(b), openLibrary, 6).then((r) => bestListMatch(b, r)).catch(() => null)));
  const view = await learnView(user.id);
  const have = new Set(view.books.map((b) => b.catalog?.work).filter(Boolean));
  const added: string[] = [];
  for (const b of found) {
    if (!b || have.has(b.work)) continue;
    have.add(b.work);
    added.push(await addBook(user.id, { title: b.title, authors: b.authors, status: "want", subjects: [], catalog: b }));
  }
  const before = view.books.filter((b) => b.status === "want").sort((a, b) => (a.wantRank ?? 1e9) - (b.wantRank ?? 1e9)).map((b) => b.id);
  await rankWant(user.id, [...before, ...added]);
  revalidatePath("/learn/reading");
  revalidatePath(`/learn/lists/${list.id}`);
  return added.length;
}

const catalog: Fetch = (url) => fetch(url, { headers: { "user-agent": "reflect-app (learn items)" }, next: { revalidate: 86400 }, signal: AbortSignal.timeout(8000) });
const field = (form: FormData, k: string) => String(form.get(k) ?? "");
const has = (form: FormData, k: string) => form.has(k);

/** Reads a pasted link into an item to add (title, maker, picture), from the site's own endpoint. */
export async function lookLinkAction(url: string): Promise<{ found: Found | null; error?: string }> {
  await requireUser();
  return lookLink(url, catalog);
}

/** Searches the free catalog for a medium. */
export async function searchItemsAction(medium: string, q: string): Promise<Searched> {
  await requireUser();
  return searchMedium(medium, q, catalog);
}

/** Keep a lecture, course, article, debate, discussion, podcast, film, music, art or game. */
export async function addItemAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const parts = Number(field(form, "parts"));
  await addItem(user.id, {
    medium: field(form, "medium"), way: field(form, "way") || null, status: field(form, "status"), title: field(form, "title"), by: field(form, "by"), url: field(form, "url"),
    image: field(form, "image"), work: field(form, "work"), lesson: field(form, "lesson"), tookPart: field(form, "took_part") === "on", day: field(form, "day"),
    ...(parts ? { parts } : {}),
  });
  revalidatePath("/learn");
}

/**
 * Adds a find from Browse to Want to, once (an item of the same medium and title already kept is reused); with
 * `go` it is started and its page opened, so it can be done straight away in Reflect.
 */
export async function browseAddAction(form: FormData): Promise<void> {
  const user = await requireUser();
  await addFromForm(user.id, form);
}

/**
 * Keeps what is causing the person to learn a course and for the sake of what, as a work of their journal.
 * With go=start it then starts the course; otherwise it returns to the course.
 */
export async function keepReasonAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const url = field(form, "url");
  const title = field(form, "title");
  await keepReason(user.id, { courseUrl: url, courseTitle: title, cause: field(form, "cause"), end: field(form, "end"), origins: form.getAll("origin"), endKind: field(form, "endKind") });
  revalidatePath("/learn/courses");
  revalidatePath("/journal");
  if (field(form, "go") === "start") await addFromForm(user.id, form);
  redirect(courseHref(url, title));
}

async function addFromForm(userId: string, form: FormData): Promise<void> {
  const user = { id: userId };
  const medium = field(form, "medium");
  const title = field(form, "title");
  const go = field(form, "go") === "start";
  const same = (await listItems(user.id)).find((i) => i.medium === medium && i.title.toLowerCase() === title.trim().toLowerCase());
  const id = same?.id ?? await addItem(user.id, { medium, title, by: field(form, "by"), url: field(form, "url"), image: field(form, "image"), status: go ? "doing" : "want", day: field(form, "day") });
  if (same && go && same.status === "want") await changeItem(user.id, id, { status: "doing", day: field(form, "day") });
  revalidatePath("/learn");
  revalidatePath("/learn/browse");
  if (go) redirect(`/learn/items/${id}`);
}

/** Keeps the candidates the person picked from an import; returns how many were new. */
export async function importItemsAction(list: Candidate[]): Promise<number> {
  const user = await requireUser();
  const n = await addItems(user.id, list.slice(0, 500));
  revalidatePath("/learn");
  return n;
}

/** Change an item's way (empty follows its medium), status, or add what it taught the person. */
export async function changeItemAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = field(form, "item");
  await changeItem(user.id, id, {
    ...(has(form, "way") ? { way: field(form, "way") || null } : {}),
    ...(has(form, "status") ? { status: field(form, "status") } : {}),
    ...(has(form, "lesson") ? { lesson: field(form, "lesson") } : {}),
    day: field(form, "day"),
  });
  revalidatePath("/learn");
  revalidatePath(`/learn/items/${id}`);
}

/** Change an item's title, maker, link, picture or the book it adapts. */
export async function editItemAction(form: FormData): Promise<void> {
  const user = await requireUser();
  await editItem(user.id, field(form, "item"), {
    ...(has(form, "title") ? { title: field(form, "title") } : {}),
    ...(has(form, "by") ? { by: field(form, "by") || null } : {}),
    ...(has(form, "url") ? { url: field(form, "url") || null } : {}),
    ...(has(form, "image") ? { image: field(form, "image") || null } : {}),
    ...(has(form, "work") ? { work: field(form, "work") || null } : {}),
  });
  revalidatePath("/learn");
  revalidatePath(`/learn/items/${field(form, "item")}`);
}

/** Remove an item; the page then offers to undo it. */
export async function removeItemAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = field(form, "item");
  await removeItem(user.id, id);
  revalidatePath("/learn");
  redirect(`/learn?removed=${id}`);
}

export async function restoreItemAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = field(form, "item");
  await restoreItem(user.id, id);
  revalidatePath("/learn");
  redirect(`/learn?item=${id}`);
}

/** Move a book to another way of learning, or back to the way its catalog gives (empty). */
export async function setBookWayAction(form: FormData): Promise<void> {
  const user = await requireUser();
  await setBookWay(user.id, field(form, "book"), field(form, "way") || null);
  revalidatePath("/learn");
}

/** Add a list's items (or one, by title) to Want to, skipping what the person keeps. */
export async function addFromItemListAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const list = itemListById(field(form, "list"));
  if (!list) throw new Error("no such list");
  const title = field(form, "title");
  const pick = list.items.filter((i) => !title || i.title === title);
  await addItems(user.id, pick.map((i) => ({ medium: i.medium, title: i.title, ...(i.by ? { by: i.by } : {}), ...(i.url ? { url: i.url } : {}), status: "want" as const })));
  revalidatePath(`/learn/lists/items/${list.id}`);
}

const itemPath = (id: string) => `/learn/items/${id}`;
const num = (v: string) => (v.trim() === "" ? null : Number(v));

/** Where the person is in a video or recording, kept as they watch or listen. */
export async function keepItemPlaceAction(id: string, seconds: number, day?: string): Promise<void> {
  const user = await requireUser();
  await keepItemPlace(user.id, id, seconds, day);
}

/** Where the person is in a lecture or episode. */
export async function keepUnitPlaceAction(id: string, unit: string, seconds: number, day?: string): Promise<void> {
  const user = await requireUser();
  await keepUnitPlace(user.id, id, unit, seconds, day);
}

/** A lecture or episode finished, or not; `of` is how many the item has. */
export async function setUnitDoneAction(id: string, unit: string, done: boolean, of: number | null, day?: string): Promise<void> {
  const user = await requireUser();
  await setUnitDone(user.id, id, unit, done, of, day);
  revalidatePath(itemPath(id));
}

/** A note at a moment, with words quoted from the transcript if any. */
export async function addItemNoteAction(id: string, n: { at?: number | null; quote?: string; note?: string; unit?: string | null }): Promise<string> {
  const user = await requireUser();
  const nid = await addItemNote(user.id, id, n);
  revalidatePath(itemPath(id));
  return nid;
}

export async function removeItemNoteAction(id: string, noteId: string): Promise<void> {
  const user = await requireUser();
  await removeItemNote(user.id, noteId);
  revalidatePath(itemPath(id));
}

export async function setTranscriptAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = field(form, "item");
  await setTranscript(user.id, id, field(form, "transcript") || null);
  revalidatePath(itemPath(id));
}

export async function setItemDatesAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = field(form, "item");
  await setItemDates(user.id, id, { startedOn: field(form, "started") || null, finishedOn: field(form, "finished") || null });
  revalidatePath(itemPath(id));
}

/** Took part or only heard it; how far through a course and of how many parts. */
export async function setItemFactsAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = field(form, "item");
  const step = field(form, "step");
  const cur = await getItem(user.id, id);
  if (!cur) throw new Error("no such item");
  await setItemFacts(user.id, id, {
    ...(has(form, "took_part") ? { tookPart: field(form, "took_part") === "yes" } : {}),
    ...(has(form, "parts") ? { parts: num(field(form, "parts")) } : {}),
    ...(has(form, "part") ? { part: num(field(form, "part")) } : {}),
    ...(step ? { part: Math.max(0, (cur.part ?? 0) + Number(step)) } : {}),
  });
  revalidatePath(itemPath(id));
}

/** A practice session: the day, minutes if given, and a note. */
export async function addSessionAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = field(form, "item");
  const minutes = num(field(form, "minutes"));
  await addSession(user.id, id, { day: field(form, "on") || field(form, "day"), ...(minutes ? { minutes } : {}), note: field(form, "note") });
  revalidatePath(itemPath(id));
}

export async function removeSessionAction(form: FormData): Promise<void> {
  const user = await requireUser();
  await removeSession(user.id, field(form, "session"));
  revalidatePath(itemPath(field(form, "item")));
}

/** Subjects the item is about, comma separated. */
export async function addItemSubjectsAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = field(form, "item");
  await addItemSubjects(user.id, id, field(form, "subjects").split(","));
  revalidatePath(itemPath(id));
}

export async function removeItemSubjectAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = field(form, "item");
  await removeItemSubject(user.id, id, field(form, "subject"));
  revalidatePath(itemPath(id));
}

/** A claim argued in a debate or discussion: its words, who argued it (the sides), and the person's stance. */
export async function keepItemClaimAction(form: FormData): Promise<void> {
  const user = await requireUser();
  const id = field(form, "item");
  const item = await getItem(user.id, id);
  if (!item) throw new Error("no such item");
  const holders = field(form, "holders").split("\n").map((l) => l.trim()).filter(Boolean).map((name) => ({ name: name.slice(0, 200), citation: item.title.slice(0, 300) }));
  const claim = await keepClaim(user.id, { text: field(form, "claim"), holders, ...(field(form, "subject") ? { subject: field(form, "subject") } : {}), stance: field(form, "stance") });
  await linkItemClaim(user.id, id, claim);
  revalidatePath(itemPath(id));
}

export async function setItemStanceAction(form: FormData): Promise<void> {
  const user = await requireUser();
  await setStance(user.id, field(form, "claim"), field(form, "stance"));
  revalidatePath(itemPath(field(form, "item")));
}

/** Publish the item's latest lesson at its own page, once the person confirms. */
export async function publishLessonAction(id: string): Promise<string> {
  const user = await requireUser();
  const item = await getItem(user.id, id);
  if (!item?.lesson) throw new Error("this item has no lesson to publish");
  return publishLesson(user.id, { lesson: item.lesson, title: item.title, by: item.by, medium: mediumOf(item.medium).label });
}

/** Take a published quote or lesson down, from a form. */
export async function unpublishQuoteFormAction(form: FormData): Promise<void> {
  const user = await requireUser();
  await unpublishQuote(user.id, field(form, "id"));
  if (field(form, "item")) revalidatePath(`/learn/items/${field(form, "item")}`);
}

/** A lecture's recording, for playing it from a course's list of lectures without opening its page. */
export async function lectureMediaAction(courseUrl: string, n: number): Promise<{ youtube: string | null; video: string | null; audio: string | null } | null> {
  await requireUser();
  const course = await lecturesOf(String(courseUrl));
  const ref = course?.lectures.find((l) => l.n === Number(n));
  if (!course || !ref) return null;
  const l = await lectureOf(course.site, ref);
  return { youtube: l?.youtube ?? ref.youtube, video: l?.video ?? null, audio: l?.audio ?? null };
}
