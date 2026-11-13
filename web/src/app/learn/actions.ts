"use server";

import { revalidatePath } from "next/cache";
import { BOOK_STATUS, type BookStatus } from "@reflect/core/learning";
import { cleanEntry, workEntry } from "@reflect/core/openlibrary";
import { importGoodreads, type ImportReport } from "@/server/goodreads-import";
import { requireUser } from "@/server/session";
import { addBook, keepPlace, setBookStatus, type BookPlace } from "@/server/learn-store";
import { keepClaim, setStance } from "@/server/claims-store";

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
