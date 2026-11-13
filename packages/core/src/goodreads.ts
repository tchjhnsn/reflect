// Bringing a Goodreads library in. Goodreads closed its public API to new keys in December 2020, so the way
// in is the export every member can download (My Books, Import and export, Export library): a CSV with one
// row per book. This reads that file; it has no network. The shelf each book is on ("Exclusive Shelf")
// becomes where the person is with it, and "Date Read" the day they finished.
import type { BookStatus } from "./learning.ts";

export type GoodreadsBook = { title: string; authors: string[]; isbn?: string; status: BookStatus; finishedOn?: string; goodreadsId?: string };

/** RFC 4180 rows: quoted fields may hold commas, newlines and doubled quotes. */
export function csvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  const t = text.replace(/^﻿/, "");
  for (let i = 0; i < t.length; i++) {
    const c = t[i]!;
    if (quoted) {
      if (c === '"' && t[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && t[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((f) => f.trim())) rows.push(row);
      row = [];
    } else field += c;
  }
  row.push(field);
  if (row.some((f) => f.trim())) rows.push(row);
  return rows;
}

const SHELF: Record<string, BookStatus> = { read: "read", "currently-reading": "reading", "to-read": "want" };
// Goodreads writes ISBNs as ="0140449132" so spreadsheets keep the leading zero.
const isbnOf = (v: string | undefined) => (v ?? "").replace(/^="?|"$/g, "").replace(/[^0-9X]/gi, "").toUpperCase();
const dayOf = (v: string | undefined) => {
  const m = /^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$/.exec((v ?? "").trim());
  return m ? `${m[1]}-${m[2]!.padStart(2, "0")}-${m[3]!.padStart(2, "0")}` : undefined;
};

/**
 * The books in a Goodreads export, in file order. Throws when the file is not one (no Title column). A book
 * on a shelf of the person's own making, not read or currently reading, is kept as want to read.
 */
export function readGoodreads(text: string): GoodreadsBook[] {
  const [head, ...rows] = csvRows(text);
  const col = (name: string) => (head ?? []).findIndex((h) => h.trim().toLowerCase() === name);
  const at = { id: col("book id"), title: col("title"), author: col("author"), more: col("additional authors"), isbn: col("isbn"), isbn13: col("isbn13"), shelf: col("exclusive shelf"), read: col("date read") };
  if (at.title < 0) throw new Error("This is not a Goodreads export: it has no Title column.");
  const out: GoodreadsBook[] = [];
  for (const r of rows) {
    const title = (r[at.title] ?? "").replace(/\s+/g, " ").trim();
    if (!title) continue;
    const authors = [r[at.author] ?? "", ...(r[at.more] ?? "").split(",")].map((a) => a.replace(/\s+/g, " ").trim()).filter(Boolean);
    const isbn = [isbnOf(r[at.isbn13]), isbnOf(r[at.isbn])].find((i) => /^(97[89])?\d{9}[\dX]$/.test(i));
    const status = SHELF[(r[at.shelf] ?? "").trim().toLowerCase()] ?? "want";
    const finishedOn = status === "read" ? dayOf(r[at.read]) : undefined;
    const id = (r[at.id] ?? "").trim();
    out.push({ title, authors: [...new Set(authors)], status, ...(isbn ? { isbn } : {}), ...(finishedOn ? { finishedOn } : {}), ...(id ? { goodreadsId: id } : {}) });
  }
  return out;
}
