"use server";

import { isSort, type CatalogEntry } from "@reflect/core/openlibrary";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { cleanSource, isFilter, listBooks } from "@/server/browse";
import { ROWS, rowsAt, type Row } from "@/server/rows";

/**
 * The next page of a list, for More: its books and which of them are on the person's shelves. Pages past the
 * twentieth are not asked for.
 */
export async function moreBooksAction(rawSource: unknown, page: number, sort: unknown, filter?: unknown): Promise<{ books: CatalogEntry[]; mine: Record<string, string> }> {
  const user = await requireUser();
  const source = cleanSource(rawSource);
  if (!source || !Number.isInteger(page) || page < 2 || page > 20) return { books: [], mine: {} };
  const [books, view] = await Promise.all([listBooks(source, page, isSort(sort) ? sort : undefined, isFilter(filter) ? filter : undefined), learnView(user.id)]);
  const works = new Set(books.map((b) => b.work));
  return { books, mine: Object.fromEntries(view.books.filter((b) => b.catalog?.work && works.has(b.catalog.work)).map((b) => [b.catalog!.work, b.status])) };
}

/** The next rows of For you, from row `from`, three at a time, with which of their books are on the person's shelves. */
export async function moreRowsAction(from: number): Promise<{ rows: Row[]; mine: Record<string, string>; next: number | null }> {
  const user = await requireUser();
  if (!Number.isInteger(from) || from < 0 || from >= ROWS.length) return { rows: [], mine: {}, next: null };
  const [rows, view] = await Promise.all([rowsAt(from, 3), learnView(user.id)]);
  const works = new Set(rows.flatMap((r) => r.books.map((b) => b.work)));
  return { rows, mine: Object.fromEntries(view.books.filter((b) => b.catalog?.work && works.has(b.catalog.work)).map((b) => [b.catalog!.work, b.status])), next: from + 3 < ROWS.length ? from + 3 : null };
}
