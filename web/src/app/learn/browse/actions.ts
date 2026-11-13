"use server";

import { isSort, type CatalogEntry } from "@reflect/core/openlibrary";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { cleanSource, isFilter, listBooks } from "@/server/browse";

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
