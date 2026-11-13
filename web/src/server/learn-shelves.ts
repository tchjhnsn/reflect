import { shelve, type Shelves } from "@reflect/core/shelves";
import type { LearnView } from "./learn-store";

export type ShelvedBook = { id: string; title: string; subjects: string[]; lcc?: string[] };

/** The person's books and subjects on their Library of Congress shelves, as Learn and the chat show them. */
export function shelvesOf(v: LearnView, all = false): Shelves<ShelvedBook, { id: string; name: string }> {
  const seen = new Map<string, { id: string; name: string }>();
  for (const s of [...v.known, ...v.shown, ...v.vouched, ...v.suggested, ...v.books.flatMap((b) => b.subjects)]) seen.set(s.id, { id: s.id, name: s.name });
  return shelve(
    v.books.map((b) => ({ id: b.id, title: b.title, subjects: b.subjects.map((x) => x.id), ...(b.catalog?.lcc ? { lcc: b.catalog.lcc } : {}) })),
    [...seen.values()].sort((a, b) => a.name.localeCompare(b.name)),
    all,
  );
}
