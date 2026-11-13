import { shelveThema, type ThemaShelves } from "@reflect/core/thema";
import type { LearnView } from "./learn-store";

export type ShelvedBook = { id: string; title: string; subjects: string[]; lcc?: string[]; catalogSubjects?: string[] };

/**
 * The person's books and subjects on their Thema sections, as Learn and the chat show them. A book is placed
 * by its catalog's subjects and call numbers; the call number stays its exact address.
 */
export function shelvesOf(v: LearnView, all = false): ThemaShelves<ShelvedBook, { id: string; name: string }> {
  const seen = new Map<string, { id: string; name: string }>();
  for (const s of [...v.known, ...v.shown, ...v.vouched, ...v.suggested, ...v.books.flatMap((b) => b.subjects)]) seen.set(s.id, { id: s.id, name: s.name });
  return shelveThema(
    v.books.map((b) => ({ id: b.id, title: b.title, subjects: b.subjects.map((x) => x.id), ...(b.catalog?.lcc ? { lcc: b.catalog.lcc } : {}), ...(b.catalog?.subjects.length ? { catalogSubjects: b.catalog.subjects } : {}) })),
    [...seen.values()].sort((a, b) => a.name.localeCompare(b.name)),
    all,
  );
}
