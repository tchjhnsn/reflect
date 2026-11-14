// Browse beyond books: finds for each medium from free catalogs (@reflect/core/browse-media), cached by Next's fetch
// cache, and the lists' items of a medium.
import { findMedium, freeFilmsToWatch, type Fetch } from "@reflect/core/browse-media";
import { ITEM_LISTS } from "@reflect/core/item-lists";
import type { Found } from "@reflect/core/lookup";
import type { Medium } from "@reflect/core/ways";

export const textFetch: Fetch = async (url) => {
  try {
    const r = await fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 21600 }, signal: AbortSignal.timeout(8000) });
    return r.ok ? await r.text() : null;
  } catch {
    return null;
  }
};

/** Finds for a medium: a search, or the catalog's own list when there is no search. */
export async function finds(m: Medium, q = ""): Promise<{ found: Found[]; error?: string }> {
  const oycHtml = m === "course" ? await textFetch("https://oyc.yale.edu/courses") : null;
  return findMedium(m, q, textFetch, { oycHtml });
}

export const filmsToWatch = (): Promise<Found[]> => freeFilmsToWatch(textFetch).catch(() => []);

/** The items of the lists beyond books in one medium, credited to their list. */
export function listFinds(m: Medium): (Found & { list: string })[] {
  return ITEM_LISTS.flatMap((l) => l.items.filter((i) => i.medium === m).map((i) => ({ medium: i.medium, title: i.title, ...(i.by ? { by: i.by } : {}), ...(i.url ? { url: i.url } : {}), source: l.name, list: l.name })));
}
