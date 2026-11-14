"use server";

import { mediumOf } from "@reflect/core/ways";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { listItems } from "@/server/ways-store";

export type Found = { title: string; href: string; kind: string };

/** What on the person's shelves matches a search, books and everything else, by title or maker; up to eight. */
export async function searchMineAction(q: string): Promise<Found[]> {
  const user = await requireUser();
  const words = q.toLowerCase().split(/\s+/).filter(Boolean).slice(0, 6);
  if (!words.length) return [];
  const hit = (s: string) => words.every((w) => s.toLowerCase().includes(w));
  const [view, items] = await Promise.all([learnView(user.id), listItems(user.id)]);
  const books: Found[] = view.books.filter((b) => b.catalog?.work && hit(`${b.title} ${b.authors.join(" ")}`)).map((b) => ({ title: b.title, href: `/learn/works/${b.catalog!.work}`, kind: "Book" }));
  const rest: Found[] = items.filter((i) => hit(`${i.title} ${i.by ?? ""}`)).map((i) => ({ title: i.title, href: `/learn/items/${i.id}`, kind: mediumOf(i.medium).label }));
  return [...books, ...rest].slice(0, 8);
}
