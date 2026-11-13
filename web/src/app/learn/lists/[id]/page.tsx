import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { bestListMatch, listBookQuery, listById, pathProgress } from "@reflect/core/lists";
import { coverUrl, searchBooks, type CatalogEntry } from "@reflect/core/openlibrary";
import { cn } from "@/lib/utils";
import { FollowList } from "./follow";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { openLibrary } from "@/server/browse";
import { LearnHeader } from "../../header";

export const metadata: Metadata = { title: "List" };
export const dynamic = "force-dynamic";

/**
 * One list, read in its order: each book numbered, found in the open library by title and author, with where the
 * person is with it, how many they have read, and the next one. Read in this order puts the books not on their
 * shelves on Want to read in the list's order.
 */
export default async function ListPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const list = listById((await params).id);
  if (!list) notFound();
  const [found, view] = await Promise.all([
    Promise.all(list.books.map((b) => searchBooks(listBookQuery(b), openLibrary, 6).then((r) => bestListMatch(b, r)).catch(() => null))),
    learnView(user.id),
  ]);
  const books = [...new Map(found.filter((b): b is CatalogEntry => b !== null).map((b) => [b.work, b])).values()];
  const missing = list.books.filter((_, i) => !found[i]);
  const mine = new Map(view.books.filter((b) => b.catalog?.work).map((b) => [b.catalog!.work, b.status]));
  const progress = pathProgress(books.map((b) => mine.get(b.work) ?? null));
  return (
    <main className={PAGE}>
      <LearnHeader on="browse" />
      <section className="mt-8 space-y-5">
        <div className="space-y-1">
          <p className="text-muted-foreground text-sm"><Link href="/learn/lists" className="hover:underline">Lists</Link></p>
          <h2 className="text-2xl font-semibold tracking-tight">{list.name}</h2>
          <p className="text-muted-foreground max-w-prose text-sm">{list.about}</p>
          <p className="text-muted-foreground text-xs">Source: {list.source}</p>
        </div>
        {books.length ? (
          <>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm">{progress.read} of {books.length} read{progress.next !== null ? <>; next, <Link href={`/learn/works/${books[progress.next]!.work}`} className="font-medium underline underline-offset-2">{books[progress.next]!.title}</Link></> : ", the whole list"}.</p>
              <FollowList id={list.id} missing={books.filter((b) => !mine.has(b.work)).length} />
            </div>
            <div className="bg-border h-1.5 overflow-hidden rounded-full"><div className="bg-foreground/70 h-full" style={{ width: `${(progress.read / books.length) * 100}%` }} /></div>
            <ol className="divide-y rounded-md border">
              {books.map((b, i) => {
                const st = mine.get(b.work);
                const cover = coverUrl(b, "S");
                return (
                  <li key={b.work} className={cn("flex items-center gap-3 px-3 py-2", i === progress.next && "bg-muted/60")}>
                    <span className="text-muted-foreground w-6 text-right text-xs tabular-nums">{i + 1}</span>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    {cover ? <img src={cover} alt="" loading="lazy" className="h-12 w-8 rounded-sm object-cover" /> : <span className="bg-muted h-12 w-8 rounded-sm" />}
                    <span className="min-w-0 flex-1">
                      <Link href={`/learn/works/${b.work}`} className="block truncate text-sm font-medium hover:underline">{b.title}</Link>
                      <span className="text-muted-foreground block truncate text-xs">{b.authors[0] ?? ""}{b.free?.read ? " · read free in Reflect" : ""}</span>
                    </span>
                    <span className="text-muted-foreground shrink-0 text-xs">{st === "read" ? "Read" : st === "reading" ? "Reading" : st === "want" ? "Want to read" : ""}</span>
                  </li>
                );
              })}
            </ol>
          </>
        ) : <p className="text-muted-foreground text-sm">The open library did not answer. Try again in a moment.</p>}
        {books.length && missing.length ? <p className="text-muted-foreground text-sm">Not found in the open library: {missing.map((b) => (b.author ? `${b.title} (${b.author})` : b.title)).join(", ")}.</p> : null}
      </section>
    </main>
  );
}
