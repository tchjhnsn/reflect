import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { listBookQuery, listById } from "@reflect/core/lists";
import { searchBooks, type CatalogEntry } from "@reflect/core/openlibrary";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { openLibrary } from "@/server/browse";
import { LearnHeader } from "../../header";
import { BookGrid } from "../../book-card";

export const metadata: Metadata = { title: "List" };
export const dynamic = "force-dynamic";

/** One list: its books in the list's order, each found in the open library by title and author. */
export default async function ListPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const list = listById((await params).id);
  if (!list) notFound();
  const [found, view] = await Promise.all([
    Promise.all(list.books.map((b) => searchBooks(listBookQuery(b), openLibrary, 1).then((r) => r[0] ?? null).catch(() => null))),
    learnView(user.id),
  ]);
  const books = [...new Map(found.filter((b): b is CatalogEntry => b !== null).map((b) => [b.work, b])).values()];
  const missing = list.books.filter((_, i) => !found[i]);
  const mine = new Map(view.books.filter((b) => b.catalog?.work).map((b) => [b.catalog!.work, b.status]));
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
        {books.length ? <BookGrid books={books} mine={mine} /> : <p className="text-muted-foreground text-sm">The open library did not answer. Try again in a moment.</p>}
        {books.length && missing.length ? <p className="text-muted-foreground text-sm">Not found in the open library: {missing.map((b) => (b.author ? `${b.title} (${b.author})` : b.title)).join(", ")}.</p> : null}
      </section>
    </main>
  );
}
