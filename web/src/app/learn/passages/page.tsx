import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";
import { SearchIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { requireUser } from "@/server/session";
import { PASSAGE_BOOKS, searchShelves } from "@/server/passages";
import { LearnHeader } from "../header";

export const metadata: Metadata = { title: "Search inside your books" };
export const dynamic = "force-dynamic";

const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/**
 * A phrase looked for inside the whole text of the books on the person's shelves that can be read free in Reflect,
 * with each place it occurs opening the reader at that paragraph.
 */
export default async function PassagesPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const q = one((await searchParams).q).replace(/\s+/g, " ").trim().slice(0, 100);
  return (
    <>
      <LearnHeader on="shelves" />
      <main className="mx-auto w-full max-w-3xl px-4 pt-4 pb-24 sm:px-6">
        <h2 className="text-2xl font-semibold tracking-tight">Search inside your books</h2>
        <p className="text-muted-foreground mt-1 text-sm">Looks for words in the whole text of the books on your shelves that you can read free in Reflect (up to {PASSAGE_BOOKS} of them, the ones you are reading first).</p>
        <form className="mt-4 flex gap-2" role="search">
          <Input name="q" defaultValue={q} placeholder="A word or phrase" aria-label="A word or phrase" className="h-9" />
          <Button type="submit" variant="outline" size="sm" className="h-9" aria-label="Search"><SearchIcon /></Button>
        </form>
        {q.length >= 2 ? (
          <Suspense key={q} fallback={<p className="text-muted-foreground mt-6 text-sm">Searching your books…</p>}>
            <Results userId={user.id} q={q} />
          </Suspense>
        ) : null}
      </main>
    </>
  );
}

async function Results({ userId, q }: { userId: string; q: string }) {
  const { books, searched, unread } = await searchShelves(userId, q);
  const total = books.reduce((n, b) => n + b.total, 0);
  return (
    <section className="mt-6 space-y-8">
      <p className="text-muted-foreground text-sm">
        {searched === 0 ? "None of the books on your shelves can be read free in Reflect yet." : total === 0 ? `“${q}” is not in the ${searched} ${searched === 1 ? "book" : "books"} searched.` : `“${q}” occurs ${total} ${total === 1 ? "time" : "times"} in ${books.length} of the ${searched} ${searched === 1 ? "book" : "books"} searched.`}
        {unread ? ` ${unread} ${unread === 1 ? "text" : "texts"} did not load; try again in a moment.` : ""}
      </p>
      {books.map((b) => (
        <div key={b.work} className="space-y-2">
          <p className="flex flex-wrap items-baseline justify-between gap-2">
            <Link href={`/learn/works/${b.work}`} className="font-serif text-lg hover:underline">{b.title}</Link>
            <Link href={`/learn/works/${b.work}/read?q=${encodeURIComponent(q)}`} className="text-muted-foreground text-xs underline-offset-2 hover:underline">{b.total} {b.total === 1 ? "place" : "places"}{b.total > b.hits.length ? ", see all" : ""}</Link>
          </p>
          <p className="text-muted-foreground -mt-2 text-sm">{b.author}</p>
          <ol className="space-y-2">
            {b.hits.map((h, i) => (
              <li key={i}>
                <Link href={`/learn/works/${b.work}/read?c=${h.chapter}&find=${encodeURIComponent(q)}&p=${h.p}`} className="hover:bg-muted block rounded-md border p-3">
                  <span className="text-muted-foreground block text-xs">{b.chapters[i]}</span>
                  <span className="mt-1 block font-serif">{h.before}<mark className="rounded-sm bg-sky-200/80 text-inherit">{h.match}</mark>{h.after}</span>
                </Link>
              </li>
            ))}
          </ol>
        </div>
      ))}
    </section>
  );
}
