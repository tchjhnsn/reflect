import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import { browseQuery, searchBooks, trendingBooks, type CatalogEntry } from "@reflect/core/openlibrary";
import { THEMA, themaCategory, themaQuery, themaSection } from "@reflect/core/thema";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PAGE } from "@/components/page-header";
import { cn } from "@/lib/utils";
import { requireUser } from "@/server/session";
import { learnView, type LearnView } from "@/server/learn-store";
import { LearnHeader } from "../header";
import { GoodreadsImport } from "../goodreads";
import { BookGrid } from "../book-card";

export const metadata: Metadata = { title: "Browse books" };
export const dynamic = "force-dynamic";

const PER = 24;
const ROW = 6;
const openLibrary = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 3600 } });
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";
const chip = (on: boolean) => cn("shrink-0 rounded-full border px-3 py-1 text-xs transition", on ? "bg-foreground text-background border-foreground" : "hover:border-foreground/30");
// Catalog subjects too broad to make a list from.
const BROAD = /^(fiction|nonfiction|history|biography|literature|juvenile fiction|juvenile literature|english language|large type books|textbooks|in library|accessible book)$/i;

/**
 * Browse books that are not on your shelves yet: lists for you, this week's reads, a Thema section or
 * category best rated first, or a search of the open library.
 */
export default async function BrowsePage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const q = one(sp.q).trim().slice(0, 200);
  const code = one(sp.cat).toUpperCase().slice(0, 2);
  const section = code ? themaSection(code) : undefined;
  const category = code.length === 2 ? themaCategory(code) : undefined;
  const week = !q && !section && one(sp.view) === "week";
  const page = Math.max(1, Math.min(20, Number(one(sp.page)) || 1));
  const view = await learnView(user.id);
  const mine = new Map(view.books.filter((b) => b.catalog?.work).map((b) => [b.catalog!.work, b.status]));

  let body: ReactNode;
  if (q || section || week) {
    let books: CatalogEntry[] | null;
    let title: string;
    if (q) {
      books = await searchBooks(q, openLibrary, PER, page).catch(() => null);
      title = `Results for “${q}”`;
    } else if (section) {
      books = await browseQuery(themaQuery(category?.code ?? section.code)!, openLibrary, { limit: PER, page }).catch(() => null);
      title = `${(category ?? section).heading}, best rated`;
    } else {
      books = await trendingBooks(openLibrary, PER).catch(() => null);
      title = "Read this week";
    }
    const more = !week && books?.length === PER ? `/learn/browse?${new URLSearchParams({ ...(q ? { q } : { cat: code }), page: String(page + 1) })}` : null;
    body = (
      <>
        <h2 className="pt-2 text-lg font-semibold">{title}</h2>
        <Books books={books} mine={mine} />
        {more ? <div className="flex justify-center pt-2"><Button asChild variant="outline"><Link href={more}>More</Link></Button></div> : null}
      </>
    );
  } else {
    body = <ForYou view={view} mine={mine} />;
  }

  return (
    <main className={PAGE}>
      <LearnHeader on="browse" />
      <div className="mt-6 space-y-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <form action="/learn/browse" className="flex w-full max-w-xl gap-2">
            <label htmlFor="q" className="sr-only">Search books by title, author or ISBN</label>
            <Input id="q" name="q" defaultValue={q} placeholder="Search books by title, author or ISBN" className="h-10" />
            <Button type="submit" className="h-10">Search</Button>
          </form>
          <GoodreadsImport />
        </div>
        <p className="text-muted-foreground text-sm">Books from Open Library that are not on your shelves yet. Add one and it goes in its section in My shelves.</p>
        <nav aria-label="Browse by category" className="-mx-6 flex gap-1.5 overflow-x-auto px-6 pb-1 sm:mx-0 sm:flex-wrap sm:overflow-visible sm:px-0">
          <Link href="/learn/browse" aria-current={!q && !section && !week ? "page" : undefined} className={chip(!q && !section && !week)}>For you</Link>
          <Link href="/learn/browse?view=week" aria-current={week ? "page" : undefined} className={chip(week)}>This week</Link>
          {THEMA.map((s) => (
            <Link key={s.code} href={`/learn/browse?cat=${s.code}`} title={s.heading} aria-current={s.code === section?.code ? "page" : undefined} className={chip(s.code === section?.code)}>{s.short}</Link>
          ))}
        </nav>
        {section && section.children.length > 1 ? (
          <nav aria-label={`Categories in ${section.short}`} className="-mx-6 flex gap-1.5 overflow-x-auto border-t px-6 pt-3 pb-1 sm:mx-0 sm:flex-wrap sm:overflow-visible sm:px-0">
            <Link href={`/learn/browse?cat=${section.code}`} aria-current={!category ? "page" : undefined} className={chip(!category)}>All of {section.short.toLowerCase()}</Link>
            {section.children.map((c) => (
              <Link key={c.code} href={`/learn/browse?cat=${c.code}`} title={c.heading} aria-current={c.code === category?.code ? "page" : undefined} className={chip(c.code === category?.code)}>{c.short}</Link>
            ))}
          </nav>
        ) : null}
        {body}
        <p className="text-muted-foreground pt-6 text-center text-xs">Categories from Thema, the book trade&apos;s subject scheme (EDItEUR). Books, covers and ratings from Open Library.</p>
      </div>
    </main>
  );
}

function Books({ books, mine }: { books: CatalogEntry[] | null; mine: Map<string, string> }) {
  if (books === null) return <p className="text-muted-foreground text-sm">The open library did not answer. Try again in a moment.</p>;
  if (!books.length) return <p className="text-muted-foreground text-sm">Nothing found.</p>;
  return <BookGrid books={books} mine={mine} />;
}

/**
 * Lists made from the person's own records: books like the one they are reading or read last, books on the
 * subjects they are learning, then this week's reads. Each list leaves out books already on their shelves.
 */
async function ForYou({ view, mine }: { view: LearnView; mine: Map<string, string> }) {
  const rank = { reading: 0, read: 1, want: 2 } as Record<string, number>;
  const seed = [...view.books].filter((b) => b.catalog?.subjects.some((s) => !BROAD.test(s))).sort((a, b) => (rank[a.status] ?? 3) - (rank[b.status] ?? 3))[0];
  const seedSubject = seed?.catalog?.subjects.find((s) => !BROAD.test(s));
  const learning = [...new Set([...view.books.filter((b) => b.status !== "read").flatMap((b) => b.subjects), ...view.known, ...view.shown].map((s) => s.name))].slice(0, 2);
  const fresh = (books: CatalogEntry[]) => books.filter((b) => !mine.has(b.work)).slice(0, ROW);
  const [like, ...bySubject] = await Promise.all([
    seedSubject ? browseQuery(`subject:"${seedSubject.replace(/"/g, "")}"`, openLibrary, { limit: ROW * 2, sort: "readinglog" }).then(fresh).catch(() => null) : Promise.resolve(null),
    ...learning.map((name) => searchBooks(name, openLibrary, ROW * 2).then(fresh).catch(() => null)),
  ]);
  const week = await trendingBooks(openLibrary, ROW * 2).then((b) => b.slice(0, ROW * 2)).catch(() => null);
  const row = (title: ReactNode, books: CatalogEntry[] | null, href?: string) =>
    books?.length ? (
      <section key={String(href ?? title)} className="space-y-4">
        <div className="flex items-baseline justify-between gap-3 pt-2">
          <h2 className="text-lg font-semibold">{title}</h2>
          {href ? <Link href={href} className="text-muted-foreground shrink-0 text-sm underline-offset-4 hover:underline">See all</Link> : null}
        </div>
        <BookGrid books={books} mine={mine} />
      </section>
    ) : null;
  return (
    <div className="space-y-8">
      {seed && seedSubject ? row(<>Because you shelved <em className="font-serif">{seed.title}</em></>, like, `/learn/browse?q=${encodeURIComponent(seedSubject)}`) : null}
      {learning.map((name, i) => row(<>On {name}, one of your subjects</>, bySubject[i] ?? null, `/learn/browse?q=${encodeURIComponent(name)}`))}
      {row("Read this week", week, "/learn/browse?view=week")}
      {!week ? <p className="text-muted-foreground text-sm">The open library did not answer. Try again in a moment.</p> : null}
    </div>
  );
}
