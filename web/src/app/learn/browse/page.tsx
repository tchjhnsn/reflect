import type { Metadata } from "next";
import Link from "next/link";
import { Suspense, type ReactNode } from "react";
import { BROAD_SUBJECT as BROAD, browseQuery, isSort, searchAuthors, searchBooks, SORTS, trendingBooks, type AuthorHit, type CatalogEntry, type Sort } from "@reflect/core/openlibrary";
import { THEMA, themaCategory, themaQuery, themaSection } from "@reflect/core/thema";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { learnView, type LearnView } from "@/server/learn-store";
import { LearnHeader } from "../header";
import { GoodreadsImport } from "../goodreads";
import { BookGrid } from "../book-card";
import { Menu } from "./category-menu";
import { MoreBooks } from "./more-books";
import { BookRow, MoreRows, RowsLoading } from "./more-rows";
import { rowsAt } from "@/server/rows";
import { FilterMenu, SortMenu, type ListState } from "./list-menus";
import { isFilter, listBooks, newQuery, openLibrary, PER, type BookSource, type Filter } from "@/server/browse";

export const metadata: Metadata = { title: "Browse books" };
export const dynamic = "force-dynamic";

const ROW = 6;
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

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
  const fresh = !q && !section && one(sp.view) === "new";
  const free = !q && !section && one(sp.view) === "free";
  const sort: Sort | undefined = isSort(one(sp.sort)) ? (one(sp.sort) as Sort) : undefined;
  const state: ListState = { ...(sort ? { sort } : {}), ...(isFilter(one(sp.filter)) ? { filter: one(sp.filter) as Filter } : {}), hide: one(sp.hide) === "1" };
  const view = await learnView(user.id);
  const mine = new Map(view.books.filter((b) => b.catalog?.work).map((b) => [b.catalog!.work, b.status]));

  let body: ReactNode;
  if (q || section || fresh || free) {
    const source: BookSource = q ? { kind: "q", q } : fresh ? { kind: "new" } : free ? { kind: "free" } : { kind: "cat", code: category?.code ?? section!.code };
    const [books, authors] = await Promise.all([listBooks(source, 1, sort, state.filter).catch(() => null), q ? searchAuthors(q, openLibrary).catch((): AuthorHit[] => []) : Promise.resolve([])]);
    const title = q ? `Results for “${q}”` : free ? `Free in the public domain, ${SORTS[sort ?? "readinglog"].toLowerCase()}` : fresh ? `New this year, ${SORTS[sort ?? "readinglog"].toLowerCase()}` : `${(category ?? section)!.heading}, ${SORTS[sort ?? "rating"].toLowerCase()}`;
    const base: Record<string, string> = q ? { q } : fresh ? { view: "new" } : free ? { view: "free" } : { cat: code };
    body = (
      <>
        {authors.length ? <Authors authors={authors} /> : null}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
          <h2 className="text-lg font-semibold">{title}</h2>
          <div className="flex flex-wrap gap-2">
            <FilterMenu path="/learn/browse" base={base} state={state} />
            <SortMenu path="/learn/browse" base={base} state={state} match={Boolean(q)} fallback={fresh || free ? "readinglog" : "rating"} />
          </div>
        </div>
        <div className="space-y-8">
          <Books books={books && state.hide ? books.filter((b) => !mine.has(b.work)) : books} mine={mine} />
          {books?.length ? <MoreBooks key={JSON.stringify([source, state])} source={source} {...(sort ? { sort } : {})} {...(state.filter ? { filter: state.filter } : {})} hide={state.hide} per={PER} first={books.length} /> : null}
        </div>
      </>
    );
  } else if (week) {
    const books = await trendingBooks(openLibrary, PER).catch(() => null);
    body = (
      <>
        <h2 className="pt-2 text-lg font-semibold">Read this week</h2>
        <Books books={books} mine={mine} />
      </>
    );
  } else {
    body = <ForYou view={view} mine={mine} />;
  }

  return (
    <main className={PAGE}>
      <LearnHeader on="browse" />
      <div className="mt-6 space-y-5">
        <div className="flex flex-wrap items-center gap-2">
          <form action="/learn/browse" className="flex min-w-0 flex-1 basis-72 gap-2">
            <label htmlFor="q" className="sr-only">Search books by title, author or ISBN</label>
            <Input id="q" name="q" defaultValue={q} placeholder="Search books by title, author or ISBN" className="h-10" />
            <Button type="submit" className="h-10">Search</Button>
          </form>
          <Menu
            label="Browse"
            lead={5}
            items={[
              { href: "/learn/browse", label: "For you", on: !q && !section && !week && !fresh && !free },
              { href: "/learn/browse?view=week", label: "This week", on: week },
              { href: "/learn/browse?view=new", label: "New this year", on: fresh },
              { href: "/learn/browse?view=free", label: "Free in the public domain", on: free },
              { href: "/learn/lists", label: "Lists", on: false },
              ...THEMA.map((x) => ({ href: `/learn/browse?cat=${x.code}`, label: x.short, on: x.code === section?.code })),
            ]}
          />
          {section && section.children.length > 1 ? (
            <Menu
              label={`In ${section.short}`}
              lead={1}
              items={[
                { href: `/learn/browse?cat=${section.code}`, label: `All of ${section.short.toLowerCase()}`, on: !category },
                ...section.children.map((c) => ({ href: `/learn/browse?cat=${c.code}`, label: c.short, on: c.code === category?.code })),
              ]}
            />
          ) : null}
          <GoodreadsImport />
        </div>
        <p className="text-muted-foreground text-sm">Books from Open Library. Add one and it goes in its section in My shelves.</p>
        {body}
        <p className="text-muted-foreground pt-6 text-center text-xs">Categories from Thema, the book trade&apos;s subject scheme (EDItEUR). Books, covers and ratings from Open Library.</p>
      </div>
    </main>
  );
}

/** Authors whose name matches a search, each opening their page. */
function Authors({ authors }: { authors: AuthorHit[] }) {
  return (
    <section className="space-y-3 pt-2">
      <h2 className="text-lg font-semibold">Authors</h2>
      <ul className="flex flex-wrap gap-2">
        {authors.map((a) => (
          <li key={a.id}>
            <Link href={`/learn/authors/${a.id}`} className="hover:bg-muted flex items-center gap-3 rounded-md border py-2 pr-4 pl-2 transition">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`https://covers.openlibrary.org/a/olid/${a.id}-S.jpg`} alt="" className="bg-muted size-10 rounded-full object-cover" loading="lazy" />
              <span className="min-w-0">
                <span className="block truncate text-sm font-medium">{a.name}</span>
                <span className="text-muted-foreground block truncate text-xs">{[a.birth || a.death ? `${a.birth ?? ""}–${a.death ?? ""}` : null, a.topWork, `${a.works.toLocaleString("en-US")} works`].filter(Boolean).join(" · ")}</span>
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Books({ books, mine }: { books: CatalogEntry[] | null; mine: Map<string, string> }) {
  if (books === null) return <p className="text-muted-foreground text-sm">The open library did not answer. Try again in a moment.</p>;
  if (!books.length) return <p className="text-muted-foreground text-sm">Nothing found.</p>;
  return <BookGrid books={books} mine={mine} />;
}

/**
 * Lists made from the person's own records: books like the one they are reading or read last, books on the
 * subjects they are learning, their authors, this week's reads and new books; then rows for everyone, loading
 * as the person scrolls (`src/server/rows.ts`). Books already on their shelves stay in, marked as theirs.
 */
async function ForYou({ view, mine }: { view: LearnView; mine: Map<string, string> }) {
  const rank = { reading: 0, read: 1, want: 2 } as Record<string, number>;
  const seed = [...view.books].filter((b) => b.catalog?.subjects.some((s) => !BROAD.test(s))).sort((a, b) => (rank[a.status] ?? 3) - (rank[b.status] ?? 3))[0];
  const seedSubject = seed?.catalog?.subjects.find((s) => !BROAD.test(s));
  const learning = [...new Set([...view.books.filter((b) => b.status !== "read").flatMap((b) => b.subjects), ...view.known, ...view.shown].map((s) => s.name))].slice(0, 2);
  // Books already on the shelves stay in a list, marked as yours, so a list does not change as you add.
  const fresh = (books: CatalogEntry[]) => books.filter((b) => b.work !== seed?.catalog?.work).slice(0, ROW);
  const [like, ...bySubject] = await Promise.all([
    seedSubject ? browseQuery(`subject:"${seedSubject.replace(/"/g, "")}"`, openLibrary, { limit: ROW * 2, sort: "readinglog" }).then(fresh).catch(() => null) : Promise.resolve(null),
    ...learning.map((name) => searchBooks(name, openLibrary, ROW * 2).then(fresh).catch(() => null)),
  ]);
  const [week, recent] = await Promise.all([
    trendingBooks(openLibrary, ROW * 2).then((b) => b.slice(0, ROW * 2)).catch(() => null),
    browseQuery(newQuery(), openLibrary, { limit: ROW, sort: "readinglog" }).catch(() => null),
  ]);
  // The authors of the person's own books, most books first.
  const authors = new Map<string, { name: string; id?: string; n: number }>();
  for (const b of view.books) {
    const name = b.catalog?.authors[0] ?? b.authors[0];
    if (!name) continue;
    const id = b.catalog?.authorKeys?.[0];
    const a = authors.get(name) ?? { name, n: 0, ...(id ? { id } : {}) };
    authors.set(name, { ...a, n: a.n + 1, ...(id && !a.id ? { id } : {}) });
  }
  const yours = [...authors.values()].sort((a, b) => b.n - a.n || a.name.localeCompare(b.name)).slice(0, 12);
  const row = (title: ReactNode, books: CatalogEntry[] | null, href?: string) =>
    books?.length ? (
      <section key={String(href ?? title)} className="space-y-4">
        <div className="flex items-baseline justify-between gap-3 pt-2">
          <h2 className="text-lg font-semibold">{title}</h2>
          {href ? <Link href={href} className="text-muted-foreground shrink-0 text-sm underline-offset-4 hover:underline">See all</Link> : null}
        </div>
        <BookGrid books={books} mine={mine} row />
      </section>
    ) : null;
  return (
    <div className="space-y-8">
      {seed && seedSubject ? row(<>Because you shelved <em className="font-serif">{seed.title}</em></>, like, `/learn/browse?q=${encodeURIComponent(seedSubject)}`) : null}
      {learning.map((name, i) => row(<>On {name}, one of your subjects</>, bySubject[i] ?? null, `/learn/browse?q=${encodeURIComponent(name)}`))}
      {yours.length ? (
        <section className="space-y-3">
          <h2 className="pt-2 text-lg font-semibold">Your authors</h2>
          <ul className="flex flex-wrap gap-2">
            {yours.map((a) => (
              <li key={a.name}>
                <Link href={a.id ? `/learn/authors/${a.id}` : `/learn/browse?q=${encodeURIComponent(a.name)}`} className="hover:bg-muted flex items-center gap-2 rounded-full border py-1 pr-3 pl-1 text-sm transition">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  {a.id ? <img src={`https://covers.openlibrary.org/a/olid/${a.id}-S.jpg`} alt="" loading="lazy" className="bg-muted size-7 rounded-full object-cover" /> : <span className="bg-muted size-7 rounded-full" />}
                  {a.name}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      {row("Read this week", week, "/learn/browse?view=week")}
      {row("New this year", recent, "/learn/browse?view=new")}
      {!week ? <p className="text-muted-foreground text-sm">The open library did not answer. Try again in a moment.</p> : null}
      <Suspense fallback={<RowsLoading />}>
        <FirstRows mine={mine} />
      </Suspense>
      <MoreRows from={FIRST} />
    </div>
  );
}

/** How many of the rows below the person's own are made with the page; the rest load as the person scrolls. */
const FIRST = 3;

/** The first rows below the person's own: free to read, free to hear, and a list. */
async function FirstRows({ mine }: { mine: Map<string, string> }) {
  const rows = await rowsAt(0, FIRST).catch(() => []);
  return <>{rows.map((r) => <BookRow key={r.key} row={r} mine={mine} />)}</>;
}
