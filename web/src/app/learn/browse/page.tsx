import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import { browseQuery, isSort, searchAuthors, searchBooks, SORTS, trendingBooks, type AuthorHit, type CatalogEntry, type Sort } from "@reflect/core/openlibrary";
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
import { listBooks, openLibrary, PER, type BookSource } from "@/server/browse";

export const metadata: Metadata = { title: "Browse books" };
export const dynamic = "force-dynamic";

const ROW = 6;
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";
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
  const sort: Sort | undefined = isSort(one(sp.sort)) ? (one(sp.sort) as Sort) : undefined;
  const view = await learnView(user.id);
  const mine = new Map(view.books.filter((b) => b.catalog?.work).map((b) => [b.catalog!.work, b.status]));

  let body: ReactNode;
  if (q || section) {
    const source: BookSource = q ? { kind: "q", q } : { kind: "cat", code: category?.code ?? section!.code };
    const [books, authors] = await Promise.all([listBooks(source, 1, sort).catch(() => null), q ? searchAuthors(q, openLibrary).catch((): AuthorHit[] => []) : Promise.resolve([])]);
    const title = q ? `Results for “${q}”` : `${(category ?? section)!.heading}, ${SORTS[sort ?? "rating"].toLowerCase()}`;
    const base: Record<string, string> = q ? { q } : { cat: code };
    body = (
      <>
        {authors.length ? <Authors authors={authors} /> : null}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
          <h2 className="text-lg font-semibold">{title}</h2>
          <SortMenu base={base} sort={sort} match={Boolean(q)} />
        </div>
        <div className="space-y-8">
          <Books books={books} mine={mine} />
          {books?.length ? <MoreBooks key={`${JSON.stringify(source)}-${sort ?? ""}`} source={source} {...(sort ? { sort } : {})} per={PER} first={books.length} /> : null}
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
            lead={2}
            items={[
              { href: "/learn/browse", label: "For you", on: !q && !section && !week },
              { href: "/learn/browse?view=week", label: "This week", on: week },
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

/** The orders a list can be in, as a menu; a search can also be in best-match order, its default. */
function SortMenu({ base, sort, match }: { base: Record<string, string>; sort: Sort | undefined; match: boolean }) {
  const href = (s?: Sort) => `/learn/browse?${new URLSearchParams({ ...base, ...(s ? { sort: s } : {}) })}`;
  const items = (Object.keys(SORTS) as Sort[]).filter((s) => match || s !== "rating").map((s) => ({ href: href(s), label: SORTS[s], on: s === sort }));
  return <Menu label="Sort" items={[{ href: href(), label: match ? "Best match" : SORTS.rating, on: !sort }, ...items]} />;
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
 * subjects they are learning, then this week's reads. Books already on their shelves stay in, marked as theirs.
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
