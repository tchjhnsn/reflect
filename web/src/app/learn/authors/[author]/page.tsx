import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ExternalLinkIcon, UserIcon } from "lucide-react";
import { AUTHOR_ID, authorDetails, authorPhotoUrl, booksLike, isSort, SORTS, type CatalogEntry, type Sort } from "@reflect/core/openlibrary";
import { Button } from "@/components/ui/button";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { isFilter, listBooks, openLibrary, PER, type BookSource, type Filter } from "@/server/browse";
import { LearnHeader } from "../../header";
import { BookGrid } from "../../book-card";
import { FilterMenu, SortMenu, type ListState } from "../../browse/list-menus";
import { MoreBooks } from "../../browse/more-books";

export const metadata: Metadata = { title: "Author" };
export const dynamic = "force-dynamic";

const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/** One author: their photo, life dates and biography from Open Library, how many of their books can be read or heard free in Reflect and which, and their books in English, best rated first. */
export default async function AuthorPage({ params, searchParams }: { params: Promise<{ author: string }>; searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const { author } = await params;
  if (!AUTHOR_ID.test(author)) notFound();
  const sp = await searchParams;
  const sort: Sort | undefined = isSort(one(sp.sort)) ? (one(sp.sort) as Sort) : undefined;
  const state: ListState = { ...(sort ? { sort } : {}), ...(isFilter(one(sp.filter)) ? { filter: one(sp.filter) as Filter } : {}), hide: one(sp.hide) === "1" };
  const source: BookSource = { kind: "author", id: author };
  const [a, books, view, open] = await Promise.all([
    authorDetails(author, openLibrary).catch(() => null),
    listBooks(source, 1, sort ?? "rating", state.filter).catch(() => null),
    learnView(user.id),
    // Their public-domain books, most read first, kept to those Reflect can show as text or play as audio.
    listBooks(source, 1, "readinglog", "free").then((b) => b.filter((x) => x.free?.read || x.free?.listen)).catch((): CatalogEntry[] => []),
  ]);
  if (!a) notFound();
  // Authors near this one: the authors of the most read books on the subject of their best rated book.
  const near = books?.[0] ? await booksLike(books[0], openLibrary, 24).catch((): CatalogEntry[] => []) : [];
  const related = [...new Map(near.flatMap((b) => (b.authorKeys?.[0] && b.authors[0] && b.authorKeys[0] !== author ? [[b.authorKeys[0], b.authors[0]] as const] : []))).entries()].slice(0, 8);
  const mine = new Map(view.books.filter((b) => b.catalog?.work).map((b) => [b.catalog!.work, b.status]));
  const photo = authorPhotoUrl(a.photo, "L");

  return (
    <main className={PAGE}>
      <LearnHeader on="browse" />
      <div className="mt-8 grid gap-8 md:grid-cols-[13rem_1fr]">
        <div className="mx-auto w-40 md:mx-0 md:w-full">
          <div className="bg-muted aspect-[4/5] overflow-hidden rounded-md border shadow-sm">
            {photo ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={photo} alt={`Photo of ${a.name}`} className="size-full object-cover" />
            ) : <div className="text-muted-foreground flex size-full items-center justify-center"><UserIcon className="size-8" /></div>}
          </div>
        </div>
        <div className="min-w-0 space-y-4">
          <div className="space-y-1">
            <h2 className="text-3xl leading-tight font-semibold tracking-tight">{a.name}</h2>
            {a.birth || a.death ? <p className="text-muted-foreground">{a.birth ?? "?"} – {a.death ?? ""}</p> : null}
          </div>
          {a.bio ? <Bio text={a.bio} /> : <p className="text-muted-foreground text-sm">The open library has no biography for this author yet.</p>}
          {open.length ? <p className="text-sm">{freeLine(open.filter((b) => b.free?.read).length, open.filter((b) => b.free?.listen).length)}</p> : null}
          <div className="flex flex-wrap gap-2">
            {a.wikipedia ? <Button asChild variant="ghost" size="sm"><a href={a.wikipedia} target="_blank" rel="noreferrer">Wikipedia<ExternalLinkIcon /></a></Button> : null}
            <Button asChild variant="ghost" size="sm"><a href={`https://openlibrary.org/authors/${author}`} target="_blank" rel="noreferrer">Open Library<ExternalLinkIcon /></a></Button>
          </div>
        </div>
      </div>
      {open.length ? (
        <section className="mt-10 space-y-5">
          <h2 className="text-lg font-semibold">Free to read or hear in Reflect</h2>
          <BookGrid books={open.slice(0, 12)} mine={mine} />
        </section>
      ) : null}
      <section className="mt-10 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-semibold">Books by {a.name}, {SORTS[sort ?? "rating"].toLowerCase()}</h2>
          <div className="flex flex-wrap gap-2">
            <FilterMenu path={`/learn/authors/${author}`} base={{}} state={state} />
            <SortMenu path={`/learn/authors/${author}`} base={{}} state={state} />
          </div>
        </div>
        {books === null ? <p className="text-muted-foreground text-sm">The open library did not answer. Try again in a moment.</p> : !books.length ? <p className="text-muted-foreground text-sm">No books in English by this author in the catalog.</p> : (
          <div className="space-y-8">
            <BookGrid books={state.hide ? books.filter((b) => !mine.has(b.work)) : books} mine={mine} />
            <MoreBooks key={JSON.stringify(state)} source={source} sort={sort ?? "rating"} {...(state.filter ? { filter: state.filter } : {})} hide={state.hide} per={PER} first={books.length} />
          </div>
        )}
        {related.length ? (
          <div className="space-y-3 pt-4">
            <h2 className="text-lg font-semibold">Authors on the same subjects</h2>
            <ul className="flex flex-wrap gap-2">
              {related.map(([id, name]) => (
                <li key={id}>
                  <Link href={`/learn/authors/${id}`} className="hover:bg-muted flex items-center gap-2 rounded-full border py-1 pr-3 pl-1 text-sm transition">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={`https://covers.openlibrary.org/a/olid/${id}-S.jpg`} alt="" loading="lazy" className="bg-muted size-7 rounded-full object-cover" />
                    {name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        <p className="text-muted-foreground pt-4 text-xs">Biography, photo and books from Open Library, written by its contributors. <Link href="/learn/browse" className="underline underline-offset-2">Browse more books</Link></p>
      </section>
    </main>
  );
}

/** "12 of their most read books can be read free in Reflect, and 9 heard." */
function freeLine(read: number, listen: number): string {
  const books = (n: number) => `${n} of their most read books`;
  if (read && listen) return `${books(read)} can be read free in Reflect, and ${listen} heard.`;
  if (read) return `${books(read)} can be read free in Reflect.`;
  return `${books(listen)} can be heard free in Reflect.`;
}

/** A biography, its opening shown and the rest a click away when it runs long. */
function Bio({ text }: { text: string }) {
  const cut = text.length > 700 ? text.indexOf(". ", 450) : -1;
  if (cut < 0 || cut > 800) return <p className="max-w-prose text-sm leading-relaxed whitespace-pre-line">{text}</p>;
  return (
    <details className="group max-w-prose text-sm leading-relaxed">
      <summary className="cursor-pointer list-none whitespace-pre-line [&::-webkit-details-marker]:hidden">
        {text.slice(0, cut + 1)}
        <span className="text-muted-foreground group-open:hidden"> … <span className="text-foreground underline underline-offset-2">Read more</span></span>
      </summary>
      <p className="whitespace-pre-line">{text.slice(cut + 2)}</p>
    </details>
  );
}
