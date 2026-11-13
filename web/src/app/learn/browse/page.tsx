import type { Metadata } from "next";
import Link from "next/link";
import { BookIcon } from "lucide-react";
import { browseClass, coverUrl, searchBooks, trendingBooks, type CatalogEntry } from "@reflect/core/openlibrary";
import { LCC_CLASSES } from "@reflect/core/shelves";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PAGE } from "@/components/page-header";
import { cn } from "@/lib/utils";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { LearnHeader } from "../header";
import { AddBook } from "../add-book";
import { GoodreadsImport } from "../goodreads";

export const metadata: Metadata = { title: "Browse books" };
export const dynamic = "force-dynamic";

const PER = 24;
const STATUS: Record<string, string> = { want: "Want to read", reading: "Reading", read: "Read", abandoned: "Set aside", referenced: "Referenced" };
const openLibrary = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 3600 } });
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/** Browse books that are not on your shelves yet: search the open library, look along a shelf, or see what is read this week. */
export default async function BrowsePage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const q = one(sp.q).trim().slice(0, 200);
  const cls = LCC_CLASSES.find((c) => c.id === one(sp.class).toUpperCase());
  const page = Math.max(1, Math.min(20, Number(one(sp.page)) || 1));
  const view = await learnView(user.id);
  let books: CatalogEntry[] | null;
  let title: string;
  if (q) {
    books = await searchBooks(q, openLibrary, PER, page).catch(() => null);
    title = `Results for “${q}”`;
  } else if (cls) {
    books = await browseClass(cls.id, openLibrary, { limit: PER, page }).catch(() => null);
    title = `${cls.name}, best rated`;
  } else {
    books = await trendingBooks(openLibrary, PER).catch(() => null);
    title = "Read this week";
  }
  const mine = new Map(view.books.filter((b) => b.catalog?.work).map((b) => [b.catalog!.work, b.status]));
  const more = (q || cls) && books?.length === PER ? `/learn/browse?${new URLSearchParams({ ...(q ? { q } : {}), ...(cls ? { class: cls.id } : {}), page: String(page + 1) })}` : null;
  return (
    <main className={PAGE}>
      <LearnHeader on="browse" />
      <div className="mt-6 space-y-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <form action="/learn/browse" className="flex w-full max-w-xl gap-2">
            <label htmlFor="q" className="sr-only">Search books by title or author</label>
            <Input id="q" name="q" defaultValue={q} placeholder="Search books by title or author" className="h-10" />
            <Button type="submit" className="h-10">Search</Button>
          </form>
          <GoodreadsImport />
        </div>
        <p className="text-muted-foreground text-sm">Books from Open Library that are not on your shelves yet. Add one and it goes on its Library of Congress shelf in My shelves.</p>
        <nav aria-label="Browse a shelf" className="-mx-6 flex gap-1.5 overflow-x-auto px-6 pb-1 sm:mx-0 sm:flex-wrap sm:overflow-visible sm:px-0">
          <Link href="/learn/browse" aria-current={!q && !cls ? "page" : undefined} className={cn("shrink-0 rounded-full border px-3 py-1 text-xs transition", !q && !cls ? "bg-foreground text-background border-foreground" : "hover:border-foreground/30")}>This week</Link>
          {LCC_CLASSES.map((c) => (
            <Link key={c.id} href={`/learn/browse?class=${c.id}`} title={c.name} aria-current={c.id === cls?.id ? "page" : undefined}
              className={cn("shrink-0 rounded-full border px-3 py-1 text-xs transition", c.id === cls?.id ? "bg-foreground text-background border-foreground" : "hover:border-foreground/30")}>
              {c.name.split(/[.,(]/)[0]!.trim()}
            </Link>
          ))}
        </nav>
        <h2 className="pt-2 text-lg font-semibold">{title}</h2>
        {books === null ? <p className="text-muted-foreground text-sm">The open library did not answer. Try again in a moment.</p> : null}
        {books && !books.length ? <p className="text-muted-foreground text-sm">Nothing found.</p> : null}
        {books?.length ? (
          <ul className="grid grid-cols-2 gap-x-5 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
            {books.map((b) => {
              const cover = coverUrl(b);
              const have = mine.get(b.work);
              return (
                <li key={b.work} className="flex flex-col gap-2">
                  <div className="bg-muted relative aspect-[2/3] overflow-hidden rounded-md border">
                    {cover ? (
                      // Open Library's covers; next/image would need its host configured, a plain img is enough here.
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={cover} alt="" loading="lazy" className="size-full object-cover" />
                    ) : (
                      <div className="text-muted-foreground flex size-full flex-col items-center justify-center gap-2 p-3 text-center"><BookIcon className="size-6" /><span className="line-clamp-4 text-xs">{b.title}</span></div>
                    )}
                  </div>
                  <div className="min-w-0">
                    <p className="line-clamp-2 text-sm leading-snug font-medium">{b.title}</p>
                    <p className="text-muted-foreground truncate text-xs">{b.authors[0] ?? "Author not given"}{b.year ? ` · ${b.year}` : ""}</p>
                  </div>
                  {have ? <Badge variant="secondary" className="h-8 w-fit max-w-full truncate">On your shelf · {STATUS[have] ?? have}</Badge> : <AddBook entry={b} />}
                </li>
              );
            })}
          </ul>
        ) : null}
        {more ? <div className="flex justify-center pt-2"><Button asChild variant="outline"><Link href={more}>More</Link></Button></div> : null}
      </div>
    </main>
  );
}
