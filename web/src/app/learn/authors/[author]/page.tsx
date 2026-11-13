import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ExternalLinkIcon, UserIcon } from "lucide-react";
import { AUTHOR_ID, authorDetails, authorPhotoUrl, isSort, SORTS, type Sort } from "@reflect/core/openlibrary";
import { Button } from "@/components/ui/button";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { listBooks, openLibrary, PER, type BookSource } from "@/server/browse";
import { LearnHeader } from "../../header";
import { BookGrid } from "../../book-card";
import { Menu } from "../../browse/category-menu";
import { MoreBooks } from "../../browse/more-books";

export const metadata: Metadata = { title: "Author" };
export const dynamic = "force-dynamic";

const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/** One author: their photo, life dates and biography from Open Library, and their books in English, best rated first. */
export default async function AuthorPage({ params, searchParams }: { params: Promise<{ author: string }>; searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const { author } = await params;
  if (!AUTHOR_ID.test(author)) notFound();
  const sp = await searchParams;
  const sort: Sort | undefined = isSort(one(sp.sort)) ? (one(sp.sort) as Sort) : undefined;
  const source: BookSource = { kind: "author", id: author };
  const [a, books, view] = await Promise.all([authorDetails(author, openLibrary).catch(() => null), listBooks(source, 1, sort ?? "rating").catch(() => null), learnView(user.id)]);
  if (!a) notFound();
  const mine = new Map(view.books.filter((b) => b.catalog?.work).map((b) => [b.catalog!.work, b.status]));
  const photo = authorPhotoUrl(a.photo, "L");
  const href = (s: Sort) => `/learn/authors/${author}${s === "rating" ? "" : `?sort=${s}`}`;

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
          <div className="flex flex-wrap gap-2">
            {a.wikipedia ? <Button asChild variant="ghost" size="sm"><a href={a.wikipedia} target="_blank" rel="noreferrer">Wikipedia<ExternalLinkIcon /></a></Button> : null}
            <Button asChild variant="ghost" size="sm"><a href={`https://openlibrary.org/authors/${author}`} target="_blank" rel="noreferrer">Open Library<ExternalLinkIcon /></a></Button>
          </div>
        </div>
      </div>
      <section className="mt-10 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-semibold">Books by {a.name}, {SORTS[sort ?? "rating"].toLowerCase()}</h2>
          <Menu label="Sort" items={(Object.keys(SORTS) as Sort[]).map((s) => ({ href: href(s), label: SORTS[s], on: s === (sort ?? "rating") }))} />
        </div>
        {books === null ? <p className="text-muted-foreground text-sm">The open library did not answer. Try again in a moment.</p> : !books.length ? <p className="text-muted-foreground text-sm">No books in English by this author in the catalog.</p> : (
          <div className="space-y-8">
            <BookGrid books={books} mine={mine} />
            <MoreBooks key={sort ?? "rating"} source={source} sort={sort ?? "rating"} per={PER} first={books.length} />
          </div>
        )}
        <p className="text-muted-foreground pt-4 text-xs">Biography, photo and books from Open Library, written by its contributors. <Link href="/learn/browse" className="underline underline-offset-2">Browse more books</Link></p>
      </section>
    </main>
  );
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
