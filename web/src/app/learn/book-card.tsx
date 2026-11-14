import Link from "next/link";
import { BookIcon, BookOpenIcon, HeadphonesIcon, StarIcon } from "lucide-react";
import type { CatalogEntry } from "@reflect/core/openlibrary";
import { categoryOf } from "@reflect/core/thema";
import { WAYS, bookWay } from "@reflect/core/ways";
import { Badge } from "@/components/ui/badge";
import { AddBook } from "./add-book";

// Also rendered in the browser (More), so it imports nothing from openlibrary.ts at run time: that module reads files.
const coverUrl = (e: Pick<CatalogEntry, "cover">, size: "M" | "L" = "M") => (e.cover ? `https://covers.openlibrary.org/b/id/${e.cover}-${size}.jpg` : null);

export const STATUS: Record<string, string> = { want: "Want to read", reading: "Reading", read: "Read", abandoned: "Set aside", referenced: "Referenced" };

/** One book from the library: its cover and title open its book page; below, add it or see it is yours. */
export function BookCard({ b, have, className = "" }: { b: CatalogEntry; have: string | undefined; className?: string }) {
  const cover = coverUrl(b);
  return (
    <li className={`flex flex-col gap-2 ${className}`}>
      <Link href={`/learn/works/${b.work}`} className="group flex flex-col gap-2">
        <div className="bg-muted relative aspect-[2/3] overflow-hidden rounded-md border transition group-hover:opacity-90">
          {cover ? (
            // Open Library's covers; next/image would need its host configured, a plain img is enough here.
            // eslint-disable-next-line @next/next/no-img-element
            <img src={cover} srcSet={`${cover} 1x, ${coverUrl(b, "L")} 2x`} alt="" loading="lazy" decoding="async" className="size-full object-cover" />
          ) : (
            <div className="text-muted-foreground flex size-full flex-col items-center justify-center gap-2 p-3 text-center"><BookIcon className="size-6" /><span className="line-clamp-4 text-xs">{b.title}</span></div>
          )}
          <FreeMarks free={b.free} />
        </div>
        <p className="line-clamp-2 text-sm leading-snug font-medium group-hover:underline group-hover:underline-offset-2">{b.title}</p>
      </Link>
      <div className="-mt-1.5 min-w-0">
        <p className="text-muted-foreground truncate text-xs">
          <AuthorName name={b.authors[0]} id={b.authorKeys?.[0]} />
          {b.year ? ` · ${b.year}` : ""}
        </p>
        {b.rating ? <p className="text-muted-foreground flex items-center gap-1 text-xs"><StarIcon className="size-3 fill-current" />{b.rating.average.toFixed(1)} <span>({b.rating.count.toLocaleString("en-US")})</span></p> : null}
        <p className="text-muted-foreground text-[11px]" title="Way of learning">{WAYS.find((w) => w.id === bookWay(categoryOf(b), b.subjects))!.label}</p>
      </div>
      {have ? <Badge variant="secondary" className="h-8 w-fit max-w-full truncate">On your shelf · {STATUS[have] ?? have}</Badge> : <AddBook entry={b} />}
    </li>
  );
}

/** On a cover: whether the book can be read in Reflect (its text) and whether its audiobook plays in Reflect. */
export function FreeMarks({ free }: { free: CatalogEntry["free"] }) {
  if (!free) return null;
  return (
    <div className="absolute bottom-1.5 left-1.5 flex gap-1">
      {free.read ? <span className="bg-background/90 text-foreground flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[11px] font-medium shadow-sm" title="Read it in Reflect"><BookOpenIcon className="size-3" />Read</span> : null}
      {free.listen ? <span className="bg-background/90 text-foreground flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[11px] font-medium shadow-sm" title="Listen to it in Reflect"><HeadphonesIcon className="size-3" />Listen</span> : null}
    </div>
  );
}

/** An author's name, linked to their page when the catalog gives their id. */
export function AuthorName({ name, id, className }: { name: string | undefined; id: string | undefined; className?: string }) {
  if (!name) return <span className={className}>Author not given</span>;
  return id ? <Link href={`/learn/authors/${id}`} className={`underline-offset-2 hover:text-foreground hover:underline ${className ?? ""}`}>{name}</Link> : <span className={className}>{name}</span>;
}

/** A grid of books, as many columns as the page allows. */
export function BookGrid({ books, mine, cols = "md:grid-cols-4 lg:grid-cols-6", row = false }: { books: CatalogEntry[]; mine: Map<string, string>; /** Column classes from md up, for a grid in a narrower column. */ cols?: string; /** On a phone, one row to swipe through instead of a grid. */ row?: boolean }) {
  if (row) {
    return (
      <ul className={`-mx-4 flex snap-x snap-mandatory gap-4 overflow-x-auto px-4 pb-2 sm:mx-0 sm:grid sm:grid-cols-3 sm:gap-x-5 sm:gap-y-8 sm:overflow-visible sm:px-0 sm:pb-0 ${cols}`}>
        {books.map((b) => <BookCard key={b.work} b={b} have={mine.get(b.work)} className="w-36 shrink-0 snap-start sm:w-auto" />)}
      </ul>
    );
  }
  return (
    <ul className={`grid grid-cols-2 gap-x-5 gap-y-8 sm:grid-cols-3 ${cols}`}>
      {books.map((b) => <BookCard key={b.work} b={b} have={mine.get(b.work)} />)}
    </ul>
  );
}
