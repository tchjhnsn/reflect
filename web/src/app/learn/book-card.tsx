import Link from "next/link";
import { BookIcon, StarIcon } from "lucide-react";
import type { CatalogEntry } from "@reflect/core/openlibrary";
import { Badge } from "@/components/ui/badge";
import { AddBook } from "./add-book";

// Also rendered in the browser (More), so it imports nothing from openlibrary.ts at run time: that module reads files.
const coverUrl = (e: Pick<CatalogEntry, "cover">) => (e.cover ? `https://covers.openlibrary.org/b/id/${e.cover}-M.jpg` : null);

export const STATUS: Record<string, string> = { want: "Want to read", reading: "Reading", read: "Read", abandoned: "Set aside", referenced: "Referenced" };

/** One book from the library: its cover and title open its book page; below, add it or see it is yours. */
export function BookCard({ b, have }: { b: CatalogEntry; have: string | undefined }) {
  const cover = coverUrl(b);
  return (
    <li className="flex flex-col gap-2">
      <Link href={`/learn/works/${b.work}`} className="group flex flex-col gap-2">
        <div className="bg-muted relative aspect-[2/3] overflow-hidden rounded-md border transition group-hover:opacity-90">
          {cover ? (
            // Open Library's covers; next/image would need its host configured, a plain img is enough here.
            // eslint-disable-next-line @next/next/no-img-element
            <img src={cover} alt="" loading="lazy" className="size-full object-cover" />
          ) : (
            <div className="text-muted-foreground flex size-full flex-col items-center justify-center gap-2 p-3 text-center"><BookIcon className="size-6" /><span className="line-clamp-4 text-xs">{b.title}</span></div>
          )}
        </div>
        <p className="line-clamp-2 text-sm leading-snug font-medium group-hover:underline group-hover:underline-offset-2">{b.title}</p>
      </Link>
      <div className="-mt-1.5 min-w-0">
        <p className="text-muted-foreground truncate text-xs">
          <AuthorName name={b.authors[0]} id={b.authorKeys?.[0]} />
          {b.year ? ` · ${b.year}` : ""}
        </p>
        {b.rating ? <p className="text-muted-foreground flex items-center gap-1 text-xs"><StarIcon className="size-3 fill-current" />{b.rating.average.toFixed(1)} <span>({b.rating.count.toLocaleString("en-US")})</span></p> : null}
      </div>
      {have ? <Badge variant="secondary" className="h-8 w-fit max-w-full truncate">On your shelf · {STATUS[have] ?? have}</Badge> : <AddBook entry={b} />}
    </li>
  );
}

/** An author's name, linked to their page when the catalog gives their id. */
export function AuthorName({ name, id, className }: { name: string | undefined; id: string | undefined; className?: string }) {
  if (!name) return <span className={className}>Author not given</span>;
  return id ? <Link href={`/learn/authors/${id}`} className={`underline-offset-2 hover:text-foreground hover:underline ${className ?? ""}`}>{name}</Link> : <span className={className}>{name}</span>;
}

/** A grid of books, as many columns as the page allows. */
export function BookGrid({ books, mine, cols = "md:grid-cols-4 lg:grid-cols-6" }: { books: CatalogEntry[]; mine: Map<string, string>; /** Column classes from md up, for a grid in a narrower column. */ cols?: string }) {
  return (
    <ul className={`grid grid-cols-2 gap-x-5 gap-y-8 sm:grid-cols-3 ${cols}`}>
      {books.map((b) => <BookCard key={b.work} b={b} have={mine.get(b.work)} />)}
    </ul>
  );
}
