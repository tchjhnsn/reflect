import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";
import { BookIcon, ExternalLinkIcon, StarIcon } from "lucide-react";
import { googleVolume, readerUrl } from "@reflect/core/googlebooks";
import { librivoxBook, type Audiobook } from "@reflect/core/librivox";
import { authorWorks, booksLike, coverUrl, workAccess, workDetails, workEntry, type CatalogEntry } from "@reflect/core/openlibrary";
import { categoryOf, showCall, themaCategory } from "@reflect/core/thema";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { listClaims, type ClaimOut } from "@/server/claims-store";
import { LearnHeader } from "../../header";
import { AddBook } from "../../add-book";
import { Reader, type ReadLink } from "./reader";
import { textSource } from "@/server/texts";
import { excerptsOf } from "@/server/excerpt-store";
import { Excerpts } from "./excerpts";
import { Listen } from "./listen";
import { AuthorName, BookGrid } from "../../book-card";
import { setBookStatusAction } from "../../actions";

export const metadata: Metadata = { title: "Book" };
export const dynamic = "force-dynamic";

const STATUS = [["want", "Want to read"], ["reading", "Reading"], ["read", "Read"]] as const;
const STANCE: Record<string, string> = { belief: "Belief", disbelief: "Disbelief", suspension: "Suspension of judgment" };
const cached = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 86400 } });
// LibriVox can be slow to answer; the page does not wait on it for long.
const slow = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 86400 }, signal: AbortSignal.timeout(6000) });
const Label = ({ children }: { children: ReactNode }) => <h3 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">{children}</h3>;

/**
 * One book's page, for any book in the open library: its cover, what it is about, its readers' ratings, a
 * the whole book to read or hear free when it is in the public domain (the Internet Archive, Project Gutenberg,
 * Standard Ebooks, LibriVox) or a preview from Google Books, its contents and editions, where it sits in Thema,
 * books like it and more by its author, and, when it is yours, where you are with it and the claims on its
 * subjects.
 */
export default async function WorkPage({ params }: { params: Promise<{ work: string }> }) {
  const user = await requireUser();
  const { work } = await params;
  if (!/^OL\d+W$/.test(work)) notFound();
  const [entry, details, access, view, claims, excerpts] = await Promise.all([
    workEntry(work, cached).catch(() => null),
    workDetails(work, cached).catch(() => null),
    workAccess(work, cached).catch(() => null),
    learnView(user.id),
    listClaims(user.id).catch((): ClaimOut[] => []),
    excerptsOf(user.id, work).catch(() => []),
  ]);
  if (!entry && !details) notFound();
  const mine = view.books.find((b) => b.catalog?.work === work);
  const title = entry?.title ?? mine?.title ?? "Untitled";
  const authors = entry?.authors ?? mine?.authors ?? [];
  const authorKeys = entry?.authorKeys ?? mine?.catalog?.authorKeys ?? [];
  const [google, byAuthor, audio, like, text] = await Promise.all([
    googleVolume({ title, authors, ...(entry?.isbn ? { isbn: entry.isbn } : {}) }, cached, process.env.GOOGLE_BOOKS_API_KEY).catch(() => null),
    authorKeys[0] ? authorWorks(authorKeys[0], cached, { limit: 7, sort: "rating" }).then((b) => b.filter((x) => x.work !== work).slice(0, 6)).catch((): CatalogEntry[] => []) : Promise.resolve([] as CatalogEntry[]),
    access?.librivox.length ? librivoxBook(access.librivox, slow).catch((): Audiobook | null => null) : Promise.resolve(null),
    entry ? booksLike({ work, subjects: details?.subjects.length ? details.subjects : entry.subjects }, cached).catch((): CatalogEntry[] => []) : Promise.resolve([] as CatalogEntry[]),
    access?.access === "public" || access?.gutenberg.length || access?.standardEbooks.length || (entry?.year ?? 9999) < 1930 ? textSource({ title, authors }, access) : Promise.resolve(null),
  ]);
  // Without a text in Reflect: a scan at the Internet Archive, or a loan from Open Library, off the page.
  const links: ReadLink[] = text ? [] : [
    ...(access?.access === "public" && access.archive ? [{ name: "Scanned book at the Internet Archive", href: `https://archive.org/details/${access.archive}` }] : []),
    ...(access?.access === "borrowable" && access.edition ? [{ name: "Borrow from Open Library", href: `https://openlibrary.org/books/${access.edition}` }] : []),
  ];
  const preview = google?.read ? { src: readerUrl(google.read.id), full: google.read.full } : null;
  const shelved = new Map(view.books.filter((b) => b.catalog?.work).map((b) => [b.catalog!.work, b.status]));
  const subjects = details?.subjects.length ? details.subjects : entry?.subjects ?? [];
  const category = themaCategory(categoryOf({ ...(entry?.lcc ? { lcc: entry.lcc } : {}), subjects }) ?? "");
  const cover = entry?.cover ? coverUrl(entry, "L") : mine?.catalog ? coverUrl(mine.catalog, "L") : null;
  const description = details?.description ?? google?.description ?? null;
  const mySubjects = new Set(mine?.subjects.map((s) => s.id) ?? []);
  const lower = title.toLowerCase();
  const related = claims.filter((c) => (c.subject && mySubjects.has(c.subject.id)) || c.holders.some((h) => `${h.name} ${h.citation ?? ""}`.toLowerCase().includes(lower)));

  return (
    <main className={PAGE}>
      <LearnHeader on={mine ? "shelves" : "browse"} />
      <div className="mt-8 grid gap-8 md:grid-cols-[13rem_1fr]">
        <div className="mx-auto w-40 md:mx-0 md:w-full">
          <div className="bg-muted aspect-[2/3] overflow-hidden rounded-md border shadow-sm">
            {cover ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={cover} alt={`Cover of ${title}`} className="size-full object-cover" />
            ) : <div className="text-muted-foreground flex size-full items-center justify-center"><BookIcon className="size-8" /></div>}
          </div>
        </div>
        <div className="min-w-0 space-y-6">
          <div className="space-y-2">
            <h2 className="text-3xl leading-tight font-semibold tracking-tight">{title}</h2>
            <p className="text-muted-foreground">{authors.length ? authors.map((name, i) => <span key={`${name}-${i}`}>{i ? ", " : ""}<AuthorName name={name} id={authorKeys[i]} /></span>) : "Author not given"}{entry?.year ? ` · first published ${entry.year}` : ""}</p>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
              {details?.rating ? <Rating source="Open Library" r={details.rating} /> : null}
              {!details?.rating ? <span className="text-muted-foreground">No ratings yet</span> : null}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            {mine ? (
              <form action={setBookStatusAction} className="flex flex-wrap gap-2">
                <input type="hidden" name="book" value={mine.id} />
                {STATUS.map(([id, label]) => <Button key={id} size="sm" name="status" value={id} variant={mine.status === id ? "default" : "outline"} aria-pressed={mine.status === id}>{label}</Button>)}
              </form>
            ) : entry ? <AddBook entry={entry} /> : null}
            {mine ? <span className="text-muted-foreground text-xs">On your shelves</span> : null}
          </div>
          {text || preview || links.length ? <Reader links={links} work={work} kept={mine?.place ?? null} preview={preview} {...(text ? { here: `/learn/works/${work}/read` } : {})} /> : null}
          {audio ? <Listen book={audio} /> : null}
          <Excerpts work={work} title={title} author={authors[0] ?? ""} rows={excerpts.map((e) => ({ id: e.id, quote: e.quote, where: e.anchor?.chapterTitle || e.place, chapter: e.anchor?.chapter ?? null, note: e.note }))} {...(text ? { reader: `/learn/works/${work}/read` } : {})} />
          <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
            <div><dt className="text-muted-foreground text-xs">Category</dt><dd>{category ? <Link className="underline-offset-4 hover:underline" href={`/learn/browse?cat=${category.code}`}>{category.heading}</Link> : "Not yet placed"}</dd></div>
            <div><dt className="text-muted-foreground text-xs">Library of Congress call number</dt><dd>{entry?.lcc?.[0] ? showCall(entry.lcc[0]) : "None in the catalog"}</dd></div>
          </dl>
          {description ? (
            <section className="space-y-2">
              <Label>About this book</Label>
              <p className="max-w-prose text-sm leading-relaxed whitespace-pre-line">{description}</p>
              <p className="text-muted-foreground text-xs">From {details?.description ? "Open Library" : "Google Books"}, written by its contributors.</p>
            </section>
          ) : null}
          {subjects.length ? (
            <section className="space-y-2">
              <Label>Subjects in the catalog</Label>
              <div className="flex flex-wrap gap-1.5">{subjects.slice(0, 16).map((s) => <Badge key={s} asChild variant="outline"><Link href={`/learn/browse?q=${encodeURIComponent(s)}`}>{s}</Link></Badge>)}</div>
            </section>
          ) : null}
          {details?.contents.length ? (
            <details className="group space-y-2">
              <summary className="cursor-pointer list-none [&::-webkit-details-marker]:hidden"><Label>Contents <span className="normal-case group-open:hidden">· show</span></Label></summary>
              <ol className="mt-2 max-w-prose space-y-0.5 text-sm">{details.contents.map((c, i) => <li key={i} className="whitespace-pre">{c}</li>)}</ol>
            </details>
          ) : null}
          {details?.editions.count ? (
            <section className="space-y-2">
              <Label>{details.editions.count.toLocaleString("en-US")} {details.editions.count === 1 ? "edition" : "editions"}</Label>
              <ul className="divide-y rounded-md border text-sm">
                {details.editions.first.slice(0, 6).map((e, i) => (
                  <li key={`${e.isbn ?? e.title}-${i}`} className="flex items-center gap-3 px-3 py-2">
                    <div className="bg-muted h-12 w-8 shrink-0 overflow-hidden rounded-sm border">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      {e.cover ? <img src={`https://covers.openlibrary.org/b/id/${e.cover}-S.jpg`} alt="" loading="lazy" className="size-full object-cover" /> : null}
                    </div>
                    <span className="min-w-0 flex-1 truncate">{e.title}</span>
                    <span className="text-muted-foreground shrink-0 text-right text-xs">{[e.publisher, e.year, e.pages ? `${e.pages} pages` : null, e.language?.toUpperCase()].filter(Boolean).join(" · ")}</span>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
          {mine || related.length ? (
            <Card className="gap-3 py-5">
              <CardHeader className="px-5"><CardTitle className="text-sm">Your claims on this book&apos;s subjects</CardTitle></CardHeader>
              <CardContent className="space-y-3 px-5 text-sm">
                {related.length ? related.map((c) => (
                  <div key={c.id}>
                    <p className="font-medium">{c.text}</p>
                    <p className="text-muted-foreground text-xs">{[c.stance ? STANCE[c.stance.stance] : "No stance", c.subject?.name].filter(Boolean).join(" · ")}</p>
                  </div>
                )) : <p className="text-muted-foreground">None yet. Claims you keep on this book&apos;s subjects, or that cite it, show here.</p>}
              </CardContent>
            </Card>
          ) : null}
          {like.length ? (
            <section className="space-y-4">
              <Label>Books like this</Label>
              <BookGrid books={like} mine={shelved} cols="md:grid-cols-3 lg:grid-cols-6" />
            </section>
          ) : null}
          {byAuthor.length && authorKeys[0] ? (
            <section className="space-y-4">
              <div className="flex items-baseline justify-between gap-3">
                <Label>More by {authors[0]}</Label>
                <Link href={`/learn/authors/${authorKeys[0]}`} className="text-muted-foreground shrink-0 text-sm underline-offset-4 hover:underline">All their books</Link>
              </div>
              <BookGrid books={byAuthor} mine={shelved} cols="md:grid-cols-3 lg:grid-cols-6" />
            </section>
          ) : null}
          <div className="flex flex-wrap gap-2">
            <Button asChild variant="ghost" size="sm"><a href={`https://openlibrary.org/works/${work}`} target="_blank" rel="noreferrer">Open Library<ExternalLinkIcon /></a></Button>
            {google?.link ? <Button asChild variant="ghost" size="sm"><a href={google.link} target="_blank" rel="noreferrer">Google Books<ExternalLinkIcon /></a></Button> : null}
          </div>
        </div>
      </div>
    </main>
  );
}

function Rating({ source, r }: { source: string; r: { average: number; count: number } }) {
  return (
    <span className="inline-flex items-center gap-1">
      <StarIcon className="size-4 fill-current" />
      <span className="font-medium">{r.average.toFixed(1)}</span>
      <span className="text-muted-foreground">from {r.count.toLocaleString("en-US")} on {source}</span>
    </span>
  );
}
