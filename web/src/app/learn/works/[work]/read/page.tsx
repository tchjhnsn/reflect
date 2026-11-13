import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeftIcon, ChevronLeftIcon, ChevronRightIcon, SearchIcon } from "lucide-react";
import { workAccess, workEntry } from "@reflect/core/openlibrary";
import { carryPlace, findSpans, searchBook, wordsOf } from "@reflect/core/reading";
import { locate } from "@reflect/core/excerpts";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { readText, textSource } from "@/server/texts";
import { excerptsOf } from "@/server/excerpt-store";
import { Page, type Mark } from "./page-text";
import { Contents, ReaderFrame, ReaderSettings, TimeLeft } from "./reader-frame";
import { ListenAlong } from "./listen-along";
import { audiobookFor } from "@/server/audio";
import { gutenbergEditions } from "@/server/free";
import { Menu } from "../../../browse/category-menu";
import { matchSections } from "@reflect/core/listening";

export const metadata: Metadata = { title: "Reading" };
export const dynamic = "force-dynamic";

const cached = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 86400 } });
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/**
 * A public-domain book read inside Reflect, one chapter at a time, from its Standard Ebooks or Project Gutenberg
 * text, in the person's font, size, spacing and page colour. Where you are is kept in this browser, and with your
 * account when the book is on your shelves; a passage you select can be highlighted, copied as a cited quote, kept
 * as a claim, or (one word) looked up. `?q=` searches the whole book; a result opens its chapter at the paragraph
 * with the phrase marked.
 */
export default async function ReadPage({ params, searchParams }: { params: Promise<{ work: string }>; searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const { work } = await params;
  if (!/^OL\d+W$/.test(work)) notFound();
  const sp = await searchParams;
  const [entry, access, view] = await Promise.all([workEntry(work, cached).catch(() => null), workAccess(work, cached).catch(() => null), learnView(user.id)]);
  // The book on the person's shelves, when it is there: its place is kept with their account and shown on the shelf.
  const shelved = view.books.find((b) => b.catalog?.work === work);
  if (!entry) notFound();
  // A book kept in several translations: each one, so the person can pick one or read two side by side.
  const base = await textSource(entry, access);
  const editions = gutenbergEditions(entry, base?.kind === "gutenberg" ? base.id : undefined);
  const picked = editions.find((e) => e.id === one(sp.t));
  const source = picked ? { kind: "gutenberg" as const, id: picked.id } : base;
  const besideId = editions.find((e) => e.id === one(sp.side) && e.id !== source?.id)?.id;
  // A Standard Ebooks edition that cannot be read falls back to the Gutenberg text the catalog links.
  const [book, audio, beside] = await Promise.all([
    source ? readText(source, access?.gutenberg[0] ? { kind: "gutenberg", id: access.gutenberg[0] } : null) : Promise.resolve(null),
    audiobookFor(entry, access),
    besideId ? readText({ kind: "gutenberg", id: besideId }) : Promise.resolve(null),
  ]);
  if (!book) notFound();
  const total = book.chapters.length;
  // A place kept against another text of the book (another chapter count) is carried to this one.
  const kept = shelved?.place ? carryPlace(shelved.place, total) : null;
  const asked = Number(one(sp.c));
  const n = Math.min(Math.max(asked || kept?.chapter || 1, 1), total);
  const chapter = book.chapters[n - 1]!;
  const q = one(sp.q).slice(0, 100);
  const find = one(sp.find).slice(0, 100);
  const goTo = sp.p !== undefined ? Number(one(sp.p)) : undefined;
  const words = book.chapters.map((c) => wordsOf(c.paragraphs));
  // The person's highlights in this chapter, at their kept place or found again by their words.
  const marks: Mark[] = (await excerptsOf(user.id, work).catch(() => [])).flatMap((e) => {
    if (!e.anchor || e.anchor.chapter !== n) return [];
    const s = locate(chapter.paragraphs, e.anchor, e.quote);
    return s ? [{ id: e.id, quote: e.quote, color: e.color, note: e.note, ...s }] : [];
  });
  const keep = new URLSearchParams({ ...(picked ? { t: picked.id } : {}), ...(beside && besideId ? { side: besideId } : {}) });
  const href = (c: number) => `/learn/works/${work}/read?c=${c}${keep.size ? `&${keep}` : ""}`;
  const label = (id: string) => {
    const e = editions.find((x) => x.id === id);
    return e?.translators ? `Translated by ${e.translators}` : `Project Gutenberg text ${id}`;
  };
  const current = book.source.kind === "standard" ? "Standard Ebooks edition" : label(book.source.id);
  const withT = (p: Record<string, string>) => `/learn/works/${work}/read?${new URLSearchParams({ c: String(n), ...p })}`;
  // The other translation's part for this chapter: the one called the same ("Book II"), else the one in the same place.
  const otherAt = beside ? (matchSections(book.chapters.map((c) => c.title), beside.chapters.map((c) => c.title))[n - 1] ?? Math.min(n, beside.chapters.length) - 1) : -1;
  const otherChapter = beside ? beside.chapters[otherAt] : undefined;
  const notes = chapter.notes ?? [];
  const noteText = Object.fromEntries(notes.map((x) => [x.id, book.notes[x.id] ?? ""]));
  const results = q ? searchBook(book.chapters, q) : null;
  // Which recorded part of the free audiobook reads each chapter, where they line up.
  const map = audio ? matchSections(book.chapters.map((c) => c.title), audio.chapters.map((c) => c.title)) : [];

  return (
    <ReaderFrame>
      <main className={`mx-auto w-full ${beside ? "max-w-6xl" : "max-w-3xl"} px-4 pt-6 pb-24 sm:px-6`}>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Button asChild variant="ghost" size="sm" className="max-w-[50%]"><Link href={`/learn/works/${work}`}><ArrowLeftIcon /><span className="truncate">{entry.title}</span></Link></Button>
          <div className="flex items-center gap-1">
            {audio ? <ListenAlong book={audio} work={work} title={entry.title} map={map} chapter={n} /> : null}
            <Contents work={work} current={n} suffix={keep.size ? `&${keep}` : ""} chapters={book.chapters.map((c, i) => ({ title: c.title, words: words[i]! }))} />
            <ReaderSettings />
          </div>
        </div>
        <form action={`/learn/works/${work}/read`} className="mt-4 flex gap-2" role="search">
          <input type="hidden" name="c" value={n} />
          {picked ? <input type="hidden" name="t" value={picked.id} /> : null}
          <Input name="q" defaultValue={q} placeholder="Search in this book" aria-label="Search in this book" className="h-9" />
          <Button type="submit" variant="outline" size="sm" className="h-9" aria-label="Search"><SearchIcon /></Button>
        </form>
        {editions.length > 1 || (editions.length === 1 && book.source.kind === "standard") ? (
          <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
            <span className="text-muted-foreground">Translation:</span>
            <Menu label="Translation" shown={current} items={[...(book.source.kind === "standard" ? [{ href: withT({}), label: "Standard Ebooks edition", on: true }] : []), ...editions.map((e) => ({ href: withT({ t: e.id }), label: label(e.id), on: book.source.kind === "gutenberg" && book.source.id === e.id }))]} />
            <Menu label="Side by side" shown={beside && besideId ? `Beside: ${label(besideId)}` : "Read two side by side"} items={[...(beside ? [{ href: withT(picked ? { t: picked.id } : {}), label: "One text only", on: false }] : []), ...editions.filter((e) => e.id !== book.source.id).map((e) => ({ href: withT({ ...(picked ? { t: picked.id } : {}), side: e.id }), label: label(e.id), on: e.id === besideId }))]} />
          </div>
        ) : null}
        {results ? (
          <section className="mt-8">
            <p className="text-muted-foreground text-sm">{results.total === 0 ? `“${q}” is not in this book.` : `“${q}” occurs ${results.total} ${results.total === 1 ? "time" : "times"}${results.total > results.hits.length ? `; the first ${results.hits.length} are shown` : ""}.`} <Link href={href(n)} className="underline underline-offset-2">Back to the chapter</Link></p>
            <ol className="mt-4 space-y-3">
              {results.hits.map((h, i) => (
                <li key={i}>
                  <Link href={`/learn/works/${work}/read?c=${h.chapter}&find=${encodeURIComponent(q)}&p=${h.p}`} className="hover:bg-muted block rounded-md border p-3">
                    <span className="text-muted-foreground block text-xs">{book.chapters[h.chapter - 1]!.title}</span>
                    <span className="mt-1 block font-serif">{h.before}<mark className="rounded-sm bg-sky-200/80 text-inherit">{h.match}</mark>{h.after}</span>
                  </Link>
                </li>
              ))}
            </ol>
          </section>
        ) : (
          <>
            <p className="text-muted-foreground mt-8 text-center text-xs tracking-wide uppercase">{entry.authors[0] ?? ""}{entry.authors[0] ? " · " : ""}{n} of {total}</p>
            <h1 className="mt-2 text-center font-serif text-3xl font-medium">{chapter.title}</h1>
            <p className="text-muted-foreground mt-2 text-center text-xs"><TimeLeft words={words} chapter={n} /></p>
            <div className={beside ? "grid gap-10 lg:grid-cols-2" : ""}>
            <div className="min-w-0">
            {beside ? <p className="text-muted-foreground mt-6 text-xs">{current}</p> : null}
            <Page
              work={work}
              chapter={n}
              chapters={total}
              title={entry.title}
              author={entry.authors[0] ?? ""}
              chapterTitle={chapter.title}
              paragraphs={chapter.paragraphs}
              source={`${book.source.kind}:${book.source.id}`}
              marks={marks}
              notes={notes}
              noteText={noteText}
              found={find ? findSpans(chapter.paragraphs, find) : []}
              {...(goTo !== undefined && Number.isInteger(goTo) ? { goTo } : {})}
              {...(n < total ? { next: href(n + 1) } : {})}
              {...(shelved ? { bookId: shelved.id, kept: kept && kept.chapter === n ? kept : null } : {})}
            />
            </div>
            {beside && besideId ? (
              <div className="min-w-0 border-t pt-6 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-10">
                <p className="text-muted-foreground mt-6 text-xs">{label(besideId)}{otherChapter ? ` · ${otherChapter.title}` : ""}</p>
                <div className="reading mt-10 space-y-[0.9em] text-pretty">{(otherChapter?.paragraphs ?? []).map((p, i) => <p key={i}>{p}</p>)}</div>
              </div>
            ) : null}
            </div>
            <nav className="mt-12 flex items-center justify-between gap-3 border-t pt-6">
              {n > 1 ? <Button asChild variant="outline"><Link href={href(n - 1)}><ChevronLeftIcon />Previous</Link></Button> : <span />}
              {n < total ? <Button asChild variant="outline"><Link href={href(n + 1)}>Next<ChevronRightIcon /></Link></Button> : <Button asChild variant="outline"><Link href={`/learn/works/${work}`}>Back to the book</Link></Button>}
            </nav>
          </>
        )}
        <p className="text-muted-foreground mt-8 text-center text-xs">Text from <a href={book.credit.href} target="_blank" rel="noreferrer" className="underline underline-offset-2">{book.credit.name}</a>, in the public domain in the United States.</p>
      </main>
    </ReaderFrame>
  );
}
