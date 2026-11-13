import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeftIcon, ChevronLeftIcon, ChevronRightIcon } from "lucide-react";
import { gutenbergBook } from "@reflect/core/gutenberg";
import { workAccess, workEntry } from "@reflect/core/openlibrary";
import { Button } from "@/components/ui/button";
import { requireUser } from "@/server/session";
import { Menu } from "../../../browse/category-menu";
import { Page } from "./page-text";

export const metadata: Metadata = { title: "Reading" };
export const dynamic = "force-dynamic";

const cached = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, next: { revalidate: 86400 } });
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/**
 * A public-domain book read inside Reflect, one chapter at a time, from Project Gutenberg's text. Where you
 * are is kept in this browser; a passage you select can be kept as a claim, cited to the book and chapter.
 */
export default async function ReadPage({ params, searchParams }: { params: Promise<{ work: string }>; searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  await requireUser();
  const { work } = await params;
  if (!/^OL\d+W$/.test(work)) notFound();
  const [entry, access] = await Promise.all([workEntry(work, cached).catch(() => null), workAccess(work, cached).catch(() => null)]);
  const id = access?.gutenberg[0];
  if (!entry || !id) notFound();
  const book = await gutenbergBook(id, cached).catch(() => null);
  if (!book) notFound();
  const n = Math.min(Math.max(Number(one((await searchParams).c)) || 1, 1), book.chapters.length);
  const chapter = book.chapters[n - 1]!;
  const href = (c: number) => `/learn/works/${work}/read?c=${c}`;

  return (
    <main className="mx-auto w-full max-w-3xl px-4 pt-6 pb-24 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Button asChild variant="ghost" size="sm"><Link href={`/learn/works/${work}`}><ArrowLeftIcon />{entry.title}</Link></Button>
        <Menu label="Chapter" items={book.chapters.map((c, i) => ({ href: href(i + 1), label: c.title.length > 60 ? `${c.title.slice(0, 57)}…` : c.title, on: i + 1 === n }))} />
      </div>
      <p className="text-muted-foreground mt-8 text-center text-xs tracking-wide uppercase">{entry.authors[0] ?? ""}{entry.authors[0] ? " · " : ""}{n} of {book.chapters.length}</p>
      <h1 className="mt-2 text-center font-serif text-3xl font-medium">{chapter.title}</h1>
      <Page work={work} chapter={n} chapters={book.chapters.length} title={entry.title} author={entry.authors[0] ?? ""} chapterTitle={chapter.title} paragraphs={chapter.paragraphs} />
      <nav className="mt-12 flex items-center justify-between gap-3 border-t pt-6">
        {n > 1 ? <Button asChild variant="outline"><Link href={href(n - 1)}><ChevronLeftIcon />Previous</Link></Button> : <span />}
        {n < book.chapters.length ? <Button asChild variant="outline"><Link href={href(n + 1)}>Next<ChevronRightIcon /></Link></Button> : <Button asChild variant="outline"><Link href={`/learn/works/${work}`}>Back to the book</Link></Button>}
      </nav>
      <p className="text-muted-foreground mt-8 text-center text-xs">Text from <a href={`https://www.gutenberg.org/ebooks/${id}`} target="_blank" rel="noreferrer" className="underline underline-offset-2">Project Gutenberg</a>, in the public domain in the United States.</p>
    </main>
  );
}
