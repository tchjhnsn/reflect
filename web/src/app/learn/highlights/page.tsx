import type { Metadata } from "next";
import Link from "next/link";
import { DownloadIcon, SearchIcon } from "lucide-react";
import { coverUrl } from "@reflect/core/openlibrary";
import { HIGHLIGHT_COLORS, isHighlightColor } from "@reflect/core/excerpts";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { requireUser } from "@/server/session";
import { allExcerpts } from "@/server/excerpt-store";
import { highlightBooks } from "@/server/highlights";
import { publishedBy } from "@/server/published-store";
import { LearnHeader } from "../header";
import { ExcerptItem } from "../excerpt-item";
import { COLOR } from "../colors";
import { KindleImport } from "./kindle";
import { ToJournal } from "./to-journal";

export const metadata: Metadata = { title: "Highlights" };
export const dynamic = "force-dynamic";

const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/**
 * Every highlight and excerpt the person keeps, from every book, grouped by book (the book with the newest first)
 * and in reading order within it; searchable by words, note or book, and narrowed by colour. Exported as Markdown,
 * sent into the journal, or added to from a Kindle's My Clippings.txt.
 */
export default async function HighlightsPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const q = one(sp.q).trim().slice(0, 100);
  const color = one(sp.color);
  const [excerpts, published] = await Promise.all([allExcerpts(user.id), publishedBy(user.id).catch(() => [])]);
  const books = await highlightBooks(user.id, excerpts);
  const low = q.toLowerCase();
  const shown = books
    .map((b) => ({ ...b, excerpts: b.excerpts.filter((e) => (!isHighlightColor(color) || e.color === color) && (!low || e.quote.toLowerCase().includes(low) || (e.note ?? "").toLowerCase().includes(low) || b.title.toLowerCase().includes(low) || b.author.toLowerCase().includes(low))) }))
    .filter((b) => b.excerpts.length);
  const count = shown.reduce((n, b) => n + b.excerpts.length, 0);
  const href = (c: string) => `/learn/highlights?${new URLSearchParams({ ...(q ? { q } : {}), ...(c ? { color: c } : {}) })}`;

  return (
    <>
      <LearnHeader on="highlights" />
      <main className="mx-auto w-full max-w-3xl px-4 pt-4 pb-24 sm:px-6">
        <div className="flex flex-wrap items-center gap-2">
          <Button asChild variant="outline" size="sm"><a href="/learn/highlights/export"><DownloadIcon />Export Markdown</a></Button>
          <ToJournal />
          <KindleImport />
        </div>
        <form className="mt-4 flex gap-2" role="search">
          {isHighlightColor(color) ? <input type="hidden" name="color" value={color} /> : null}
          <Input name="q" defaultValue={q} placeholder="Search your highlights" aria-label="Search your highlights" className="h-9" />
          <Button type="submit" variant="outline" size="sm" className="h-9" aria-label="Search"><SearchIcon /></Button>
        </form>
        <p className="text-muted-foreground mt-2 text-xs"><Link href={`/learn/passages${q ? `?q=${encodeURIComponent(q)}` : ""}`} className="underline underline-offset-2">Search inside the books themselves</Link>, not only your highlights.</p>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
          <Link href={href("")} className={cn("rounded-full border px-3 py-1", !isHighlightColor(color) && "bg-muted")}>All colours</Link>
          {HIGHLIGHT_COLORS.map((c) => (
            <Link key={c} href={href(c)} className={cn("inline-flex items-center gap-1.5 rounded-full border px-3 py-1", color === c && "bg-muted")}><span className={cn("size-2.5 rounded-full", COLOR[c].dot)} />{COLOR[c].name}</Link>
          ))}
        </div>
        <p className="text-muted-foreground mt-4 text-sm">{excerpts.length === 0 ? "Highlight a passage while you read, add an excerpt on a book's page, or import your Kindle highlights, and it shows here." : `${count} ${count === 1 ? "excerpt" : "excerpts"} from ${shown.length} ${shown.length === 1 ? "book" : "books"}${q || isHighlightColor(color) ? ` (of ${excerpts.length})` : ""}.`}</p>
        <div className="mt-6 space-y-10">
          {shown.map((b) => {
            const cover = coverUrl(b, "S");
            return (
              <section key={b.work} className="space-y-3">
                <Link href={`/learn/works/${b.work}`} className="flex items-center gap-3">
                  {cover ? <img src={cover} alt="" className="h-14 w-10 rounded-sm object-cover" /> : <span className="bg-muted h-14 w-10 rounded-sm" />}
                  <span>
                    <span className="block font-serif text-lg leading-tight">{b.title}</span>
                    <span className="text-muted-foreground text-sm">{b.author}</span>
                  </span>
                </Link>
                <ul className="space-y-4">
                  {b.excerpts.map((e) => (
                    <ExcerptItem key={e.id} reader={`/learn/works/${b.work}/read`} row={{ id: e.id, work: b.work, title: b.title, author: b.author, quote: e.quote, where: e.where ?? null, chapter: e.anchor?.chapter ?? null, note: e.note, color: e.color, published: published.find((p) => p.work === b.work && p.quote === e.quote.replace(/\s+/g, " ").trim())?.id ?? null }} />
                  ))}
                </ul>
              </section>
            );
          })}
        </div>
      </main>
    </>
  );
}
