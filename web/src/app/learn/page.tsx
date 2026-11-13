import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import { redirect } from "next/navigation";
import { XIcon } from "lucide-react";
import { STANCES } from "@reflect/core/claims";
import { coverUrl } from "@reflect/core/openlibrary";
import { categoryOf, showCall, themaCategory } from "@reflect/core/thema";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { PAGE } from "@/components/page-header";
import { LearnHeader } from "./header";
import { cn } from "@/lib/utils";
import { requireUser } from "@/server/session";
import { learnView, type LearnView } from "@/server/learn-store";
import { listClaims, type ClaimOut } from "@/server/claims-store";
import { claimWorks } from "@/server/book-store";
import { shelvesOf } from "@/server/learn-shelves";
import { keepClaimAction, setBookStatusAction, setStanceAction } from "./actions";
import { bookTone, subjectTone, type ShelfTone } from "./tones";

export const metadata: Metadata = { title: "Learn" };
export const dynamic = "force-dynamic";

type Tab = "shelves" | "claims";
const BOOK = "M8 4 h84 v92 h-84 z";
const SUBJ = "M50 4 A46 46 0 1 1 49.99 4 Z";
const STATUS = [["want", "Want to read"], ["reading", "Reading"], ["read", "Read"]] as const;
const STANCE_LABEL: Record<string, string> = { belief: "Belief", disbelief: "Disbelief", suspension: "Suspension of judgment" };
const TONE: Record<ShelfTone, string> = { known: "fill-foreground", learning: "fill-neutral-400", registered: "fill-neutral-200" };
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";
const Note = ({ children }: { children: ReactNode }) => <p className="text-muted-foreground text-sm">{children}</p>;

/** Learn: your shelves by Thema section, and your claims. Browsing the library is its own page. */
export default async function LearnPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  // Browsing moved to its own page (/learn/browse); old links still land there.
  if (one(sp.tab) === "library") redirect(`/learn/browse${one(sp.q) ? `?q=${encodeURIComponent(one(sp.q))}` : ""}`);
  const tab: Tab = one(sp.tab) === "claims" ? "claims" : "shelves";
  const view = await learnView(user.id);
  return (
    <main className={PAGE}>
      <LearnHeader on={tab} />
      <div className="mt-6">
        {tab === "shelves" ? <Shelves view={view} all={one(sp.filled) !== "1"} open={one(sp.book)} /> : null}
        {tab === "claims" ? <Claims claims={await listClaims(user.id).catch((): ClaimOut[] | null => null)} works={await claimWorks(user.id).catch(() => new Map<string, { workId: string; title: string }>())} /> : null}
      </div>
      {tab === "shelves" ? (
        <footer className="text-muted-foreground mt-8 flex flex-wrap justify-center gap-x-6 gap-y-2 text-xs">
          <span>■ Book</span>
          <span>● Subject</span>
          <span>Dark = known · grey = learning · light = registered</span>
          <span>Thema subject categories</span>
        </footer>
      ) : null}
    </main>
  );
}

/** How far through a book the person is, by chapter and how far down it, read in Reflect. */
function Progress({ place, className }: { place: { chapter: number; chapters: number; at: number }; className: string }) {
  const share = Math.min(1, (place.chapter - 1 + place.at) / place.chapters);
  return (
    <span className={cn("bg-muted block h-1 overflow-hidden rounded-full", className)} role="progressbar" aria-label={`Chapter ${place.chapter} of ${place.chapters}`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(share * 100)}>
      <span className="bg-foreground block h-full" style={{ width: `${Math.max(2, share * 100)}%` }} />
    </span>
  );
}

function Glyph({ d, tone, className }: { d: string; tone: ShelfTone; className: string }) {
  return <svg className={className} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d={d} className={TONE[tone]} /></svg>;
}

function Shelves({ view, all, open }: { view: LearnView; all: boolean; open: string }) {
  const sh = shelvesOf(view, all);
  const byId = new Map(view.books.map((b) => [b.id, b]));
  const subjectBooks = new Map<string, LearnView["books"]>();
  for (const b of view.books) for (const s of b.subjects) subjectBooks.set(s.id, [...(subjectBooks.get(s.id) ?? []), b]);
  const grasp = new Map(view.books.flatMap((b) => b.subjects).concat(view.known).map((s) => [s.id, s.grasp?.grasp ?? null]));
  const book = open ? byId.get(open) : undefined;
  const items = (books: { id: string; title: string; category?: string }[], subjects: { id: string; name: string }[]) => (
    <div className="flex flex-wrap gap-4">
      {books.map((b) => {
        const full = byId.get(b.id)!;
        return (
          <Link key={b.id} href={`/learn?book=${b.id}${all ? "" : "&filled=1"}`} className={cn("hover:bg-accent flex w-24 flex-col items-center gap-2 rounded-lg p-2 text-center transition", open === b.id && "bg-accent")} aria-label={`${b.title}, book, ${full.status}`}>
            {full.catalog && coverUrl(full.catalog, "S") ? (
              <span className="flex flex-col overflow-hidden rounded-sm border">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={coverUrl(full.catalog, "M")!} alt="" loading="lazy" className="h-20 w-14 object-cover" />
                <Glyph d={BOOK} tone={bookTone(full.status)} className="h-1.5 w-14" />
              </span>
            ) : <Glyph d={BOOK} tone={bookTone(full.status)} className="h-20 w-14 rounded-sm" />}
            {full.place && full.status !== "read" ? <Progress place={full.place} className="w-14" /> : null}
            <span className="line-clamp-2 text-xs font-medium">{b.title}</span>
            {b.category ? <span className="text-muted-foreground -mt-1 line-clamp-1 text-[11px]">{themaCategory(b.category)?.short}</span> : null}
          </Link>
        );
      })}
      {subjects.map((s) => (
        <span key={s.id} className="flex w-24 flex-col items-center gap-2 p-2 text-center" aria-label={`${s.name}, subject`}>
          <Glyph d={SUBJ} tone={subjectTone(grasp.get(s.id), (subjectBooks.get(s.id) ?? []).map((b) => b.status))} className="size-20" />
          <span className="line-clamp-2 text-xs">{s.name}</span>
        </span>
      ))}
    </div>
  );
  const shelf = (id: string, name: string, filled: boolean, body: ReactNode) => (
    <Card key={id} className="gap-4 py-5">
      <CardHeader className="px-5">
        <CardTitle className="flex items-center gap-3 text-sm">
          <Badge variant={filled ? "default" : "outline"} className="size-6 justify-center rounded-md p-0">{id}</Badge>
          {name}
        </CardTitle>
      </CardHeader>
      {filled ? <CardContent className="px-5">{body}</CardContent> : null}
    </Card>
  );
  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        {sh.shelves.length ? <Note>Books and subjects in their Thema sections. Each book keeps its Library of Congress call number as its exact address.</Note> : <Note>No books yet. <Link className="underline underline-offset-4" href="/learn/browse">Browse books</Link> and each one you add goes on its shelf.</Note>}
        <Button asChild variant="ghost" size="sm"><Link href={all ? "/learn?filled=1" : "/learn"}>{all ? "Only filled shelves" : "All 20 shelves"}</Link></Button>
      </div>
      {sh.shelves.map((s) => shelf(s.id, s.name, Boolean(s.books.length || s.subjects.length), items(s.books, s.subjects)))}
      {sh.unshelved.books.length || sh.unshelved.subjects.length ? shelf("?", "Not yet placed (the catalog gives no category)", true, items(sh.unshelved.books, sh.unshelved.subjects)) : null}
      {book ? (
        <>
        <Link href={`/learn${all ? "" : "?filled=1"}`} aria-label="Close" tabIndex={-1} className="fixed inset-0 z-10 bg-black/40" />
        <aside className="bg-background fixed inset-y-0 right-0 z-20 w-full max-w-80 space-y-4 overflow-y-auto border-l p-6 shadow-lg" aria-label={book.title}>
          <div className="flex items-start justify-between gap-2">
            <h2 className="text-lg font-semibold leading-tight">{book.title}</h2>
            <Button asChild variant="ghost" size="icon" aria-label="Close"><Link href={`/learn${all ? "" : "?filled=1"}`}><XIcon /></Link></Button>
          </div>
          <Note>{book.authors.join(", ") || "Author not given"}{book.catalog?.year ? ` · ${book.catalog.year}` : ""}</Note>
          {book.catalog ? (
            <p className="text-muted-foreground text-xs">
              {themaCategory(categoryOf(book.catalog) ?? "")?.heading ?? "Not yet placed"}
              {book.catalog.lcc?.[0] ? <> · Call number {showCall(book.catalog.lcc[0])}</> : null}
            </p>
          ) : null}
          {book.place && book.catalog?.work ? (
            <div className="space-y-2">
              <Progress place={book.place} className="w-full" />
              <Button asChild size="sm"><Link href={`/learn/works/${book.catalog.work}/read?c=${book.place.chapter}`}>Continue reading · {book.place.chapter} of {book.place.chapters}</Link></Button>
            </div>
          ) : null}
          {book.catalog?.work ? <Button asChild variant="outline" size="sm"><Link href={`/learn/works/${book.catalog.work}`}>Open the book page</Link></Button> : null}
          <form action={setBookStatusAction} className="flex flex-wrap gap-2">
            <input type="hidden" name="book" value={book.id} />
            {STATUS.map(([id, label]) => (
              <Button key={id} size="sm" name="status" value={id} variant={book.status === id ? "default" : "outline"} aria-pressed={book.status === id}>{label}</Button>
            ))}
          </form>
          {book.subjects.length ? (
            <div className="space-y-2">
              <p className="text-xs font-medium">Subjects</p>
              <div className="flex flex-wrap gap-1.5">{book.subjects.map((s) => <Badge key={s.id} variant="secondary">{s.name}</Badge>)}</div>
            </div>
          ) : null}
          {book.catalog?.subjects.length ? (
            <div className="space-y-2">
              <p className="text-muted-foreground text-xs font-medium">The catalog suggests</p>
              <div className="flex flex-wrap gap-1.5">{book.catalog.subjects.slice(0, 8).map((s) => <Badge key={s} variant="outline">{s}</Badge>)}</div>
            </div>
          ) : null}
        </aside>
        </>
      ) : null}
    </section>
  );
}

function Claims({ claims, works }: { claims: ClaimOut[] | null; works: Map<string, { workId: string; title: string }> }) {
  if (!claims) return <Note>Claims are not set up on this database yet.</Note>;
  return (
    <section className="space-y-4">
      <Note>What you believe, disbelieve, or suspend judgment on, with who holds each view. Your latest stance counts; earlier ones are kept.</Note>
      {STANCES.map((st) => {
        const here = claims.filter((c) => c.stance?.stance === st.id);
        return (
          <Card key={st.id} className="gap-3 py-5">
            <CardHeader className="px-5">
              <CardTitle className="flex items-center gap-2 text-sm">{STANCE_LABEL[st.id]}<Badge variant="secondary">{here.length}</Badge></CardTitle>
            </CardHeader>
            <CardContent className="space-y-0 px-5">
              {here.length ? null : <Note>None</Note>}
              {here.map((c, i) => (
                <div key={c.id} className={cn("flex flex-wrap items-center justify-between gap-3 py-3", i > 0 && "border-t")}>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium">{c.text}</p>
                    <p className="text-muted-foreground text-xs">{[c.subject?.name, c.holders.map((h) => (h.citation ? `${h.name} (${h.citation})` : h.name)).join("; ")].filter(Boolean).join(" · ") || "No holder named"}
                      {works.get(c.id) ? <> · from <Link href={`/journal/w/${works.get(c.id)!.workId}`} className="underline underline-offset-2">{works.get(c.id)!.title}</Link></> : null}</p>
                  </div>
                  <form action={setStanceAction} className="flex flex-wrap gap-2">
                    <input type="hidden" name="claim" value={c.id} />
                    {STANCES.filter((x) => x.id !== st.id).map((x) => <Button key={x.id} size="sm" variant="outline" name="stance" value={x.id}>{STANCE_LABEL[x.id]}</Button>)}
                  </form>
                </div>
              ))}
            </CardContent>
          </Card>
        );
      })}
      <Card className="py-5">
        <CardHeader className="px-5"><CardTitle className="text-sm">Keep a claim</CardTitle></CardHeader>
        <CardContent className="px-5">
          <form action={keepClaimAction} className="space-y-4">
            <label className="block space-y-1.5 text-sm font-medium">Claim<Input name="claim" required maxLength={500} /></label>
            <label className="block space-y-1.5 text-sm font-medium">Subject (optional)<Input name="subject" maxLength={120} /></label>
            <label className="block space-y-1.5 text-sm font-medium">Who holds it, one per line: name | citation<Textarea name="holders" rows={2} /></label>
            <div className="flex flex-wrap gap-2">
              {STANCES.map((x) => <Button key={x.id} size="sm" variant="outline" name="stance" value={x.id}>Keep as {(STANCE_LABEL[x.id] ?? x.id).toLowerCase()}</Button>)}
            </div>
          </form>
        </CardContent>
      </Card>
    </section>
  );
}
