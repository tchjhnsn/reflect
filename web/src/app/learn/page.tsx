import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import { ArrowLeftIcon, XIcon } from "lucide-react";
import { STANCES } from "@reflect/core/claims";
import { browseClass, searchBooks, type CatalogEntry } from "@reflect/core/openlibrary";
import { LCC_CLASSES } from "@reflect/core/shelves";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Segmented } from "@/components/segmented";
import { cn } from "@/lib/utils";
import { requireUser } from "@/server/session";
import { learnView, type LearnView } from "@/server/learn-store";
import { listClaims, type ClaimOut } from "@/server/claims-store";
import { shelvesOf } from "@/server/learn-shelves";
import { addFromLibraryAction, keepClaimAction, setBookStatusAction, setStanceAction } from "./actions";
import { bookTone, subjectTone, type ShelfTone } from "./tones";

export const metadata: Metadata = { title: "Learn" };
export const dynamic = "force-dynamic";

type Tab = "shelves" | "library" | "claims";
const BOOK = "M8 4 h84 v92 h-84 z";
const SUBJ = "M50 4 A46 46 0 1 1 49.99 4 Z";
const STATUS = [["want", "Want to read"], ["reading", "Reading"], ["read", "Read"]] as const;
const openLibrary = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, cache: "no-store" });
const STANCE_LABEL: Record<string, string> = { belief: "Belief", disbelief: "Disbelief", suspension: "Suspension of judgment" };
const TONE: Record<ShelfTone, string> = { known: "fill-foreground", learning: "fill-neutral-400", registered: "fill-neutral-200" };
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";
const Note = ({ children }: { children: ReactNode }) => <p className="text-muted-foreground text-sm">{children}</p>;

/** Learn: your shelves by Library of Congress class, the whole library, and your claims. */
export default async function LearnPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const tab: Tab = one(sp.tab) === "library" ? "library" : one(sp.tab) === "claims" ? "claims" : "shelves";
  const view = await learnView(user.id);
  const href = (t: Tab) => `/learn?tab=${t}`;
  return (
    <main className="mx-auto max-w-5xl px-6 py-6">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Button asChild variant="ghost" size="icon" aria-label="Home"><Link href="/"><ArrowLeftIcon /></Link></Button>
          <h1 className="text-2xl font-semibold tracking-tight">Learn</h1>
        </div>
        <Segmented
          label="Learn"
          items={[
            { label: "My shelves", href: href("shelves"), on: tab === "shelves" },
            { label: "Library", href: href("library"), on: tab === "library" },
            { label: "Claims", href: href("claims"), on: tab === "claims" },
          ]}
        />
      </header>
      <div className="mt-6">
        {tab === "shelves" ? <Shelves view={view} all={one(sp.all) === "1"} open={one(sp.book)} /> : null}
        {tab === "library" ? <Library view={view} cls={one(sp.class).toUpperCase()} q={one(sp.q).trim()} /> : null}
        {tab === "claims" ? <Claims claims={await listClaims(user.id).catch((): ClaimOut[] | null => null)} /> : null}
      </div>
      {tab === "shelves" ? (
        <footer className="text-muted-foreground mt-8 flex flex-wrap justify-center gap-x-6 gap-y-2 text-xs">
          <span>■ Book</span>
          <span>● Subject</span>
          <span>Dark = known · grey = learning · light = registered</span>
          <span>Library of Congress Classification</span>
        </footer>
      ) : null}
    </main>
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
  const items = (books: { id: string; title: string }[], subjects: { id: string; name: string }[]) => (
    <div className="flex flex-wrap gap-4">
      {books.map((b) => {
        const full = byId.get(b.id)!;
        return (
          <Link key={b.id} href={`/learn?tab=shelves&book=${b.id}${all ? "&all=1" : ""}`} className={cn("hover:bg-accent flex w-24 flex-col items-center gap-2 rounded-lg p-2 text-center transition", open === b.id && "bg-accent")} aria-label={`${b.title}, book, ${full.status}`}>
            <Glyph d={BOOK} tone={bookTone(full.status)} className="h-20 w-14 rounded-sm" />
            <span className="line-clamp-2 text-xs font-medium">{b.title}</span>
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
        {sh.shelves.length ? <Note>Books and subjects on their Library of Congress shelves.</Note> : <Note>No shelves yet. Find a book in the <Link className="underline underline-offset-4" href="/learn?tab=library">library</Link> and it goes on its shelf.</Note>}
        <Button asChild variant="ghost" size="sm"><Link href={all ? "/learn?tab=shelves" : "/learn?tab=shelves&all=1"}>{all ? "Only filled shelves" : "All 21 shelves"}</Link></Button>
      </div>
      {sh.shelves.map((s) => shelf(s.id, s.name, Boolean(s.books.length || s.subjects.length), items(s.books, s.subjects)))}
      {sh.unshelved.books.length || sh.unshelved.subjects.length ? shelf("?", "Not yet shelved (no call number)", true, items(sh.unshelved.books, sh.unshelved.subjects)) : null}
      {book ? (
        <>
        <Link href={`/learn?tab=shelves${all ? "&all=1" : ""}`} aria-label="Close" tabIndex={-1} className="fixed inset-0 z-10 bg-black/40" />
        <aside className="bg-background fixed inset-y-0 right-0 z-20 w-full max-w-80 space-y-4 overflow-y-auto border-l p-6 shadow-lg" aria-label={book.title}>
          <div className="flex items-start justify-between gap-2">
            <h2 className="text-lg font-semibold leading-tight">{book.title}</h2>
            <Button asChild variant="ghost" size="icon" aria-label="Close"><Link href={`/learn?tab=shelves${all ? "&all=1" : ""}`}><XIcon /></Link></Button>
          </div>
          <Note>{book.authors.join(", ") || "Author not given"}{book.catalog?.year ? ` · ${book.catalog.year}` : ""}</Note>
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

async function Library({ view, cls, q }: { view: LearnView; cls: string; q: string }) {
  const pick = LCC_CLASSES.find((c) => c.id === cls);
  let books: CatalogEntry[] | null = [];
  let title = "Search the library, or pick a shelf";
  if (q) {
    books = await searchBooks(q, openLibrary, 12).catch(() => null);
    title = `Results for “${q}”`;
  } else if (pick) {
    books = await browseClass(pick.id, openLibrary, { limit: 12 }).catch(() => null);
    title = `${pick.id} · ${pick.name} · best rated`;
  }
  const kept = new Set(view.books.map((b) => b.catalog?.work).filter(Boolean));
  return (
    <section className="grid gap-6 md:grid-cols-[16rem_1fr]">
      <nav className="flex flex-col gap-0.5" aria-label="Shelves">
        {LCC_CLASSES.map((c) => (
          <Link key={c.id} href={`/learn?tab=library&class=${c.id}`} aria-current={c.id === pick?.id ? "page" : undefined} className={cn("hover:bg-accent flex gap-3 rounded-md px-3 py-1.5 text-sm transition", c.id === pick?.id && "bg-accent font-medium")}>
            <span className="text-muted-foreground w-4">{c.id}</span><span className="truncate">{c.name}</span>
          </Link>
        ))}
      </nav>
      <div className="min-w-0 space-y-4">
        <form action="/learn" className="space-y-3">
          <input type="hidden" name="tab" value="library" />
          <h2 className="font-semibold">{title}</h2>
          <label htmlFor="q" className="sr-only">Search titles or authors</label>
          <Input id="q" name="q" defaultValue={q} placeholder="Search titles or authors..." />
        </form>
        {books === null ? <Note>The open library did not answer. Try again in a moment.</Note> : null}
        {books && (q || pick) && !books.length ? <Note>Nothing found.</Note> : null}
        {books?.length ? (
          <Card className="gap-0 py-0">
            {books.map((b, i) => (
              <div key={b.work} className={cn("flex items-center gap-4 px-4 py-3", i > 0 && "border-t")}>
                <Glyph d={BOOK} tone={kept.has(b.work) ? "known" : "registered"} className="h-11 w-8 shrink-0 rounded-sm" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{b.title}</p>
                  <p className="text-muted-foreground truncate text-xs">{b.authors.join(", ") || "Author not given"}{b.year ? ` · ${b.year}` : ""}</p>
                </div>
                {kept.has(b.work) ? (
                  <Badge variant="secondary">On your shelves</Badge>
                ) : (
                  <form action={addFromLibraryAction}>
                    <input type="hidden" name="entry" value={JSON.stringify(b)} />
                    <input type="hidden" name="status" value="want" />
                    <Button size="sm">Add to shelves</Button>
                  </form>
                )}
              </div>
            ))}
          </Card>
        ) : null}
        <p className="text-muted-foreground text-xs">From Open Library</p>
      </div>
    </section>
  );
}

function Claims({ claims }: { claims: ClaimOut[] | null }) {
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
                    <p className="text-muted-foreground text-xs">{[c.subject?.name, c.holders.map((h) => (h.citation ? `${h.name} (${h.citation})` : h.name)).join("; ")].filter(Boolean).join(" · ") || "No holder named"}</p>
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
