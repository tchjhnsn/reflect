import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import { redirect } from "next/navigation";
import { XIcon } from "lucide-react";
import { STANCES } from "@reflect/core/claims";
import { coverUrl } from "@reflect/core/openlibrary";
import { categoryOf, showCall, themaCategory } from "@reflect/core/thema";
import { LEISURE, MEDIA, WAYS, bookWay, mediumOf, placeOf, type Way } from "@reflect/core/ways";
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
import { listItems, type LearnItem } from "@/server/ways-store";
import { addItemAction, changeItemAction, keepClaimAction, removeItemAction, setBookStatusAction, setBookWayAction, setStanceAction } from "./actions";
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

/** Learn: your shelves by the ways of learning (books by Thema section within them), and your claims. Browsing the library is its own page. */
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
        {tab === "shelves" ? <Shelves view={view} items={await listItems(user.id).catch((): LearnItem[] | null => null)} all={one(sp.filled) !== "1"} open={one(sp.book)} openItem={one(sp.item)} /> : null}
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

const SELECT = "border-input bg-background h-9 w-full rounded-md border px-3 text-sm";
const statusLabel = (i: LearnItem) => (i.status === "want" ? "Want to" : i.status === "doing" ? mediumOf(i.medium).doing : mediumOf(i.medium).done);

/**
 * The shelves, arranged by Aristotle's ways of learning: teaching, dialectic, imitation and practice. A book goes
 * by the way the person chose, else by its Thema category (stories under imitation); teaching keeps the Thema
 * shelves and the subjects. Lectures, debates, films, music, games and the rest are items with their medium shown;
 * a game, music or art without a lesson is kept under Rest and leisure.
 */
function Shelves({ view, items, all, open, openItem }: { view: LearnView; items: LearnItem[] | null; all: boolean; open: string; openItem: string }) {
  const byId = new Map(view.books.map((b) => [b.id, b]));
  const subjectBooks = new Map<string, LearnView["books"]>();
  for (const b of view.books) for (const s of b.subjects) subjectBooks.set(s.id, [...(subjectBooks.get(s.id) ?? []), b]);
  const grasp = new Map(view.books.flatMap((b) => b.subjects).concat(view.known).map((s) => [s.id, s.grasp?.grasp ?? null]));
  const book = open ? byId.get(open) : undefined;
  const item = openItem ? items?.find((i) => i.id === openItem) : undefined;
  const keep = all ? "" : "&filled=1";
  const close = `/learn${all ? "" : "?filled=1"}`;
  const wayOf = (b: LearnView["books"][number]): Way => b.way ?? bookWay(b.catalog ? categoryOf(b.catalog) : null);
  const row = (books: { id: string; title: string; category?: string }[], subjects: { id: string; name: string }[]) => (
    <div className="flex flex-wrap gap-4">
      {books.map((b) => {
        const full = byId.get(b.id)!;
        return (
          <Link key={b.id} href={`/learn?book=${b.id}${keep}`} className={cn("hover:bg-accent flex w-24 flex-col items-center gap-2 rounded-lg p-2 text-center transition", open === b.id && "bg-accent")} aria-label={`${b.title}, book, ${full.status}`}>
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
  const shelf = (key: string, id: string, name: string, filled: boolean, body: ReactNode) => (
    <Card key={key} className="gap-4 py-5">
      <CardHeader className="px-5">
        <CardTitle className="flex items-center gap-3 text-sm">
          <Badge variant={filled ? "default" : "outline"} className="size-6 justify-center rounded-md p-0">{id}</Badge>
          {name}
        </CardTitle>
      </CardHeader>
      {filled ? <CardContent className="px-5">{body}</CardContent> : null}
    </Card>
  );
  const sections = [...WAYS, LEISURE].map((w) => {
    const books = w.id === "leisure" ? [] : view.books.filter((b) => wayOf(b) === w.id);
    const teaching = w.id === "teaching";
    const sh = shelvesOf({ ...view, books }, teaching && all, teaching);
    const here = (items ?? []).filter((i) => placeOf(i) === w.id);
    return { w, books, sh, here, count: books.length + here.length };
  });
  return (
    <section className="space-y-10">
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Note>Arranged by the ways of learning Aristotle names: teaching, dialectic, imitation and practice. Each item shows its medium.</Note>
          <Button asChild variant="ghost" size="sm"><Link href={all ? "/learn?filled=1" : "/learn"}>{all ? "Only filled shelves" : "All 20 shelves"}</Link></Button>
        </div>
        <nav className="flex flex-wrap gap-2" aria-label="Ways of learning">
          {sections.filter((s) => s.w.id !== "leisure" || s.count).map((s) => (
            <a key={s.w.id} href={`#${s.w.id}`} className="hover:bg-accent rounded-full border px-3 py-1 text-xs transition">{s.w.label} <span className="text-muted-foreground">{s.count}</span></a>
          ))}
        </nav>
      </div>
      {sections.filter((s) => s.w.id !== "leisure" || s.count).map(({ w, books, sh, here }) => (
        <section key={w.id} id={w.id} className="scroll-mt-20 space-y-4">
          <div>
            <h2 className="text-base font-semibold">{w.label}</h2>
            <Note>{w.about}</Note>
          </div>
          {here.length ? (
            <div className="grid gap-3 sm:grid-cols-2">
              {here.map((i) => (
                <Link key={i.id} href={`/learn?item=${i.id}${keep}`} className={cn("hover:bg-accent block space-y-1 rounded-lg border p-3 transition", openItem === i.id && "bg-accent")}>
                  <span className="flex items-center gap-2">
                    <Badge variant="secondary">{mediumOf(i.medium).label}</Badge>
                    <span className="text-muted-foreground text-xs">{statusLabel(i)}</span>
                  </span>
                  <span className="line-clamp-2 block text-sm font-medium">{i.title}</span>
                  {i.by ? <span className="text-muted-foreground block text-xs">{i.by}</span> : null}
                  {i.lesson ? <span className="line-clamp-2 block text-xs">{i.lesson}</span> : null}
                </Link>
              ))}
            </div>
          ) : null}
          {sh.shelves.map((s) => shelf(`${w.id}-${s.id}`, s.id, s.name, Boolean(s.books.length || s.subjects.length), row(s.books, s.subjects)))}
          {sh.unshelved.books.length || sh.unshelved.subjects.length ? shelf(`${w.id}-?`, "?", "Not yet placed (the catalog gives no category)", true, row(sh.unshelved.books, sh.unshelved.subjects)) : null}
          {w.id === "teaching" && !view.books.length ? <Note>No books yet. <Link className="underline underline-offset-4" href="/learn/browse">Browse books</Link> and each one you add goes on its shelf.</Note> : null}
          {w.id !== "teaching" && !books.length && !here.length ? <Note>Nothing here yet.</Note> : null}
        </section>
      ))}
      {items ? <AddItem /> : <Note>Lectures, debates, films, music and games are not set up on this database yet.</Note>}
      {book ? (
        <>
        <Link href={close} aria-label="Close" tabIndex={-1} className="fixed inset-0 z-10 bg-black/40" />
        <aside className="bg-background fixed inset-y-0 right-0 z-20 w-full max-w-80 space-y-4 overflow-y-auto border-l p-6 shadow-lg" aria-label={book.title}>
          <div className="flex items-start justify-between gap-2">
            <h2 className="text-lg font-semibold leading-tight">{book.title}</h2>
            <Button asChild variant="ghost" size="icon" aria-label="Close"><Link href={close}><XIcon /></Link></Button>
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
          <form action={setBookWayAction} className="space-y-2">
            <input type="hidden" name="book" value={book.id} />
            <p className="text-xs font-medium">Way of learning</p>
            <div className="flex flex-wrap gap-2">
              {WAYS.map((w) => <Button key={w.id} size="sm" name="way" value={w.id} variant={wayOf(book) === w.id ? "default" : "outline"} aria-pressed={wayOf(book) === w.id}>{w.label}</Button>)}
            </div>
            {book.way ? <Button size="sm" variant="ghost" name="way" value="">Back to {WAYS.find((w) => w.id === bookWay(book.catalog ? categoryOf(book.catalog) : null))!.label.toLowerCase()}, from its category</Button> : null}
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
      {item ? (
        <>
        <Link href={close} aria-label="Close" tabIndex={-1} className="fixed inset-0 z-10 bg-black/40" />
        <aside className="bg-background fixed inset-y-0 right-0 z-20 w-full max-w-80 space-y-4 overflow-y-auto border-l p-6 shadow-lg" aria-label={item.title}>
          <div className="flex items-start justify-between gap-2">
            <h2 className="text-lg font-semibold leading-tight">{item.title}</h2>
            <Button asChild variant="ghost" size="icon" aria-label="Close"><Link href={close}><XIcon /></Link></Button>
          </div>
          <Note>{[mediumOf(item.medium).label, item.by].filter(Boolean).join(" · ")}</Note>
          {item.url ? <a href={item.url} target="_blank" rel="noreferrer" className="block truncate text-sm underline underline-offset-4">{item.url}</a> : null}
          <form action={changeItemAction} className="flex flex-wrap gap-2">
            <input type="hidden" name="item" value={item.id} />
            {(["want", "doing", "done"] as const).map((s) => (
              <Button key={s} size="sm" name="status" value={s} variant={item.status === s ? "default" : "outline"} aria-pressed={item.status === s}>{statusLabel({ ...item, status: s })}</Button>
            ))}
          </form>
          <form action={changeItemAction} className="space-y-2">
            <input type="hidden" name="item" value={item.id} />
            <p className="text-xs font-medium">Way of learning</p>
            <div className="flex flex-wrap gap-2">
              {WAYS.map((w) => {
                const on = (item.way ?? mediumOf(item.medium).way) === w.id;
                return <Button key={w.id} size="sm" name="way" value={w.id} variant={on ? "default" : "outline"} aria-pressed={on}>{w.label}</Button>;
              })}
            </div>
            {item.way ? <Button size="sm" variant="ghost" name="way" value="">Back to {WAYS.find((w) => w.id === mediumOf(item.medium).way)!.label.toLowerCase()}, from its medium</Button> : null}
          </form>
          <form action={changeItemAction} className="space-y-2">
            <input type="hidden" name="item" value={item.id} />
            <label className="block space-y-1.5 text-xs font-medium">What it taught you<Textarea name="lesson" rows={4} maxLength={4000} defaultValue={item.lesson ?? ""} /></label>
            {mediumOf(item.medium).lesson && !item.lesson ? <Note>Until you say what it taught you, it is kept under Rest and leisure.</Note> : null}
            <Button size="sm">Save</Button>
          </form>
          <form action={removeItemAction}>
            <input type="hidden" name="item" value={item.id} />
            <Button size="sm" variant="ghost">Remove</Button>
          </form>
        </aside>
        </>
      ) : null}
    </section>
  );
}

/** Keep a lecture, debate, discussion, podcast, film, series, theatre, music, art or game. */
function AddItem() {
  return (
    <Card className="py-5">
      <CardHeader className="px-5"><CardTitle className="text-sm">Add a lecture, debate, film, music or game</CardTitle></CardHeader>
      <CardContent className="px-5">
        <form action={addItemAction} className="grid gap-4 sm:grid-cols-2">
          <label className="block space-y-1.5 text-sm font-medium">Medium
            <select name="medium" className={SELECT} defaultValue="lecture">
              {WAYS.map((w) => <optgroup key={w.id} label={w.label}>{MEDIA.filter((m) => m.way === w.id).map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}</optgroup>)}
            </select>
          </label>
          <label className="block space-y-1.5 text-sm font-medium">Status
            <select name="status" className={SELECT} defaultValue="done">
              <option value="done">Done</option>
              <option value="doing">Under way</option>
              <option value="want">Want to</option>
            </select>
          </label>
          <label className="block space-y-1.5 text-sm font-medium sm:col-span-2">Title<Input name="title" required maxLength={300} /></label>
          <label className="block space-y-1.5 text-sm font-medium">By (optional)<Input name="by" maxLength={200} /></label>
          <label className="block space-y-1.5 text-sm font-medium">Link (optional)<Input name="url" type="url" maxLength={1000} /></label>
          <label className="block space-y-1.5 text-sm font-medium sm:col-span-2">Way of learning
            <select name="way" className={SELECT} defaultValue="">
              <option value="">The medium&apos;s way</option>
              {WAYS.map((w) => <option key={w.id} value={w.id}>{w.label}</option>)}
            </select>
          </label>
          <label className="block space-y-1.5 text-sm font-medium sm:col-span-2">What it taught you (optional)<Textarea name="lesson" rows={3} maxLength={4000} /></label>
          <p className="text-muted-foreground text-xs sm:col-span-2">A game, music or art is kept under Rest and leisure until you say what it taught you.</p>
          <div className="sm:col-span-2"><Button size="sm">Add</Button></div>
        </form>
      </CardContent>
    </Card>
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
