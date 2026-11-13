import type { Metadata } from "next";
import Link from "next/link";
import { STANCES } from "@reflect/core/claims";
import { browseClass, searchBooks, type CatalogEntry } from "@reflect/core/openlibrary";
import { LCC_CLASSES } from "@reflect/core/shelves";
import { requireUser } from "@/server/session";
import { learnView, type LearnView } from "@/server/learn-store";
import { listClaims, type ClaimOut } from "@/server/claims-store";
import { shelvesOf } from "@/server/learn-shelves";
import { mono } from "../fonts";
import { addFromLibraryAction, keepClaimAction, setBookStatusAction, setStanceAction } from "./actions";
import { bookTone, subjectTone, type ShelfTone } from "./tones";

export const metadata: Metadata = { title: "Learn" };
export const dynamic = "force-dynamic";

type Tab = "shelves" | "library" | "claims";
const BOOK = "M8 4 h84 v92 h-84 z";
const SUBJ = "M50 4 A46 46 0 1 1 49.99 4 Z";
const STATUS = [["want", "WANT TO READ"], ["reading", "READING"], ["read", "READ"]] as const;
const openLibrary = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, cache: "no-store" });
const STANCE_LABEL: Record<string, string> = { belief: "BELIEF", disbelief: "DISBELIEF", suspension: "SUSPENSION OF JUDGMENT" };
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/** Learn in the new design: your shelves by Library of Congress class, the whole library, and your claims. */
export default async function LearnPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const tab: Tab = one(sp.tab) === "library" ? "library" : one(sp.tab) === "claims" ? "claims" : "shelves";
  const view = await learnView(user.id);
  const href = (t: Tab, extra = "") => `/learn?tab=${t}${extra}`;
  return (
    <main className={`nd ${mono.className}`}>
      <div className="nd-bar" aria-hidden="true" />
      <header className="nd-head">
        <div className="nd-head-l">
          <Link href="/" className="nd-quiet">&#8592; HOME</Link>
          <h1 className="nd-title">LEARN</h1>
        </div>
        <nav className="nd-tabs" aria-label="Learn">
          <Link href={href("shelves")} aria-current={tab === "shelves" ? "page" : undefined}>MY SHELVES</Link>
          <Link href={href("library")} aria-current={tab === "library" ? "page" : undefined}>LIBRARY</Link>
          <Link href={href("claims")} aria-current={tab === "claims" ? "page" : undefined}>CLAIMS</Link>
        </nav>
      </header>
      {tab === "shelves" ? <Shelves view={view} all={one(sp.all) === "1"} open={one(sp.book)} /> : null}
      {tab === "library" ? <Library view={view} cls={one(sp.class).toUpperCase()} q={one(sp.q).trim()} /> : null}
      {tab === "claims" ? <Claims claims={await listClaims(user.id).catch((): ClaimOut[] | null => null)} /> : null}
      <footer className="nd-foot">
        <span>&#9632; BOOK</span>
        <span>&#9679; SUBJECT</span>
        <span>DARK = KNOWN · GREY = LEARNING · LIGHT = REGISTERED</span>
        <span>LIBRARY OF CONGRESS CLASSIFICATION</span>
      </footer>
    </main>
  );
}

function Glyph({ d, tone, w }: { d: string; tone: ShelfTone; w: number }) {
  return <svg width={w} height={84} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d={d} className={`nd-tone-${tone}`} /></svg>;
}

function Shelves({ view, all, open }: { view: LearnView; all: boolean; open: string }) {
  const sh = shelvesOf(view, all);
  const byId = new Map(view.books.map((b) => [b.id, b]));
  const subjectBooks = new Map<string, LearnView["books"]>();
  for (const b of view.books) for (const s of b.subjects) subjectBooks.set(s.id, [...(subjectBooks.get(s.id) ?? []), b]);
  const grasp = new Map(view.books.flatMap((b) => b.subjects).concat(view.known).map((s) => [s.id, s.grasp?.grasp ?? null]));
  const book = open ? byId.get(open) : undefined;
  const items = (books: { id: string; title: string }[], subjects: { id: string; name: string }[]) => (
    <div className="nd-items">
      {books.map((b) => {
        const full = byId.get(b.id)!;
        return (
          <Link key={b.id} href={`/learn?tab=shelves&book=${b.id}${all ? "&all=1" : ""}`} className="nd-item" aria-label={`${b.title}, book, ${full.status}`}>
            <Glyph d={BOOK} tone={bookTone(full.status)} w={62} />
            <span>{b.title.toUpperCase()}</span>
          </Link>
        );
      })}
      {subjects.map((s) => (
        <span key={s.id} className="nd-item" aria-label={`${s.name}, subject`}>
          <Glyph d={SUBJ} tone={subjectTone(grasp.get(s.id), (subjectBooks.get(s.id) ?? []).map((b) => b.status))} w={84} />
          <span>{s.name.toUpperCase()}</span>
        </span>
      ))}
    </div>
  );
  return (
    <section className="nd-body">
      <div className="nd-row-end">
        <Link href={all ? "/learn?tab=shelves" : "/learn?tab=shelves&all=1"} className="nd-quiet nd-u">{all ? "ONLY FILLED SHELVES" : "ALL 21 SHELVES"}</Link>
      </div>
      {sh.shelves.length ? null : <p className="nd-note">No shelves yet. Find a book in the <Link href="/learn?tab=library">library</Link> and it goes on its shelf.</p>}
      {sh.shelves.map((s) => (
        <div key={s.id} className="nd-shelf">
          <div className="nd-shelf-name">
            <span className={`nd-letter${s.books.length || s.subjects.length ? "" : " is-empty"}`}>{s.id}</span>
            <span>{s.name.toUpperCase()}</span>
          </div>
          {items(s.books, s.subjects)}
        </div>
      ))}
      {sh.unshelved.books.length || sh.unshelved.subjects.length ? (
        <div className="nd-shelf">
          <div className="nd-shelf-name"><span className="nd-letter is-empty">?</span><span>NOT YET SHELVED (NO CALL NUMBER)</span></div>
          {items(sh.unshelved.books, sh.unshelved.subjects)}
        </div>
      ) : null}
      {book ? (
        <aside className="nd-panel" aria-label={book.title}>
          <div className="nd-panel-head">
            <span className="nd-panel-title">{book.title.toUpperCase()}</span>
            <Link href={`/learn?tab=shelves${all ? "&all=1" : ""}`} aria-label="Close" className="nd-x">&#10005;</Link>
          </div>
          <p className="nd-small">{book.authors.join(", ") || "Author not given"}{book.catalog?.year ? ` · ${book.catalog.year}` : ""}</p>
          <form action={setBookStatusAction} className="nd-choices">
            <input type="hidden" name="book" value={book.id} />
            {STATUS.map(([id, label]) => (
              <button key={id} name="status" value={id} className={book.status === id ? "nd-btn" : "nd-btn-line"} aria-pressed={book.status === id}>{label}</button>
            ))}
          </form>
          {book.subjects.length ? <p className="nd-small">SUBJECTS · {book.subjects.map((s) => s.name).join(", ")}</p> : null}
          {book.catalog?.subjects.length ? <p className="nd-small nd-dim">THE CATALOG SUGGESTS · {book.catalog.subjects.slice(0, 8).join(", ")}</p> : null}
        </aside>
      ) : null}
    </section>
  );
}

async function Library({ view, cls, q }: { view: LearnView; cls: string; q: string }) {
  const pick = LCC_CLASSES.find((c) => c.id === cls);
  let books: CatalogEntry[] | null = [];
  let title = "SEARCH THE LIBRARY, OR PICK A SHELF";
  if (q) {
    books = await searchBooks(q, openLibrary, 12).catch(() => null);
    title = `RESULTS FOR "${q.toUpperCase()}"`;
  } else if (pick) {
    books = await browseClass(pick.id, openLibrary, { limit: 12 }).catch(() => null);
    title = `${pick.id} · ${pick.name.toUpperCase()} · BEST RATED`;
  }
  const kept = new Set(view.books.map((b) => b.catalog?.work).filter(Boolean));
  return (
    <section className="nd-body nd-library">
      <nav className="nd-classes" aria-label="Shelves">
        {LCC_CLASSES.map((c) => (
          <Link key={c.id} href={`/learn?tab=library&class=${c.id}`} aria-current={c.id === pick?.id ? "page" : undefined}>
            <span className="nd-dim">{c.id}</span><span>{c.name}</span>
          </Link>
        ))}
      </nav>
      <div className="nd-results">
        <form className="nd-search" action="/learn">
          <input type="hidden" name="tab" value="library" />
          <span className="nd-results-title">{title}</span>
          <label htmlFor="q" className="sr-only">Search titles or authors</label>
          <input id="q" name="q" defaultValue={q} placeholder="Search titles or authors..." />
        </form>
        {books === null ? <p className="nd-note">The open library did not answer. Try again in a moment.</p> : null}
        {books && (q || pick) && !books.length ? <p className="nd-note">Nothing found.</p> : null}
        {(books ?? []).map((b) => (
          <div key={b.work} className="nd-book">
            <svg width="34" height="46" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d={BOOK} className={kept.has(b.work) ? "nd-tone-known" : "nd-tone-registered"} /></svg>
            <div className="nd-book-text"><span>{b.title}</span><span className="nd-small">{b.authors.join(", ") || "Author not given"}{b.year ? ` · ${b.year}` : ""}</span></div>
            {kept.has(b.work) ? (
              <span className="nd-small">ON YOUR SHELVES</span>
            ) : (
              <form action={addFromLibraryAction}>
                <input type="hidden" name="entry" value={JSON.stringify(b)} />
                <input type="hidden" name="status" value="want" />
                <button className="nd-btn">ADD TO SHELVES</button>
              </form>
            )}
          </div>
        ))}
        <span className="nd-small">FROM OPEN LIBRARY</span>
      </div>
    </section>
  );
}

function Claims({ claims }: { claims: ClaimOut[] | null }) {
  if (!claims) return <section className="nd-body"><p className="nd-note">Claims are not set up on this database yet.</p></section>;
  return (
    <section className="nd-body nd-claims">
      <p className="nd-note">What you believe, disbelieve, or suspend judgment on, with who holds each view. Your latest stance counts; earlier ones are kept.</p>
      {STANCES.map((st) => {
        const here = claims.filter((c) => c.stance?.stance === st.id);
        return (
          <div key={st.id} className="nd-shelf">
            <div className="nd-shelf-name"><span className="nd-stance">{STANCE_LABEL[st.id]}</span><span>{here.length}</span></div>
            <div className="nd-claim-list">
              {here.length ? null : <span className="nd-small nd-dim">None</span>}
              {here.map((c) => (
                <div key={c.id} className="nd-claim">
                  <div className="nd-book-text">
                    <span>{c.text}</span>
                    <span className="nd-small">{[c.subject?.name, c.holders.map((h) => (h.citation ? `${h.name} (${h.citation})` : h.name)).join("; ")].filter(Boolean).join(" · ") || "No holder named"}</span>
                  </div>
                  <form action={setStanceAction} className="nd-choices">
                    <input type="hidden" name="claim" value={c.id} />
                    {STANCES.filter((x) => x.id !== st.id).map((x) => <button key={x.id} name="stance" value={x.id} className="nd-btn-line">{STANCE_LABEL[x.id]}</button>)}
                  </form>
                </div>
              ))}
            </div>
          </div>
        );
      })}
      <form action={keepClaimAction} className="nd-form">
        <span className="nd-results-title">KEEP A CLAIM</span>
        <label>CLAIM<input name="claim" required maxLength={500} /></label>
        <label>SUBJECT (OPTIONAL)<input name="subject" maxLength={120} /></label>
        <label>WHO HOLDS IT, ONE PER LINE: NAME | CITATION<textarea name="holders" rows={2} /></label>
        <div className="nd-choices">
          {STANCES.map((x) => <button key={x.id} name="stance" value={x.id} className="nd-btn-line">KEEP AS {STANCE_LABEL[x.id]}</button>)}
        </div>
      </form>
    </section>
  );
}
