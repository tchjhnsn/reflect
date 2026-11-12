import type { Metadata } from "next";
import Link from "next/link";
import { browseClass, searchBooks, type CatalogEntry } from "@reflect/core/openlibrary";
import { LCC_CLASSES, mainClass } from "@reflect/core/shelves";
import { requireUser } from "@/server/session";
import { currentName } from "@/server/person-store";
import { learnView } from "@/server/learn-store";
import { initialOf, Nav } from "../../nav";
import { addFromLibraryAction } from "../actions";

export const metadata: Metadata = { title: "Browse the library" };
export const dynamic = "force-dynamic";

const openLibrary = (url: string) => fetch(url, { headers: { "user-agent": "reflect-app" }, cache: "no-store" });
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

export default async function BrowsePage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const q = one(sp.q).trim().slice(0, 200);
  const cls = LCC_CLASSES.find((c) => c.id === one(sp.class))?.id ?? null;
  const page = Math.max(1, Number(one(sp.page)) || 1);
  const v = await learnView(user.id);
  const kept = new Set(v.books.map((b) => b.catalog?.work).filter(Boolean));

  let results: CatalogEntry[] = [];
  let failed = false;
  try {
    if (q) results = await searchBooks(q, openLibrary, 20);
    else if (cls) results = await browseClass(cls, openLibrary, { page, limit: 24 });
  } catch {
    failed = true;
  }
  const className = (id: string | null) => LCC_CLASSES.find((c) => c.id === id)?.name ?? "";

  return (
    <main className="page">
      <Nav current="learn" eyebrow="Learn" initial={initialOf(await currentName(user.id))} />
      <h1 className="title">Browse the library</h1>
      <p className="sec-note">The whole library, from Open Library's open catalog, arranged by the Library of Congress Classification. Put a book on your shelves and say where you are with it. <Link href="/learn">Back to your shelves</Link></p>

      <section className="sec" aria-labelledby="h-search">
        <div className="sec-head"><h2 id="h-search" className="lbl">Search</h2></div>
        <form method="get" className="add browse-search">
          <label className="sr-only" htmlFor="q">Title or author</label>
          <input id="q" name="q" defaultValue={q} placeholder="Title or author" />
          <div className="actions"><button type="submit" className="primary">Search</button></div>
        </form>
      </section>

      <section className="sec" aria-labelledby="h-classes">
        <div className="sec-head"><h2 id="h-classes" className="lbl">Shelves</h2></div>
        <ul className="note learn-list browse-classes">
          {LCC_CLASSES.map((c) => (
            <li key={c.id}>
              {cls === c.id && !q ? <strong>{c.id} {c.name}</strong> : <Link href={`/learn/browse?class=${c.id}`}><span className="muted">{c.id}</span> {c.name}</Link>}
            </li>
          ))}
        </ul>
      </section>

      {q || cls ? (
        <section className="sec" aria-labelledby="h-results">
          <div className="sec-head"><h2 id="h-results" className="lbl">{q ? `Results for "${q}"` : `${cls} ${className(cls)}`}</h2></div>
          {failed ? <p className="sec-note">Open Library did not answer. Try again in a moment.</p> : null}
          {!failed && !results.length ? <p className="sec-note">Nothing found.</p> : null}
          <ul className="note learn-list">
            {results.map((b) => {
              const shelf = mainClass(b.lcc);
              return (
                <li key={b.work}>
                  <strong>{b.title}</strong>
                  {b.authors.length ? `, ${b.authors.slice(0, 3).join(", ")}` : ""}
                  {b.year ? <span className="muted"> ({b.year})</span> : null}
                  <span className="muted"> · {shelf ? `${shelf} ${className(shelf)}` : "no call number"}</span>
                  {kept.has(b.work) ? (
                    <span className="muted"> · on your shelves</span>
                  ) : (
                    <form action={addFromLibraryAction} className="browse-add">
                      <input type="hidden" name="entry" value={JSON.stringify(b)} />
                      <label className="sr-only" htmlFor={`s-${b.work}`}>Where you are with it</label>
                      <select id={`s-${b.work}`} name="status" defaultValue="want" className="learn-select">
                        <option value="want">want to read</option>
                        <option value="reading">reading</option>
                        <option value="read">read</option>
                      </select>
                      <button type="submit" className="primary">Add to my shelves</button>
                    </form>
                  )}
                </li>
              );
            })}
          </ul>
          {cls && !q && results.length ? (
            <p className="sec-note">
              {page > 1 ? <Link href={`/learn/browse?class=${cls}&page=${page - 1}`}>Previous</Link> : null}
              {page > 1 ? " · " : ""}
              <Link href={`/learn/browse?class=${cls}&page=${page + 1}`}>More</Link>
            </p>
          ) : null}
        </section>
      ) : null}
    </main>
  );
}
