import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { BOOK_STATUS, GRASP, namedSubjects } from "@reflect/core/learning";
import { requireUser } from "@/server/session";
import { currentName } from "@/server/person-store";
import { getBook } from "@/server/learn-store";
import { initialOf, Nav } from "../../../nav";
import { addSubjectsAction, rateAction, recallAction, setStatusAction } from "../../actions";

export const metadata: Metadata = { title: "Book" };
export const dynamic = "force-dynamic";

export default async function BookPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const { id } = await params;
  const b = await getBook(user.id, id);
  if (!b) notFound();
  const last = b.recalls.at(-1);
  const named = last ? namedSubjects(last.text, b.subjects.map((s) => ({ id: s.id, title: s.name }))) : new Set<string>();
  const have = new Set(b.subjects.map((s) => s.name.toLowerCase()));
  const suggested = (b.catalog?.subjects ?? []).filter((s) => !have.has(s.toLowerCase()));
  return (
    <main className="page">
      <Nav current="learn" eyebrow="Learn" initial={initialOf(await currentName(user.id))} />
      <p className="note"><Link href="/learn">What you know</Link></p>
      <h1 className="title">{b.title}</h1>
      {b.authors.length ? <p className="sec-note">{b.authors.join(", ")}{b.isbn ? `, ISBN ${b.isbn}` : ""}</p> : null}

      <form action={setStatusAction} className="add">
        <input type="hidden" name="book" value={b.id} />
        <label className="lbl" htmlFor="status">Where you are with it{b.finishedOn ? ` (finished ${b.finishedOn})` : b.startedOn ? ` (started ${b.startedOn})` : ""}</label>
        <select id="status" name="status" defaultValue={b.status} className="learn-select">
          {BOOK_STATUS.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <div className="actions"><button type="submit" className="primary">Save</button></div>
      </form>

      {b.status === "read" || b.status === "abandoned" ? (
        <section className="sec" aria-labelledby="h-recall">
          <div className="sec-head"><h2 id="h-recall" className="lbl">What you remember</h2></div>
          <form action={recallAction} className="add">
            <input type="hidden" name="book" value={b.id} />
            <label className="sec-note" htmlFor="recall">Without looking, name its key subjects: what it is about, its main ideas, what you would tell someone.</label>
            <textarea id="recall" name="recall" rows={4} required maxLength={20000} className="learn-text" />
            <div className="actions"><button type="submit" className="primary">Keep what I named</button></div>
          </form>
          {b.recalls.map((r) => (
            <blockquote key={r.at} className="note learn-recall"><span className="muted">{r.at.slice(0, 10)}</span> {r.text}</blockquote>
          ))}
        </section>
      ) : null}

      <section className="sec" aria-labelledby="h-subjects">
        <div className="sec-head"><h2 id="h-subjects" className="lbl">Its subjects, and your grasp</h2></div>
        {b.subjects.length ? (
          <form action={rateAction}>
            <input type="hidden" name="book" value={b.id} />
            {b.subjects.map((s) => (
              <fieldset key={s.id} className="learn-grasp">
                <legend>
                  {s.name}
                  {last ? <span className="muted">{named.has(s.id) ? " (you named it)" : " (you did not name it)"}</span> : null}
                  {s.grasp ? <span className="muted">; last: {GRASP.find((g) => g.id === s.grasp!.grasp)?.text}</span> : null}
                </legend>
                <div className="choices">
                  {GRASP.map((g) => (
                    <label key={g.id} className="chip">
                      <input type="radio" name={`grasp-${s.id}`} value={g.id} /> {g.text}
                    </label>
                  ))}
                </div>
              </fieldset>
            ))}
            <div className="actions"><button type="submit" className="primary">Keep my ratings</button></div>
          </form>
        ) : (
          <p className="sec-note">No subjects yet. Add the ones you would name.</p>
        )}
        <form action={addSubjectsAction} className="add">
          <input type="hidden" name="book" value={b.id} />
          <label className="lbl" htmlFor="subjects">Add subjects (comma separated)</label>
          <input id="subjects" name="subjects" maxLength={2000} />
          {suggested.length && (last || b.status !== "read") ? (
            <>
              <p className="sec-note">Open Library lists these. Keep only the ones that fit.</p>
              <div className="choices">
                {suggested.map((s) => (
                  <label key={s} className="chip"><input type="checkbox" name="suggested" value={s} /> {s}</label>
                ))}
              </div>
            </>
          ) : null}
          <div className="actions"><button type="submit" className="primary">Add</button></div>
        </form>
      </section>

      {b.catalog?.contents.length && (last || b.status !== "read") ? (
        <section className="sec" aria-labelledby="h-contents">
          <div className="sec-head"><h2 id="h-contents" className="lbl">Contents, from Open Library</h2></div>
          <ul className="note">{b.catalog.contents.map((c, i) => <li key={i}>{c.trim()}</li>)}</ul>
        </section>
      ) : null}
    </main>
  );
}
