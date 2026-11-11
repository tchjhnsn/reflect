import type { Metadata } from "next";
import Link from "next/link";
import { BOOK_STATUS, GRASP } from "@reflect/core/learning";
import { requireUser } from "@/server/session";
import { currentName } from "@/server/person-store";
import { learnView } from "@/server/learn-store";
import { initialOf, Nav } from "../nav";
import { addBookAction } from "./actions";

export const metadata: Metadata = { title: "Learn" };
export const dynamic = "force-dynamic";

const graspText = (g: string) => GRASP.find((x) => x.id === g)?.text ?? g;

export default async function LearnPage() {
  const user = await requireUser();
  const v = await learnView(user.id);
  return (
    <main className="page">
      <Nav current="learn" eyebrow="Learn" initial={initialOf(await currentName(user.id))} />
      <h1 className="title">What you know</h1>
      <p className="sec-note">Nothing here is known until you say how well you grasp it. A book you read but have not rated counts as read.</p>

      <section className="sec" aria-labelledby="h-known">
        <div className="sec-head"><h2 id="h-known" className="lbl">Known</h2></div>
        {v.known.length ? (
          <dl className="facts" style={{ borderTop: 0 }}>
            {v.known.map((s) => (
              <div className="fact" key={s.id}><dt>{s.name}</dt><dd>{graspText(s.grasp!.grasp)}</dd></div>
            ))}
          </dl>
        ) : (
          <p className="sec-note">Nothing rated yet. Open a book you have read and name what you remember of it.</p>
        )}
      </section>

      <section className="sec" aria-labelledby="h-books">
        <div className="sec-head"><h2 id="h-books" className="lbl">Books</h2></div>
        {v.books.length ? (
          <ul className="note learn-list">
            {v.books.map((b) => (
              <li key={b.id}>
                <Link href={`/learn/book/${b.id}`}>{b.title}</Link>
                {b.authors.length ? `, ${b.authors.join(", ")}` : ""} <span className="muted">({b.status}{b.finishedOn ? ` ${b.finishedOn}` : ""})</span>
                {b.unrated.length ? <span className="muted">. Not yet rated: {b.unrated.join(", ")}</span> : null}
              </li>
            ))}
          </ul>
        ) : (
          <p className="sec-note">No books yet.</p>
        )}
      </section>

      <section className="sec" aria-labelledby="h-add">
        <div className="sec-head"><h2 id="h-add" className="lbl">Add a book</h2></div>
        <form action={addBookAction} className="add">
          <label className="lbl" htmlFor="title">Title</label>
          <input id="title" name="title" required maxLength={300} />
          <label className="lbl" htmlFor="isbn">ISBN, to look it up in Open Library (optional)</label>
          <input id="isbn" name="isbn" inputMode="numeric" maxLength={20} />
          <label className="lbl" htmlFor="status">Where you are with it</label>
          <select id="status" name="status" defaultValue="read" className="learn-select">
            {BOOK_STATUS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <label className="lbl" htmlFor="subjects">Its key subjects, as you would name them (comma separated, optional)</label>
          <input id="subjects" name="subjects" maxLength={2000} />
          <div className="actions"><button type="submit" className="primary">Keep the book</button></div>
        </form>
      </section>
    </main>
  );
}
