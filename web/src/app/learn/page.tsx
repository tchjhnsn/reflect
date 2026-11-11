import type { Metadata } from "next";
import Link from "next/link";
import { BOOK_STATUS, FILM_KINDS, FILM_STATUS, GRASP, RESOURCE_KINDS } from "@reflect/core/learning";
import { SWEEP_PROMPT } from "@reflect/core/sweep";
import { inputTitle, listInputs } from "@/server/ideas-store";
import { requireUser } from "@/server/session";
import { currentName } from "@/server/person-store";
import { learnView } from "@/server/learn-store";
import { initialOf, Nav } from "../nav";
import { addBookAction, addCredentialAction, addInputAction, rateCoursesAction, sweepAction, uploadLinkedInAction, uploadTranscriptAction } from "./actions";
import { listCourses, listCredentials, recommendations } from "@/server/learn-sources-store";

export const metadata: Metadata = { title: "Learn" };
export const dynamic = "force-dynamic";

const graspText = (g: string) => GRASP.find((x) => x.id === g)?.text ?? g;

export default async function LearnPage() {
  const user = await requireUser();
  const [v, courses, credentials, recs, inputs] = await Promise.all([learnView(user.id), listCourses(user.id), listCredentials(user.id), recommendations(user.id), listInputs(user.id)]);
  return (
    <main className="page">
      <Nav current="learn" eyebrow="Learn" initial={initialOf(await currentName(user.id))} />
      <h1 className="title">What you know</h1>
      <p className="sec-note">What you take in: books, films, talks, conversations, links and the subjects you study. Nothing here is known until you say how well you grasp it. Your own ideas are outputs and live in <Link href="/ideas">Ideas</Link>.</p>

      <section className="sec" aria-labelledby="h-known">
        <div className="sec-head"><h2 id="h-known" className="lbl">Known</h2></div>
        {v.known.length ? (
          <dl className="facts" style={{ borderTop: 0 }}>
            {v.known.map((s) => (
              <div className="fact" key={s.id}><dt>{s.name}</dt><dd>{graspText(s.grasp!.grasp)}{s.vouches ? `; vouched for by ${s.vouches}` : ""}</dd></div>
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

      {v.shown.length ? (
        <section className="sec" aria-labelledby="h-shown">
          <div className="sec-head"><h2 id="h-shown" className="lbl">Shown, not yet rated by you</h2></div>
          <p className="sec-note">Someone saw you show knowledge of these. Rate them yourself to count them as known.</p>
          <ul className="note learn-list">{v.shown.map((s) => <li key={s.id}>{s.name}</li>)}</ul>
        </section>
      ) : null}

      {v.vouched.length ? (
        <section className="sec" aria-labelledby="h-vouched">
          <div className="sec-head"><h2 id="h-vouched" className="lbl">Vouched for by others</h2></div>
          <p className="sec-note">Others say you know these. They stay apart from what you know until you rate them yourself.</p>
          <ul className="note learn-list">{v.vouched.map((s) => <li key={s.id}>{s.name}, by {s.vouches}</li>)}</ul>
        </section>
      ) : null}

      {v.suggested.length ? (
        <section className="sec" aria-labelledby="h-suggested">
          <div className="sec-head"><h2 id="h-suggested" className="lbl">Suggested, not known</h2></div>
          <p className="sec-note">Someone thinks these relate to you. They are places to look, not what you know.</p>
          <ul className="note learn-list">{v.suggested.map((s) => <li key={s.id}>{s.name}</li>)}</ul>
        </section>
      ) : null}

      <section className="sec" aria-labelledby="h-inputs">
        <div className="sec-head"><h2 id="h-inputs" className="lbl">Films, talks, conversations and links</h2></div>
        {inputs.length ? (
          <ul className="note learn-list">
            {inputs.map((i) => (
              <li key={i.id}>
                {i.body.kind === "link" && i.body.url ? <a href={i.body.url} rel="noreferrer">{inputTitle(i.body)}</a> : inputTitle(i.body)}{" "}
                <span className="muted">
                  ({i.body.kind === "film" ? `${i.body.film}${i.body.year ? ` ${i.body.year}` : ""}, ${i.body.status}` : i.body.kind === "talk" ? `talk by ${i.body.speaker}` : i.body.kind === "conversation" ? i.body.on : `${i.body.link}${i.body.source ? ` on ${i.body.source}` : ""}`})
                  {i.subjects.length ? `. ${i.subjects.join(", ")}` : ""}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="sec-note">Nothing yet beside your books.</p>
        )}
        <form action={addInputAction} className="add">
          <label className="lbl" htmlFor="ikind">What it is</label>
          <select id="ikind" name="kind" defaultValue="film" className="learn-select">
            <option value="film">a film, documentary or series</option>
            <option value="talk">a talk: someone you saw speak</option>
            <option value="conversation">a conversation</option>
            <option value="link">a link: a post, thread, video or article</option>
          </select>
          <label className="lbl" htmlFor="ititle">Title, or for a conversation what it was about</label>
          <input id="ititle" name="title" maxLength={300} />
          <label className="lbl" htmlFor="iwho">Who: the speaker, the people you talked with (separated by ;), or the author</label>
          <input id="iwho" name="who" maxLength={300} />
          <label className="lbl" htmlFor="iurl">Link (optional)</label>
          <input id="iurl" name="url" type="url" maxLength={2000} />
          <label className="lbl" htmlFor="ion">When (YYYY-MM-DD, optional)</label>
          <input id="ion" name="on" maxLength={10} />
          <label className="lbl" htmlFor="ifilm">For a film</label>
          <div className="choices">
            <select id="ifilm" name="film" defaultValue="film" className="learn-select" aria-label="Kind of film">{FILM_KINDS.map((k) => <option key={k} value={k}>{k}</option>)}</select>
            <select name="status" defaultValue="watched" className="learn-select" aria-label="Whether you watched it">{FILM_STATUS.map((k) => <option key={k} value={k}>{k}</option>)}</select>
            <input name="year" inputMode="numeric" maxLength={4} placeholder="year" aria-label="Year" />
          </div>
          <label className="lbl" htmlFor="ilink">For a link</label>
          <select id="ilink" name="link" defaultValue="post" className="learn-select">{RESOURCE_KINDS.map((k) => <option key={k} value={k}>{k}</option>)}</select>
          <label className="lbl" htmlFor="isubjects">What it touches on, as subjects (comma separated, optional)</label>
          <input id="isubjects" name="subjects" maxLength={2000} />
          <div className="actions"><button type="submit" className="primary">Keep it</button></div>
        </form>
      </section>

      <section className="sec" aria-labelledby="h-sweep">
        <div className="sec-head"><h2 id="h-sweep" className="lbl">From your AI&apos;s memory</h2></div>
        <p className="sec-note">Paste this into the AI you talk to most. It lists from what it already knows of you what you read, watched and heard, subjects you showed or it suggests, and your own ideas with what fed them. You keep items one by one.</p>
        <details className="note">
          <summary>The prompt</summary>
          <textarea readOnly className="learn-prompt" rows={12} defaultValue={SWEEP_PROMPT} aria-label="The prompt" />
        </details>
        <form action={sweepAction} className="add">
          <label className="lbl" htmlFor="reply">Its reply</label>
          <textarea id="reply" name="reply" rows={5} maxLength={2_000_000} />
          <label className="lbl" htmlFor="replyfile">Or the file it is in</label>
          <input id="replyfile" name="file" type="file" accept=".json,.md,.txt,application/json,text/plain,text/markdown" />
          <div className="actions"><button type="submit" className="primary">Read the reply</button></div>
        </form>
      </section>

      <section className="sec" aria-labelledby="h-courses">
        <div className="sec-head"><h2 id="h-courses" className="lbl">Courses</h2></div>
        {courses.length ? (
          <form action={rateCoursesAction}>
            {courses.map((c) => (
              <fieldset key={c.id} className="learn-grasp">
                <legend>
                  {c.row.term} {c.row.code ? `${c.row.code} ` : ""}{c.row.title}
                  <span className="muted">{c.row.grade ? `; grade ${c.row.grade}` : ""}{c.subject.grasp ? `; you say: ${graspText(c.subject.grasp.grasp)}` : "; not yet rated by you"}</span>
                </legend>
                <div className="choices">
                  {GRASP.map((g) => (
                    <label key={g.id} className="chip"><input type="radio" name={`grasp-${c.id}`} value={g.id} /> {g.text}</label>
                  ))}
                </div>
              </fieldset>
            ))}
            <div className="actions"><button type="submit" className="primary">Keep my ratings</button></div>
          </form>
        ) : (
          <p className="sec-note">Upload a transcript to look back over what you studied and say what you kept of it, whatever the grade.</p>
        )}
        <form action={uploadTranscriptAction} className="add">
          <label className="lbl" htmlFor="transcript">Transcript (PDF or text)</label>
          <input id="transcript" name="file" type="file" accept=".pdf,.txt,application/pdf,text/plain" required />
          <p className="sec-note">A model drafts its courses; only rows really in your transcript are offered, and you choose which to keep.</p>
          <div className="actions"><button type="submit" className="primary">Read my transcript</button></div>
        </form>
      </section>

      <section className="sec" aria-labelledby="h-creds">
        <div className="sec-head"><h2 id="h-creds" className="lbl">Credentials</h2></div>
        {credentials.length ? (
          <ul className="note learn-list">
            {credentials.map((c) => (
              <li key={c.id}>
                {c.name}, {c.issuer}{c.issued ? `, ${c.issued}` : ""}{c.subjects.length ? `. Covers ${c.subjects.join(", ")}` : ""}
                {c.checked ? <span className="muted">. The issuer&apos;s badge says {c.checked.name}</span> : null}
              </li>
            ))}
          </ul>
        ) : null}
        <form action={addCredentialAction} className="add">
          <label className="lbl" htmlFor="cname">Name</label>
          <input id="cname" name="name" maxLength={300} />
          <label className="lbl" htmlFor="issuer">Issuer</label>
          <input id="issuer" name="issuer" maxLength={300} />
          <label className="lbl" htmlFor="issued">Issued (YYYY-MM, optional)</label>
          <input id="issued" name="issued" maxLength={10} />
          <label className="lbl" htmlFor="curl">Link to the badge or certificate (optional)</label>
          <input id="curl" name="url" type="url" maxLength={2000} />
          <label className="chip"><input type="checkbox" name="check" /> Check the badge with its issuer</label>
          <label className="lbl" htmlFor="csubjects">Subjects it covers (comma separated, optional)</label>
          <input id="csubjects" name="subjects" maxLength={2000} />
          <div className="actions"><button type="submit" className="primary">Keep the credential</button></div>
        </form>
      </section>

      <section className="sec" aria-labelledby="h-li">
        <div className="sec-head"><h2 id="h-li" className="lbl">From LinkedIn</h2></div>
        {recs.length ? (
          recs.map((r, i) => (
            <blockquote key={i} className="note learn-recall">{r.text} <span className="muted">({r.by}{r.role ? `, ${r.role}` : ""})</span></blockquote>
          ))
        ) : null}
        <form action={uploadLinkedInAction} className="add">
          <label className="lbl" htmlFor="linkedin">Your LinkedIn data export (the zip)</label>
          <input id="linkedin" name="file" type="file" accept=".zip,application/zip" required />
          <p className="sec-note">Ask LinkedIn for a copy of your data under Settings, Data privacy. You see what it holds before anything is kept.</p>
          <div className="actions"><button type="submit" className="primary">Read the export</button></div>
        </form>
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
