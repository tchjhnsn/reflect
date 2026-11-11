import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { STANCES, type Target } from "@reflect/core/ideas";
import { requireUser } from "@/server/session";
import { currentName } from "@/server/person-store";
import { getIdea, labels } from "@/server/ideas-store";
import { initialOf, Nav } from "../../nav";
import { addInputAction, addRecommendationAction, dropAction, restateAction } from "../actions";

export const metadata: Metadata = { title: "Idea" };
export const dynamic = "force-dynamic";

const KIND: Record<string, string> = { book: "book", input: "kept", subject: "subject", idea: "idea" };

export default async function IdeaPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const { id } = await params;
  const idea = await getIdea(user.id, id);
  if (!idea) notFound();
  const choices = Object.entries(await labels(user.id)).filter(([k]) => k !== `idea:${id}`).sort(([, a], [, b]) => a.localeCompare(b));
  const name = (t: Target) => (t.ref ? (idea.labels[t.ref] ?? "something no longer kept") : t.text ?? "");
  const show = (t: Target) => (
    <>
      {t.ref?.startsWith("book:") ? <Link href={`/learn/book/${t.ref.slice(5)}`}>{name(t)}</Link> : t.ref?.startsWith("idea:") ? <Link href={`/ideas/${t.ref.slice(5)}`}>{name(t)}</Link> : name(t)}
      {t.ref ? <span className="muted"> ({KIND[t.ref.split(":")[0]!]})</span> : null}
      {t.url ? <> <a href={t.url} rel="noreferrer">link</a></> : null}
    </>
  );
  const drop = (t: Target) => (
    <form action={dropAction} style={{ display: "inline" }}>
      <input type="hidden" name="idea" value={idea.id} />
      <input type="hidden" name="target" value={JSON.stringify(t)} />
      <button type="submit" className="link-button" aria-label={`Drop ${name(t)}`}>drop</button>
    </form>
  );
  const pick = (
    <>
      <select name="ref" defaultValue="" className="learn-select" aria-label="Something you kept">
        <option value="">something you kept…</option>
        {choices.map(([k, v]) => <option key={k} value={k}>{v} ({KIND[k.split(":")[0]!]})</option>)}
      </select>
      <input name="text" maxLength={500} placeholder="or in your own words" aria-label="Or in your own words" />
      <input name="url" type="url" maxLength={2000} placeholder="link (optional)" aria-label="Link" />
    </>
  );
  return (
    <main className="page">
      <Nav current="ideas" eyebrow="Ideas" initial={initialOf(await currentName(user.id))} />
      <p className="note"><Link href="/ideas">Your ideas</Link></p>
      <h1 className="title">{idea.title}</h1>
      <p className="idea-statement">{idea.statement}</p>
      {idea.stated.length > 1 ? (
        <details className="note">
          <summary>Earlier wordings</summary>
          <ul className="learn-list">{idea.stated.slice(0, -1).reverse().map((s, i) => <li key={i}><span className="muted">{s.on}</span> {s.text}</li>)}</ul>
        </details>
      ) : null}
      <form action={restateAction} className="add">
        <input type="hidden" name="idea" value={idea.id} />
        <label className="lbl" htmlFor="statement">Say it again as you see it now</label>
        <textarea id="statement" name="statement" rows={3} required maxLength={20000} defaultValue={idea.statement} />
        <div className="actions"><button type="submit">Keep this wording</button></div>
      </form>

      <section className="sec" aria-labelledby="h-from">
        <div className="sec-head"><h2 id="h-from" className="lbl">What it came from</h2></div>
        {idea.from.length ? <ul className="note learn-list">{idea.from.map((x, i) => <li key={i}>{show(x)}{x.how ? `: ${x.how}` : ""} {drop(x)}</li>)}</ul> : <p className="sec-note">Nothing named yet.</p>}
        {idea.maybe.length ? (
          <>
            <p className="sec-note">Maybe from, not sure:</p>
            <ul className="note learn-list">{idea.maybe.map((x, i) => <li key={i}>{show(x)}{x.how ? `: ${x.how}` : ""} {drop(x)}</li>)}</ul>
          </>
        ) : null}
        <form action={addInputAction} className="add">
          <input type="hidden" name="idea" value={idea.id} />
          <label className="lbl">Name what fed it</label>
          <div className="choices">{pick}</div>
          <div className="choices">
            <label className="chip"><input type="radio" name="list" value="from" defaultChecked /> it fed this</label>
            <label className="chip"><input type="radio" name="list" value="maybe" /> maybe, I am not sure</label>
          </div>
          <input name="how" maxLength={1000} placeholder="how it fed the idea (optional)" aria-label="How it fed the idea" />
          <div className="actions"><button type="submit">Keep the input</button></div>
        </form>
      </section>

      <section className="sec" aria-labelledby="h-related">
        <div className="sec-head"><h2 id="h-related" className="lbl">To look at</h2></div>
        <p className="sec-note">Recommendations, from those that support the idea to those that oppose it. They are not inputs; name one as an input once it feeds your thinking.</p>
        {STANCES.map((s) => {
          const rs = idea.related.filter((r) => r.stance === s.id);
          return rs.length ? (
            <div key={s.id}>
              <p className="lbl idea-stance">{s.id}: {s.text}</p>
              <ul className="note learn-list">{rs.map((r, i) => <li key={i}>{show(r)}<span className="muted">, suggested by {r.by}</span>{r.note ? `: ${r.note}` : ""} {drop(r)}</li>)}</ul>
            </div>
          ) : null;
        })}
        <form action={addRecommendationAction} className="add">
          <input type="hidden" name="idea" value={idea.id} />
          <label className="lbl">Something that bears on it</label>
          <div className="choices">{pick}</div>
          <label className="lbl" htmlFor="stance">Where it stands</label>
          <select id="stance" name="stance" defaultValue="challenges" className="learn-select">{STANCES.map((s) => <option key={s.id} value={s.id}>{s.id}: {s.text}</option>)}</select>
          <input name="by" maxLength={100} placeholder="who suggested it (you, your AI, a friend)" aria-label="Who suggested it" />
          <input name="note" maxLength={1000} placeholder="why it bears on the idea (optional)" aria-label="Why it bears on the idea" />
          <div className="actions"><button type="submit">Keep the recommendation</button></div>
        </form>
      </section>
    </main>
  );
}
