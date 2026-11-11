import type { Metadata } from "next";
import Link from "next/link";
import { requireUser } from "@/server/session";
import { currentName } from "@/server/person-store";
import { listIdeas } from "@/server/ideas-store";
import { initialOf, Nav } from "../nav";
import { newIdeaAction } from "./actions";

export const metadata: Metadata = { title: "Ideas" };
export const dynamic = "force-dynamic";

export default async function IdeasPage() {
  const user = await requireUser();
  const ideas = await listIdeas(user.id);
  return (
    <main className="page">
      <Nav current="ideas" eyebrow="Ideas" initial={initialOf(await currentName(user.id))} />
      <h1 className="title">Your ideas</h1>
      <p className="sec-note">What you make of what you take in. Each idea is in your words and names what it came from; what you study stays in <Link href="/learn">Learn</Link>.</p>

      <section className="sec" aria-labelledby="h-ideas">
        <div className="sec-head"><h2 id="h-ideas" className="lbl">Ideas</h2></div>
        {ideas.length ? (
          <ul className="note learn-list">
            {ideas.map((i) => (
              <li key={i.id}>
                <Link href={`/ideas/${i.id}`}>{i.title}</Link>
                <span className="muted"> {i.from.length} from, {i.maybe.length} maybe from, {i.related.length} to look at</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="sec-note">No ideas yet.</p>
        )}
      </section>

      <section className="sec" aria-labelledby="h-new">
        <div className="sec-head"><h2 id="h-new" className="lbl">A new idea</h2></div>
        <form action={newIdeaAction} className="add">
          <label className="lbl" htmlFor="title">A short name for it</label>
          <input id="title" name="title" required maxLength={200} />
          <label className="lbl" htmlFor="statement">The idea, in your words</label>
          <textarea id="statement" name="statement" rows={4} required maxLength={20000} />
          <label className="lbl" htmlFor="from">What it came from, one per line: a book, film, talk or anything you kept, or your own words (optional)</label>
          <textarea id="from" name="from" rows={3} maxLength={5000} />
          <div className="actions"><button type="submit" className="primary">Keep the idea</button></div>
        </form>
      </section>
    </main>
  );
}
