import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { requireUser } from "@/server/session";
import { currentName } from "@/server/person-store";
import { getUpload } from "@/server/learn-sources-store";
import { initialOf, Nav } from "../../../nav";
import { keepCoursesAction, keepLinkedInAction, keepSweepAction } from "../../actions";
import { sweepLine } from "@reflect/core/sweep";

export const metadata: Metadata = { title: "Upload" };
export const dynamic = "force-dynamic";

export default async function UploadPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const { id } = await params;
  const u = await getUpload(user.id, id);
  if (!u) notFound();
  const name = await currentName(user.id);
  return (
    <main className="page">
      <Nav current="learn" eyebrow="Learn" initial={initialOf(name)} />
      <p className="note"><Link href="/learn">What you know</Link></p>
      {u.transcript ? (
        <>
          <h1 className="title">Your transcript</h1>
          <p className="sec-note">{u.transcript.institution ? `${u.transcript.institution}. ` : ""}Check each row against your transcript and keep the right ones. The grade stays the school&apos;s; what you kept of each course is yours to say after.</p>
          <form action={keepCoursesAction}>
            <input type="hidden" name="upload" value={u.id} />
            <ul className="note learn-list" style={{ listStyle: "none", paddingLeft: 0 }}>
              {u.transcript.courses.map((c, i) => (
                <li key={i}>
                  <label>
                    <input type="checkbox" name="row" value={i} defaultChecked={!u.kept.includes(i)} disabled={u.kept.includes(i)} /> {c.term} {c.code ? `${c.code} ` : ""}{c.title}
                    {c.credits !== undefined ? `, ${c.credits} credits` : ""}{c.grade ? `, ${c.grade}` : ""}{u.kept.includes(i) ? " (kept)" : ""}
                  </label>
                </li>
              ))}
            </ul>
            {u.transcript.dropped.length ? <p className="sec-note">Left out, because they are not in the transcript as written: {u.transcript.dropped.join("; ")}</p> : null}
            <div className="actions"><button type="submit" className="primary">Keep the checked courses</button></div>
          </form>
        </>
      ) : u.linkedin ? (
        <>
          <h1 className="title">Your LinkedIn export</h1>
          <form action={keepLinkedInAction}>
            <input type="hidden" name="upload" value={u.id} />
            <fieldset className="learn-grasp">
              <legend><label><input type="checkbox" name="endorsements" defaultChecked /> Endorsements: {u.linkedin.endorsements.length}</label></legend>
              <ul className="note learn-list">{[...new Set(u.linkedin.endorsements.map((e) => e.skill))].map((s) => <li key={s}>{s}: {u.linkedin!.endorsements.filter((e) => e.skill === s).map((e) => e.by).join(", ")}</li>)}</ul>
            </fieldset>
            <fieldset className="learn-grasp">
              <legend><label><input type="checkbox" name="recommendations" defaultChecked /> Recommendations: {u.linkedin.recommendations.length}</label></legend>
              {u.linkedin.recommendations.map((r, i) => <blockquote key={i} className="note learn-recall">{r.text} <span className="muted">({r.by})</span></blockquote>)}
            </fieldset>
            <fieldset className="learn-grasp">
              <legend><label><input type="checkbox" name="certifications" defaultChecked /> Certifications: {u.linkedin.certifications.length}</label></legend>
              <ul className="note learn-list">{u.linkedin.certifications.map((c, i) => <li key={i}>{c.name}, {c.issuer}</li>)}</ul>
            </fieldset>
            <fieldset className="learn-grasp">
              <legend><label><input type="checkbox" name="skills" /> Skills you listed, as subjects for you to rate: {u.linkedin.skills.length}</label></legend>
              <p className="note">{u.linkedin.skills.join(", ")}</p>
            </fieldset>
            <div className="actions"><button type="submit" className="primary">Keep what is checked</button></div>
          </form>
        </>
      ) : u.sweep ? (
        <>
          <h1 className="title">From {u.sweep.assistant}&apos;s memory</h1>
          <p className="sec-note">Check each item against what you know and keep the right ones. Inputs go to Learn and your ideas to Ideas; an input your AI only inferred is kept as maybe from. Nothing here sets how well you know anything.</p>
          <form action={keepSweepAction}>
            <input type="hidden" name="upload" value={u.id} />
            <ul className="note learn-list" style={{ listStyle: "none", paddingLeft: 0 }}>
              {u.sweep.items.map((it, i) => (
                <li key={i}>
                  <label>
                    <input type="checkbox" name="row" value={i} defaultChecked={!u.kept.includes(i) && it.confidence !== "low"} disabled={u.kept.includes(i)} /> {sweepLine(it)}{u.kept.includes(i) ? " (kept)" : ""}
                  </label>
                </li>
              ))}
            </ul>
            {u.sweep.openQuestions.length ? <p className="sec-note">It could not tell: {u.sweep.openQuestions.join("; ")}</p> : null}
            <div className="actions"><button type="submit" className="primary">Keep the checked items</button></div>
          </form>
        </>
      ) : null}
    </main>
  );
}
