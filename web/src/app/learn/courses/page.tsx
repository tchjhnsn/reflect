import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { COURSE_SITES, OUTLINE, courseOutline, courseSite } from "@reflect/core/courses";
import { ExternalLinkIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { listItems, listUnits } from "@/server/ways-store";
import { courseInfoOf, syllabusTexts } from "@/server/courses";
import { listExaminations } from "@/server/examination-store";
import { reasonFor } from "@/server/learn-reason-store";
import { originLabel } from "@reflect/core/origins";
import { END_KINDS } from "@reflect/core/learn-reason";
import { LearnHeader } from "../header";
import { browseAddAction } from "../actions";
import { BookGrid } from "../book-card";
import { LocalDay } from "../items/local-day";
import { CourseShell, DOCS, courseHref, lectureHref } from "./shell";
import { SyllabusTabs } from "./syllabus-tabs";

export const metadata: Metadata = { title: "Course" };
export const dynamic = "force-dynamic";

const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/**
 * A course's own page in Reflect, laid out as a documentation page and the same for every source: its facts, then one
 * set of tabs (About with the description first and the official website, Required texts shown as books, Prerequisites,
 * Assignments, Grading, and More when the source has other parts), then every lecture, each opening in place to play. Start the course keeps it on your shelves and
 * opens it; once kept, the lectures keep your place.
 */
export default async function CoursePage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const site = courseSite(one(sp.u));
  if (!site) notFound();
  const [course, items, view, exams, reason] = await Promise.all([courseInfoOf(site.url), listItems(user.id), learnView(user.id), listExaminations(user.id, site.url), reasonFor(user.id, site.url)]);
  const kept = items.find((i) => courseSite(i.url)?.url === site.url);
  const units = kept ? await listUnits(user.id, kept.id) : [];
  const finished = new Set(units.filter((u) => u.doneOn).map((u) => u.unit));
  const credit = COURSE_SITES[site.site];
  if (!course) {
    return <main className={DOCS}><LearnHeader on="browse" /><p className="text-muted-foreground mt-6 text-sm">{credit.name} did not answer just now. Try again in a moment.</p></main>;
  }
  const title = one(sp.t) || course.title;
  const by = one(sp.by) || course.by;
  const href = (n: number) => lectureHref(site.url, n, title, kept?.id);
  const next = course.lectures.find((l) => !finished.has(String(l.n)));
  const outline = courseOutline(course);
  const texts = outline.sections.texts.length ? await syllabusTexts(outline.sections.texts.flatMap((p) => p.lines)) : [];
  const books = texts.flatMap((t) => (t.book ? [t.book] : [])).filter((b, j, a) => a.findIndex((x) => x.work === b.work) === j);
  const unfound = texts.filter((t) => !t.book && !/:$/.test(t.line)).map((t) => t.line);
  const mine = new Map(view.books.filter((b) => b.catalog?.work).map((b) => [b.catalog!.work, b.status]));
  const lectures = course.lectures.map((l) => ({ n: l.n, title: l.title, href: href(l.n), done: finished.has(String(l.n)), youtube: l.youtube }));
  const toc = [{ id: "why", label: "Why I am learning this" }, { id: "course", label: "About the course" }];
  const prereqs = outline.sections.prerequisites.flatMap((p) => p.lines);
  const textCount = books.length + unfound.length;
  const facts = [course.lectures.length ? `${course.lectures.length} lectures` : "No lecture videos", textCount ? `${textCount} required ${textCount === 1 ? "text" : "texts"}` : "No required texts listed", prereqs.length ? `Prerequisites: ${prereqs.slice(0, 2).map((p) => (p.length > 60 ? `${p.slice(0, 57)}...` : p)).join("; ")}` : "No prerequisites listed", kept ? `${finished.size} finished` : null].filter(Boolean);
  const whyHref = (start: boolean) => `/learn/courses/why?u=${encodeURIComponent(site.url)}&t=${encodeURIComponent(title)}${by ? `&by=${encodeURIComponent(by)}` : ""}${course.image ? `&img=${encodeURIComponent(course.image)}` : ""}${start ? "&go=start" : ""}`;
  const allDone = Boolean(kept && course.lectures.length && course.lectures.every((l) => finished.has(String(l.n))));
  const none = (what: string) => <p className="text-muted-foreground text-sm">{credit.name} lists no {what} for this course.</p>;
  const parts = (list: { heading: string; lines: string[] }[], titled: boolean) => (
    <div className="max-w-prose space-y-5">
      {list.map((part, i) => (
        <section key={i} className="space-y-1.5">
          {titled ? <h3 className="font-medium">{part.heading}</h3> : null}
          <ul className="space-y-1.5 leading-relaxed">{part.lines.map((line, j) => <li key={j}>{line}</li>)}</ul>
        </section>
      ))}
    </div>
  );

  return (
    <CourseShell course={title} overview={courseHref(site.url, title)} lectures={lectures} current={null} toc={toc} on={kept ? "shelves" : "browse"}>
      <header className="space-y-4">
        <p className="text-muted-foreground text-sm"><Link href="/learn/browse?m=course" className="hover:underline">Courses</Link> · {credit.name}</p>
        <div className="flex flex-col gap-5 sm:flex-row">
          {course.image ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={course.image} alt="" className="aspect-video w-full rounded-lg border object-cover sm:w-56" />
          ) : null}
          <div className="min-w-0 space-y-2">
            <h1 className="text-3xl font-semibold tracking-tight">{title}</h1>
            {by ? <p className="text-muted-foreground">{by}</p> : null}
            <p className="text-muted-foreground text-sm">{facts.join(" · ")}</p>
            <div className="flex flex-wrap gap-2 pt-1">
              {kept ? (
                <>
                  {next ? <Button asChild size="sm"><Link href={href(next.n)}>{finished.size ? "Continue" : "Start"}: {next.title}</Link></Button> : null}
                  <Button asChild size="sm" variant="outline"><Link href={`/learn/items/${kept.id}`}>On your shelves</Link></Button>
                </>
              ) : (
                reason ? <form action={browseAddAction} className="flex flex-wrap gap-2">
                  <input type="hidden" name="medium" value="course" /><input type="hidden" name="title" value={title} /><input type="hidden" name="url" value={site.url} />
                  {by ? <input type="hidden" name="by" value={by} /> : null}{course.image ? <input type="hidden" name="image" value={course.image} /> : null}<LocalDay />
                  <Button size="sm" name="go" value="start">Start the course</Button>
                  <Button size="sm" variant="outline">Want to</Button>
                </form> : (
                  <>
                    <Button asChild size="sm"><Link href={whyHref(true)}>Start the course</Link></Button>
                    <form action={browseAddAction}>
                      <input type="hidden" name="medium" value="course" /><input type="hidden" name="title" value={title} /><input type="hidden" name="url" value={site.url} />
                      {by ? <input type="hidden" name="by" value={by} /> : null}{course.image ? <input type="hidden" name="image" value={course.image} /> : null}<LocalDay />
                      <Button size="sm" variant="outline">Want to</Button>
                    </form>
                  </>
                )
              )}
              <Button asChild size="sm" variant="outline"><Link href={`/learn/examinations/new?u=${encodeURIComponent(site.url)}&t=${encodeURIComponent(title)}`}>Put yourself under examination</Link></Button>
            </div>
            {exams.length ? (
              <p className="text-sm">Your examinations: {exams.map((x, i) => <span key={x.id}>{i ? ", " : ""}<Link href={`/learn/examinations/${x.id}`} className="underline underline-offset-4">{new Date(x.createdAt).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })} ({x.status === "open" ? "open" : `closed, ${x.examiners.filter((y) => y.attests).length} attested`})</Link></span>)}</p>
            ) : null}
          </div>
        </div>
      </header>

      <section id="why" className="scroll-mt-6 rounded-lg border p-4">
        {reason ? (
          <div className="space-y-3">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="font-semibold">{allDone ? "You have finished. Read again why you began" : "Why I am learning this"}</h2>
              <div className="flex gap-3 text-sm"><Link href={whyHref(false)} className="underline underline-offset-4">Change it</Link><Link href={`/journal/w/${reason.workId}`} className="text-muted-foreground underline underline-offset-4">In your journal</Link></div>
            </div>
            <p className="max-w-prose leading-relaxed whitespace-pre-line">{reason.cause}</p>
            {reason.origins.length ? <p className="text-muted-foreground text-sm">From {reason.origins.map(originLabel).join(", ").toLowerCase()}</p> : null}
            {reason.end || reason.endKind ? (
              <p className="max-w-prose text-sm leading-relaxed"><span className="font-medium">For the sake of what: </span>{reason.end}{reason.endKind ? <span className="text-muted-foreground">{reason.end ? " " : ""}({END_KINDS.find((k) => k.id === reason.endKind)!.label.toLowerCase()})</span> : null}</p>
            ) : null}
          </div>
        ) : (
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <h2 className="font-semibold">Why I am learning this</h2>
              <p className="text-muted-foreground text-sm">Before you start: what is causing you to learn this, and for the sake of what?</p>
            </div>
            <Button asChild size="sm" variant="outline"><Link href={whyHref(!kept)}>Write it</Link></Button>
          </div>
        )}
      </section>

      <section id="course" className="scroll-mt-6">
        <SyllabusTabs tabs={OUTLINE.filter((o) => o.id !== "more" || outline.sections.more.length).map((o) => ({
          id: o.id, label: o.label,
          content: o.id === "about" ? (
            <div className="space-y-6">
              {outline.description.length ? <div className="max-w-prose space-y-3 text-[15px] leading-relaxed">{outline.description.map((p, i) => <p key={i}>{p}</p>)}</div> : <p className="text-muted-foreground text-sm">{credit.name} gives no description for this course.</p>}
              {outline.sections.about.length ? parts(outline.sections.about, true) : null}
              <dl className="grid max-w-prose gap-x-6 gap-y-2 text-sm sm:grid-cols-[9rem_1fr]">
                {by ? <><dt className="text-muted-foreground">Taught by</dt><dd>{by}</dd></> : null}
                <dt className="text-muted-foreground">From</dt><dd>{credit.name}, under the {credit.license} license</dd>
                <dt className="text-muted-foreground">Official website</dt>
                <dd><a href={site.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 underline underline-offset-4">{new URL(site.url).host}{new URL(site.url).pathname.replace(/\/$/, "")}<ExternalLinkIcon className="size-3.5" /></a></dd>
              </dl>
            </div>
          ) : o.id === "texts" ? (
            texts.length ? (
              <div className="space-y-6">
                {books.length ? <BookGrid books={books} mine={mine} cols="md:grid-cols-4" /> : null}
                {unfound.length ? (
                  <ul className={`max-w-prose space-y-2 text-sm leading-relaxed ${books.length ? "text-muted-foreground" : ""}`}>
                    {books.length ? <li className="text-foreground font-medium">Also on the syllabus</li> : null}
                    {unfound.map((line, j) => <li key={j}>{line} <Link href={`/learn/browse?q=${encodeURIComponent(line.replace(/\.\s.*$/, ""))}`} className="text-muted-foreground text-xs underline underline-offset-2">Find it in Reflect</Link></li>)}
                  </ul>
                ) : null}
              </div>
            ) : none("required texts")
          ) : outline.sections[o.id].length ? parts(outline.sections[o.id], outline.sections[o.id].length > 1 || o.id === "more") : none(o.id === "prerequisites" ? "prerequisites" : o.id === "assignments" ? "assignments" : "grading"),
        }))} />
      </section>

      <p className="text-muted-foreground text-xs">From {credit.name}, under the {credit.license} license. Reflect reads it from the publisher and keeps no copy.</p>
    </CourseShell>
  );
}
