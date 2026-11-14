import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { COURSE_SITES, courseSite, isTextsHeading } from "@reflect/core/courses";
import { Button } from "@/components/ui/button";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { listItems, listUnits } from "@/server/ways-store";
import { courseInfoOf, syllabusTexts } from "@/server/courses";
import { LearnHeader } from "../header";
import { browseAddAction } from "../actions";
import { BookGrid } from "../book-card";
import { LocalDay } from "../items/local-day";
import { CourseShell, DOCS, courseHref, lectureHref } from "./shell";
import { SyllabusTabs } from "./syllabus-tabs";
import { LectureList } from "./lecture-list";

export const metadata: Metadata = { title: "Course" };
export const dynamic = "force-dynamic";

const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/**
 * A course's own page in Reflect, laid out as a documentation page: what it is about, its syllabus in tabs (the texts
 * shown as books), and every lecture, each opening in place to play. Start the course keeps it on your shelves and
 * opens it; once kept, the lectures keep your place.
 */
export default async function CoursePage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const site = courseSite(one(sp.u));
  if (!site) notFound();
  const [course, items, view] = await Promise.all([courseInfoOf(site.url), listItems(user.id), learnView(user.id)]);
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
  const texts = await Promise.all(course.syllabus.map((p) => (isTextsHeading(p.heading) ? syllabusTexts(p.lines) : null)));
  const mine = new Map(view.books.filter((b) => b.catalog?.work).map((b) => [b.catalog!.work, b.status]));
  const lectures = course.lectures.map((l) => ({ n: l.n, title: l.title, href: href(l.n), done: finished.has(String(l.n)), youtube: l.youtube }));
  const toc = [course.about.length ? { id: "about", label: "About" } : null, course.syllabus.length ? { id: "syllabus", label: "Syllabus" } : null, lectures.length ? { id: "lectures", label: "Lectures" } : null].filter((t) => t !== null);

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
            <p className="text-muted-foreground text-sm">{course.lectures.length ? `${course.lectures.length} lectures, each watched here with its transcript` : "No lecture videos"}{kept ? ` · ${finished.size} finished` : ""}</p>
            <div className="flex flex-wrap gap-2 pt-1">
              {kept ? (
                <>
                  {next ? <Button asChild size="sm"><Link href={href(next.n)}>{finished.size ? "Continue" : "Start"}: {next.title}</Link></Button> : null}
                  <Button asChild size="sm" variant="outline"><Link href={`/learn/items/${kept.id}`}>On your shelves</Link></Button>
                </>
              ) : (
                <form action={browseAddAction} className="flex flex-wrap gap-2">
                  <input type="hidden" name="medium" value="course" /><input type="hidden" name="title" value={title} /><input type="hidden" name="url" value={site.url} />
                  {by ? <input type="hidden" name="by" value={by} /> : null}{course.image ? <input type="hidden" name="image" value={course.image} /> : null}<LocalDay />
                  <Button size="sm" name="go" value="start">Start the course</Button>
                  <Button size="sm" variant="outline">Want to</Button>
                </form>
              )}
            </div>
          </div>
        </div>
      </header>

      {course.about.length ? (
        <section id="about" className="scroll-mt-6 space-y-3">
          <h2 className="text-xl font-semibold tracking-tight">About</h2>
          <div className="max-w-prose space-y-3 leading-relaxed">{course.about.map((p, i) => <p key={i}>{p}</p>)}</div>
        </section>
      ) : null}

      {course.syllabus.length ? (
        <section id="syllabus" className="scroll-mt-6 space-y-3">
          <h2 className="text-xl font-semibold tracking-tight">Syllabus</h2>
          <SyllabusTabs tabs={course.syllabus.map((part, i) => {
            const found = texts[i];
            const books = found?.flatMap((t) => (t.book ? [t.book] : [])).filter((b, j, a) => a.findIndex((x) => x.work === b.work) === j) ?? [];
            const rest = found ? found.filter((t) => !t.book).map((t) => t.line) : part.lines;
            return {
              id: `part-${i}`, label: part.heading.replace(/:$/, ""),
              content: (
                <div className="space-y-6">
                  {books.length ? <BookGrid books={books} mine={mine} cols="md:grid-cols-4" /> : null}
                  {rest.length ? (
                    <ul className={`max-w-prose space-y-2 text-sm leading-relaxed ${books.length ? "text-muted-foreground" : ""}`}>
                      {books.length ? <li className="text-foreground font-medium">Also on the syllabus</li> : null}
                      {rest.map((line, j) => (
                        <li key={j}>
                          {line}
                          {found && !/:$/.test(line) ? <> <Link href={`/learn/browse?q=${encodeURIComponent(line.replace(/\.\s.*$/, ""))}`} className="text-muted-foreground text-xs underline underline-offset-2">Find it in Reflect</Link></> : null}
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              ),
            };
          })} />
        </section>
      ) : null}

      {lectures.length ? (
        <section id="lectures" className="scroll-mt-6 space-y-3">
          <h2 className="text-xl font-semibold tracking-tight">Lectures</h2>
          <p className="text-muted-foreground text-sm">Open a lecture here to play it, or go to its page for the transcript{kept ? ", your notes and your place" : ""}.</p>
          <LectureList courseUrl={site.url} rows={lectures} />
        </section>
      ) : null}

      <p className="text-muted-foreground text-xs">From {credit.name}, under the {credit.license} license. Reflect reads it from the publisher and keeps no copy.</p>
    </CourseShell>
  );
}
