import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { COURSE_SITES, courseSite } from "@reflect/core/courses";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { listItems, listUnits } from "@/server/ways-store";
import { courseInfoOf } from "@/server/courses";
import { LearnHeader } from "../header";
import { browseAddAction } from "../actions";
import { LocalDay } from "../items/local-day";

export const metadata: Metadata = { title: "Course" };
export const dynamic = "force-dynamic";

const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";
const READINGS = /^(texts?|readings?|required texts?|books?)$/i;

/**
 * A course's own page in Reflect, to explore before taking it: what it is about, its syllabus (structure, texts,
 * requirements, grading), and every lecture, each of which can be watched here straight away. Start here keeps it on
 * your shelves and opens it; once kept, the lectures keep your place.
 */
export default async function CoursePage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const site = courseSite(one(sp.u));
  if (!site) notFound();
  const [course, items] = await Promise.all([courseInfoOf(site.url), listItems(user.id)]);
  const kept = items.find((i) => courseSite(i.url)?.url === site.url);
  const units = kept ? await listUnits(user.id, kept.id) : [];
  const finished = new Set(units.filter((u) => u.doneOn).map((u) => u.unit));
  const credit = COURSE_SITES[site.site];
  if (!course) {
    return (
      <main className={PAGE}><LearnHeader on="browse" /><p className="text-muted-foreground mt-6 text-sm">{credit.name} did not answer just now. Try again in a moment.</p></main>
    );
  }
  const title = one(sp.t) || course.title;
  const by = one(sp.by) || course.by;
  const lectureHref = (n: number) => (kept ? `/learn/items/${kept.id}/lectures/${n}` : `/learn/courses/lecture?u=${encodeURIComponent(site.url)}&n=${n}${title ? `&t=${encodeURIComponent(title)}` : ""}`);
  const next = course.lectures.find((l) => !finished.has(String(l.n)));
  return (
    <main className={PAGE}>
      <LearnHeader on="browse" />
      <div className="mt-6 space-y-6">
        <p className="text-muted-foreground text-sm"><Link href="/learn/browse?m=course" className="hover:underline">Courses</Link> · {credit.name}</p>
        <header className="flex flex-col gap-4 sm:flex-row">
          {course.image ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={course.image} alt="" className="aspect-video w-full rounded-lg border object-cover sm:w-64" />
          ) : null}
          <div className="min-w-0 space-y-2">
            <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
            {by ? <p className="text-muted-foreground">{by}</p> : null}
            <p className="text-muted-foreground text-sm">{course.lectures.length ? `${course.lectures.length} lectures, each watched here with its transcript` : "No lecture videos"}{kept ? ` · ${finished.size} finished` : ""}</p>
            <div className="flex flex-wrap gap-2 pt-1">
              {kept ? (
                <>
                  {next ? <Button asChild size="sm"><Link href={lectureHref(next.n)}>{finished.size ? "Continue" : "Start"}: {next.title}</Link></Button> : null}
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
        </header>

        {course.about.length ? <section className="max-w-prose space-y-3 text-sm leading-relaxed">{course.about.map((p, i) => <p key={i}>{p}</p>)}</section> : null}

        {course.syllabus.length ? (
          <Card id="syllabus" className="gap-3 py-5">
            <CardHeader className="px-5"><CardTitle className="text-sm">Syllabus</CardTitle></CardHeader>
            <CardContent className="space-y-5 px-5 text-sm">
              {course.syllabus.map((part) => (
                <section key={part.heading} className="space-y-1.5">
                  <h2 className="font-medium">{part.heading}</h2>
                  <ul className="space-y-1">
                    {part.lines.map((line, i) => (
                      <li key={i} className="leading-relaxed">
                        {line}
                        {READINGS.test(part.heading) && line.length < 200 && !/:$/.test(line) ? <> <Link href={`/learn/browse?q=${encodeURIComponent(line.replace(/\.\s.*$/, ""))}`} className="text-muted-foreground text-xs underline underline-offset-2">Find it in Reflect</Link></> : null}
                      </li>
                    ))}
                  </ul>
                </section>
              ))}
            </CardContent>
          </Card>
        ) : null}

        {course.lectures.length ? (
          <Card id="lectures" className="gap-3 py-5">
            <CardHeader className="px-5"><CardTitle className="text-sm">Lectures</CardTitle></CardHeader>
            <CardContent className="px-5">
              <ol className="divide-y text-sm">
                {course.lectures.map((l) => (
                  <li key={l.n} className="flex items-center gap-3 py-2">
                    <span className="text-muted-foreground w-6 shrink-0 text-right tabular-nums">{l.n}</span>
                    <Link href={lectureHref(l.n)} className="min-w-0 flex-1 underline-offset-4 hover:underline">{l.title}</Link>
                    <span className="text-muted-foreground shrink-0 text-xs">{finished.has(String(l.n)) ? "Finished" : "Watch"}</span>
                  </li>
                ))}
              </ol>
            </CardContent>
          </Card>
        ) : null}

        <p className="text-muted-foreground text-xs">From {credit.name}, under the {credit.license} license. Reflect reads it from the publisher and keeps no copy.</p>
      </div>
    </main>
  );
}
