import type { Metadata } from "next";
import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { COURSE_SITES, courseSite } from "@reflect/core/courses";
import { Button } from "@/components/ui/button";
import { requireUser } from "@/server/session";
import { listItems } from "@/server/ways-store";
import { lectureOf, lecturesOf } from "@/server/courses";
import { browseAddAction } from "../../actions";
import { LocalDay } from "../../items/local-day";
import { UnitPlayer } from "../../items/unit-player";
import { CourseShell, courseHref, lectureHeader, lectureHref } from "../shell";

export const metadata: Metadata = { title: "Lecture" };
export const dynamic = "force-dynamic";

const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/** A lecture of a course not yet on your shelves, watched here with its transcript; starting the course keeps your place. */
export default async function PreviewLecturePage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const site = courseSite(one(sp.u));
  if (!site) notFound();
  const n = one(sp.n);
  const kept = (await listItems(user.id)).find((i) => courseSite(i.url)?.url === site.url);
  if (kept) redirect(`/learn/items/${kept.id}/lectures/${n}`);
  const course = await lecturesOf(site.url);
  const at = course?.lectures.findIndex((l) => String(l.n) === n) ?? -1;
  if (!course || at < 0) notFound();
  const ref = course.lectures[at]!;
  const lecture = await lectureOf(course.site, ref);
  const title = one(sp.t);
  const credit = COURSE_SITES[course.site];
  const nav = course.lectures.map((l) => ({ n: l.n, title: l.title, href: lectureHref(site.url, l.n, title, undefined), done: false }));
  const prev = nav[at - 1], next = nav[at + 1];
  return (
    <CourseShell course={title || "The course"} overview={courseHref(site.url, title)} lectures={nav} current={ref.n} prev={prev} next={next} header={lectureHeader({ course: title || "The course", overview: courseHref(site.url, title), at: at + 1, of: course.lectures.length, prev, next })}>
      <div className="space-y-5">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight">{ref.title}</h1>
          {lecture?.overview ? <p className="max-w-prose leading-relaxed">{lecture.overview}</p> : null}
        </header>
        <form action={browseAddAction} className="bg-muted/50 flex flex-wrap items-center gap-2 rounded-lg border p-3 text-sm">
          <span className="min-w-0 flex-1">Start the course to keep your place in each lecture, take notes and mark lectures finished.</span>
          <input type="hidden" name="medium" value="course" /><input type="hidden" name="title" value={title || ref.title} /><input type="hidden" name="url" value={site.url} /><LocalDay />
          <Button size="sm" name="go" value="start">Start the course</Button>
        </form>
        {lecture ? (
          <UnitPlayer
            item={null} unit={n} title={ref.title} by={null} cite={`${title || credit.name}, ${ref.title}`}
            youtube={lecture.youtube ?? ref.youtube} media={lecture.video ? { src: lecture.video, kind: "video" } : lecture.audio ? { src: lecture.audio, kind: "audio" } : null}
            place={null} done={false} of={course.lectures.length} transcript={lecture.transcript} notes={[]} next={next?.href ?? null}
          />
        ) : <p className="text-muted-foreground text-sm">{credit.name} did not answer just now. Try again in a moment.</p>}
        <p className="text-muted-foreground text-xs">From {credit.name}, under the {credit.license} license. Reflect plays it from the publisher and keeps no copy.</p>
      </div>
    </CourseShell>
  );
}
