import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { COURSE_SITES } from "@reflect/core/courses";
import { requireUser } from "@/server/session";
import { getItem, listUnits } from "@/server/ways-store";
import { lectureOf, lecturesOf } from "@/server/courses";
import { UnitPlayer } from "../../../unit-player";
import { CourseShell, courseHref } from "../../../../courses/shell";

export const metadata: Metadata = { title: "Learn" };
export const dynamic = "force-dynamic";

/**
 * One lecture of a course, taken inside Reflect, in the course's docs layout: the lecture plays from its publisher,
 * the transcript opens beneath it, the place is kept for this lecture, and finishing it moves the course forward and
 * opens the next.
 */
export default async function LecturePage({ params }: { params: Promise<{ id: string; n: string }> }) {
  const user = await requireUser();
  const { id, n } = await params;
  const item = await getItem(user.id, id);
  if (!item) notFound();
  const course = await lecturesOf(item.url);
  const at = course?.lectures.findIndex((l) => String(l.n) === n) ?? -1;
  if (!course || at < 0) notFound();
  const ref = course.lectures[at]!;
  const [lecture, units] = await Promise.all([lectureOf(course.site, ref), listUnits(user.id, id)]);
  const unit = units.find((u) => u.unit === n);
  const done = new Set(units.filter((u) => u.doneOn).map((u) => u.unit));
  const nav = course.lectures.map((l) => ({ n: l.n, title: l.title, href: `/learn/items/${id}/lectures/${l.n}`, done: done.has(String(l.n)) }));
  const prev = nav[at - 1], next = nav[at + 1];
  const site = COURSE_SITES[course.site];
  return (
    <CourseShell course={item.title} overview={courseHref(course.url, item.title)} lectures={nav} current={ref.n} prev={prev} next={next} on="shelves">
      <div className="space-y-5">
        <header className="space-y-2">
          <p className="text-muted-foreground text-sm"><Link href={`/learn/items/${id}`} className="hover:underline">{item.title}</Link> · Lecture {at + 1} of {course.lectures.length}</p>
          <h1 className="text-3xl font-semibold tracking-tight">{ref.title}</h1>
          {item.by ? <p className="text-muted-foreground">{item.by}</p> : null}
          {lecture?.overview ? <p className="max-w-prose leading-relaxed">{lecture.overview}</p> : null}
        </header>
        {lecture ? (
          <UnitPlayer
            item={id} unit={n} title={ref.title} by={item.by} cite={[item.by, `${item.title}, ${ref.title}`].filter(Boolean).join(", ")}
            youtube={lecture.youtube ?? ref.youtube} media={lecture.video ? { src: lecture.video, kind: "video" } : lecture.audio ? { src: lecture.audio, kind: "audio" } : null}
            place={unit?.place ?? null} done={Boolean(unit?.doneOn)} of={course.lectures.length}
            transcript={lecture.transcript} notes={item.notes.filter((x) => x.unit === n)} next={next?.href ?? null}
          />
        ) : <p className="text-muted-foreground text-sm">{site.name} did not answer just now. Try again in a moment.</p>}
        <p className="text-muted-foreground text-xs">From {site.name}, under the {site.license} license. Reflect plays it from the publisher and keeps no copy.</p>
      </div>
    </CourseShell>
  );
}
