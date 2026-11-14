import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { COURSE_SITES } from "@reflect/core/courses";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { getItem, listUnits } from "@/server/ways-store";
import { lectureOf, lecturesOf } from "@/server/courses";
import { LearnHeader } from "../../../../header";
import { UnitPlayer } from "../../../unit-player";

export const metadata: Metadata = { title: "Learn" };
export const dynamic = "force-dynamic";

/**
 * One lecture of a course, taken inside Reflect: the lecture plays from its publisher with the transcript beside it,
 * the place is kept for this lecture, and finishing it moves the course forward and opens the next.
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
  const next = course.lectures[at + 1];
  const prev = course.lectures[at - 1];
  const site = COURSE_SITES[course.site];
  return (
    <main className={PAGE}>
      <LearnHeader on="shelves" />
      <div className="mt-6 space-y-5">
        <p className="text-muted-foreground text-sm"><Link href={`/learn/items/${id}`} className="hover:underline">{item.title}</Link> · Lecture {at + 1} of {course.lectures.length}</p>
        <header className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">{ref.title}</h1>
          {item.by ? <p className="text-muted-foreground">{item.by}</p> : null}
          {lecture?.overview ? <p className="max-w-prose text-sm">{lecture.overview}</p> : null}
        </header>
        {lecture ? (
          <UnitPlayer
            item={id} unit={n} title={ref.title} by={item.by} cite={[item.by, `${item.title}, ${ref.title}`].filter(Boolean).join(", ")}
            youtube={lecture.youtube ?? ref.youtube} media={lecture.video ? { src: lecture.video, kind: "video" } : lecture.audio ? { src: lecture.audio, kind: "audio" } : null}
            place={unit?.place ?? null} done={Boolean(unit?.doneOn)} of={course.lectures.length}
            transcript={lecture.transcript} notes={item.notes.filter((x) => x.unit === n)} next={next ? `/learn/items/${id}/lectures/${next.n}` : null}
          />
        ) : <p className="text-muted-foreground text-sm">{site.name} did not answer just now. Try again in a moment.</p>}
        <nav className="flex justify-between gap-2 text-sm">
          {prev ? <Link href={`/learn/items/${id}/lectures/${prev.n}`} className="underline-offset-4 hover:underline">← {prev.title}</Link> : <span />}
          {next ? <Link href={`/learn/items/${id}/lectures/${next.n}`} className="text-right underline-offset-4 hover:underline">{next.title} →</Link> : null}
        </nav>
        <p className="text-muted-foreground text-xs">From {site.name}, under the {site.license} license. Reflect plays it from the publisher and keeps no copy.</p>
      </div>
    </main>
  );
}
