import type { Metadata } from "next";
import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { COURSE_SITES, courseSite } from "@reflect/core/courses";
import { Button } from "@/components/ui/button";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { listItems } from "@/server/ways-store";
import { lectureOf, lecturesOf } from "@/server/courses";
import { LearnHeader } from "../../header";
import { browseAddAction } from "../../actions";
import { LocalDay } from "../../items/local-day";
import { UnitPlayer } from "../../items/unit-player";

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
  const back = `/learn/courses?u=${encodeURIComponent(site.url)}${title ? `&t=${encodeURIComponent(title)}` : ""}`;
  const next = course.lectures[at + 1];
  const credit = COURSE_SITES[course.site];
  return (
    <main className={PAGE}>
      <LearnHeader on="browse" />
      <div className="mt-6 space-y-5">
        <p className="text-muted-foreground text-sm"><Link href={back} className="hover:underline">{title || "The course"}</Link> · Lecture {at + 1} of {course.lectures.length}</p>
        <header className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">{ref.title}</h1>
          {lecture?.overview ? <p className="max-w-prose text-sm">{lecture.overview}</p> : null}
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
            place={null} done={false} of={course.lectures.length} transcript={lecture.transcript} notes={[]} next={next ? `/learn/courses/lecture?u=${encodeURIComponent(site.url)}&n=${next.n}${title ? `&t=${encodeURIComponent(title)}` : ""}` : null}
          />
        ) : <p className="text-muted-foreground text-sm">{credit.name} did not answer just now. Try again in a moment.</p>}
        <p className="text-muted-foreground text-xs">From {credit.name}, under the {credit.license} license. Reflect plays it from the publisher and keeps no copy.</p>
      </div>
    </main>
  );
}
