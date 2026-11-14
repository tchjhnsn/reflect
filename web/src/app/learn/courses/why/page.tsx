import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { courseSite } from "@reflect/core/courses";
import { ORIGINS } from "@reflect/core/origins";
import { END_KINDS, REASON, REASON_HEADINGS } from "@reflect/core/learn-reason";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { reasonFor } from "@/server/learn-reason-store";
import { LearnHeader } from "../../header";
import { browseAddAction, keepReasonAction } from "../../actions";
import { LocalDay } from "../../items/local-day";
import { courseHref } from "../shell";

export const metadata: Metadata = { title: "Why I am learning this" };
export const dynamic = "force-dynamic";

const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/**
 * Before a course: what is causing you to learn it, marked with the seven origins (Rhetoric I.10), and for the
 * sake of what (Physics II.3), for its own sake or for a use. It is kept as a work of your journal. With go=start,
 * keeping it starts the course.
 */
export default async function WhyPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const sp = await searchParams;
  const site = courseSite(one(sp.u));
  if (!site) notFound();
  const title = one(sp.t) || "this course";
  const start = one(sp.go) === "start";
  const prior = await reasonFor(user.id, site.url);
  const course = { medium: "course", title, url: site.url, by: one(sp.by), image: one(sp.img) };
  const hidden = (
    <>
      {Object.entries(course).map(([k, v]) => (v ? <input key={k} type="hidden" name={k} value={v} /> : null))}
      {start ? <input type="hidden" name="go" value="start" /> : null}
      <LocalDay />
    </>
  );
  const group = (from: "not-me" | "me", label: string) => (
    <div className="space-y-1.5">
      <p className="text-muted-foreground text-xs">{label}</p>
      <div className="flex flex-wrap gap-2">
        {ORIGINS.filter((o) => o.from === from).map((o) => (
          <label key={o.id} className="has-[:checked]:bg-primary has-[:checked]:text-primary-foreground flex cursor-pointer items-center gap-2 rounded-md border px-3 py-1.5 text-sm">
            <input type="checkbox" name="origin" value={o.id} defaultChecked={prior?.origins.includes(o.id)} className="sr-only" />{o.label}
          </label>
        ))}
      </div>
    </div>
  );

  return (
    <main className={PAGE}>
      <LearnHeader on="browse" />
      <div className="mt-6 max-w-2xl space-y-6">
        <header className="space-y-2">
          <p className="text-muted-foreground text-sm"><Link href={courseHref(site.url, title)} className="hover:underline">{title}</Link></p>
          <h1 className="text-3xl font-semibold tracking-tight">Why I am learning this</h1>
          <p className="text-muted-foreground leading-relaxed">Before you start, write what is causing you to learn this and what it is for. It is kept in your journal, and you will read it again when you finish and before you are examined.</p>
        </header>
        <form action={keepReasonAction} className="space-y-6">
          {hidden}
          <div className="space-y-1.5">
            <label htmlFor="cause" className="font-medium">{REASON_HEADINGS.cause}</label>
            <Textarea id="cause" name="cause" required rows={6} maxLength={REASON.cause} defaultValue={prior?.cause} />
          </div>
          <fieldset className="space-y-3">
            <legend className="text-sm font-medium">Which of these are true of it?</legend>
            {group("not-me", "Not from me")}
            {group("me", "From me")}
          </fieldset>
          <div className="space-y-1.5">
            <label htmlFor="end" className="font-medium">{REASON_HEADINGS.end}</label>
            <Textarea id="end" name="end" rows={4} maxLength={REASON.end} defaultValue={prior?.end} />
            <div className="flex flex-wrap gap-2 pt-1">
              {END_KINDS.map((k) => (
                <label key={k.id} className="has-[:checked]:bg-primary has-[:checked]:text-primary-foreground flex cursor-pointer items-center gap-2 rounded-md border px-3 py-1.5 text-sm">
                  <input type="radio" name="endKind" value={k.id} defaultChecked={prior?.endKind === k.id} className="sr-only" />{k.label}
                </label>
              ))}
            </div>
          </div>
          <Button type="submit">{start ? "Keep it and start the course" : "Keep it"}</Button>
        </form>
        {start ? (
          <form action={browseAddAction}>
            {hidden}
            <button className="text-muted-foreground text-sm underline underline-offset-4">Start without writing it</button>
          </form>
        ) : null}
      </div>
    </main>
  );
}
