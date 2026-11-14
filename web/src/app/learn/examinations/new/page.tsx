import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { courseSite } from "@reflect/core/courses";
import { EXAM } from "@reflect/core/examination";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { LearnHeader } from "../../header";
import { createExaminationAction } from "../actions";

export const metadata: Metadata = { title: "Put yourself under examination" };
export const dynamic = "force-dynamic";

const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/** Putting yourself under examination on a course: what you will answer for, and the name your examiners will see. */
export default async function NewExaminationPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  await requireUser();
  const sp = await searchParams;
  const site = courseSite(one(sp.u));
  if (!site) notFound();
  const title = one(sp.t) || "This course";
  return (
    <main className={PAGE}>
      <LearnHeader on="shelves" />
      <div className="mt-6 max-w-2xl space-y-6">
        <header className="space-y-2">
          <p className="text-muted-foreground text-sm"><Link href={`/learn/courses?u=${encodeURIComponent(site.url)}&t=${encodeURIComponent(title)}`} className="hover:underline">{title}</Link></p>
          <h1 className="text-3xl font-semibold tracking-tight">Put yourself under examination</h1>
          <p className="text-muted-foreground leading-relaxed">Say what you know from this course and will answer for. You then invite people to examine you, on a call in Reflect or in person. There is no grade: each examiner writes in their own words whether they attest that you know it, and that is kept with your record.</p>
        </header>
        <form action={createExaminationAction} className="space-y-4">
          <input type="hidden" name="url" value={site.url} /><input type="hidden" name="title" value={title} />
          <div className="space-y-1.5">
            <label htmlFor="name" className="text-sm font-medium">Your name, as your examiners will see it</label>
            <Input id="name" name="name" required maxLength={EXAM.name} autoComplete="name" />
          </div>
          <div className="space-y-1.5">
            <label htmlFor="claim" className="text-sm font-medium">What you will answer for</label>
            <Textarea id="claim" name="claim" required rows={6} maxLength={EXAM.claim} placeholder="For example: why death is bad for the one who dies, and the strongest objections to that view." />
          </div>
          <Button type="submit">Put yourself under examination</Button>
        </form>
      </div>
    </main>
  );
}
