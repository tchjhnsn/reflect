import type { Metadata } from "next";
import { notFound, redirect } from "next/navigation";
import { EXAM } from "@reflect/core/examination";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { PAGE } from "@/components/page-header";
import { currentUser } from "@/server/session";
import { examinationByToken } from "@/server/examination-store";
import { videoReady } from "@/server/video";
import { LearnHeader } from "../../header";
import { attestAction, joinExaminationAction } from "../../examinations/actions";
import { ExamCall } from "../../examinations/call";
import { ExamRecord } from "../../examinations/record";

export const metadata: Metadata = { title: "Examine" };
export const dynamic = "force-dynamic";

/**
 * Asked to examine someone: what they answer for, joining with your name, the call, and your attestation in your own
 * words, written once.
 */
export default async function ExaminePage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = await params;
  const user = await currentUser();
  if (!user) redirect(`/sign-in?next=${encodeURIComponent(`/learn/examine/${token}`)}`);
  const e = await examinationByToken(token);
  if (!e) notFound();
  if (e.candidate === user.id) redirect(`/learn/examinations/${e.id}`);
  const me = e.examiners.find((x) => x.userId === user.id);
  return (
    <main className={PAGE}>
      <LearnHeader on="shelves" />
      <div className="mt-6 max-w-3xl space-y-8">
        <header className="space-y-2">
          <p className="text-muted-foreground text-sm">{e.courseTitle} · {e.status === "open" ? "Open" : "Closed"}</p>
          <h1 className="text-3xl font-semibold tracking-tight">{e.name} asks you to examine them</h1>
          <p className="text-muted-foreground leading-relaxed">Question them on what they say they know from {e.courseTitle}. It is not a test with a grade: afterwards, say in your own words whether you attest that they know it.</p>
        </header>
        {!me ? (
          e.status === "open" ? (
            <form action={joinExaminationAction} className="max-w-md space-y-3">
              <input type="hidden" name="token" value={token} />
              <label htmlFor="name" className="text-sm font-medium">Your name, as {e.name} will see it</label>
              <Input id="name" name="name" required maxLength={EXAM.name} autoComplete="name" />
              <Button type="submit">Join as an examiner</Button>
            </form>
          ) : <p className="text-muted-foreground text-sm">This examination is closed.</p>
        ) : (
          <>
            {e.status === "open" ? (
              <section className="space-y-2">
                <h2 className="text-lg font-semibold">The call</h2>
                {videoReady() ? <ExamCall id={e.id} /> : <p className="text-muted-foreground text-sm">Calls in Reflect are not switched on yet. Meet in person for now, and attest here.</p>}
              </section>
            ) : null}
            {!me.attestation && e.status === "open" ? (
              <form action={attestAction} className="space-y-3">
                <input type="hidden" name="id" value={e.id} /><input type="hidden" name="token" value={token} />
                <h2 className="text-lg font-semibold">Your attestation</h2>
                <Textarea name="words" required rows={5} maxLength={EXAM.attestation} aria-label="Your attestation" placeholder="What they showed they know, and where they could not yet answer." />
                <p className="text-muted-foreground text-sm">Written once; it cannot be changed after.</p>
                <div className="flex flex-wrap gap-2">
                  <Button type="submit" name="attests" value="yes">I attest that they know it</Button>
                  <Button type="submit" name="attests" value="no" variant="outline">Not yet</Button>
                </div>
              </form>
            ) : null}
          </>
        )}
        <ExamRecord e={e} me={user.id} hideOthers={!me?.attestation && e.status === "open"} />
      </div>
    </main>
  );
}
