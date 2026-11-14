import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Button } from "@/components/ui/button";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { getExamination } from "@/server/examination-store";
import { videoReady } from "@/server/video";
import { LearnHeader } from "../../header";
import { closeExaminationAction } from "../actions";
import { ExamCall } from "../call";
import { CopyLink } from "../copy-link";
import { ExamRecord } from "../record";

export const metadata: Metadata = { title: "Examination" };
export const dynamic = "force-dynamic";

/** Your examination: the link to invite examiners, the call, and the record as it stands. */
export default async function ExaminationPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const { id } = await params;
  const e = await getExamination(user.id, id);
  if (!e) notFound();
  return (
    <main className={PAGE}>
      <LearnHeader on="shelves" />
      <div className="mt-6 max-w-3xl space-y-8">
        <header className="space-y-2">
          <p className="text-muted-foreground text-sm"><Link href={`/learn/courses?u=${encodeURIComponent(e.courseUrl)}&t=${encodeURIComponent(e.courseTitle)}`} className="hover:underline">{e.courseTitle}</Link> · {e.status === "open" ? "Open" : "Closed"}</p>
          <h1 className="text-3xl font-semibold tracking-tight">Your examination</h1>
        </header>
        {e.status === "open" ? (
          <>
            <section className="space-y-2">
              <h2 className="text-lg font-semibold">Invite your examiners</h2>
              <p className="text-muted-foreground text-sm">Send this link to the people you ask to examine you. They sign in, give their name and join; up to 12 can.</p>
              <CopyLink path={`/learn/examine/${e.token}`} />
            </section>
            <section className="space-y-2">
              <h2 className="text-lg font-semibold">The call</h2>
              {videoReady() ? <ExamCall id={e.id} /> : <p className="text-muted-foreground text-sm">Calls in Reflect are not switched on yet. Meet in person for now; your examiners can still attest here.</p>}
            </section>
          </>
        ) : null}
        <ExamRecord e={e} me={user.id} />
        {e.status === "open" ? (
          <form action={closeExaminationAction} className="space-y-2 border-t pt-6">
            <input type="hidden" name="id" value={e.id} />
            <p className="text-muted-foreground text-sm">Closing keeps the record as it stands; nobody can join or attest after.</p>
            <Button type="submit" variant="outline" size="sm">Close the examination</Button>
          </form>
        ) : null}
      </div>
    </main>
  );
}
