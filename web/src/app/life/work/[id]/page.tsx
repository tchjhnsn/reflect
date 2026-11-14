import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowRightIcon } from "lucide-react";
import { kindOf, titleOf } from "@reflect/core/life";
import { parseRef, refHref } from "@reflect/core/work-browse";
import { ThingHeader } from "@/components/headers";
import { PAGE } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { requireUser } from "@/server/session";
import { lifeRecord } from "@/server/life-store";
import { WANTED_KINDS } from "@/server/work-space";
import LifeRecordPage from "../../[app]/[id]/page";
import { RecordForm, States } from "../../record";
import { removeLifeAction } from "../../actions";
import { Toward } from "../toward";

export const metadata: Metadata = { title: "Work" };
export const dynamic = "force-dynamic";

/** Work the person wants (from a catalog or their own): its state, what moves them toward it, what they learn toward it, and its fields. Other Work records open as any record does. */
export default async function WorkRecordPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const user = await requireUser();
  const record = await lifeRecord(user.id, id);
  if (!record || record.app !== "work") notFound();
  if (!WANTED_KINDS.has(record.kind)) return LifeRecordPage({ params: Promise.resolve({ app: "work", id }) });
  const kind = kindOf("work", record.kind)!;
  const title = titleOf(kind, record.body);
  const ref = typeof record.body.ref === "string" ? parseRef(record.body.ref) : null;
  const back = `/life/work/${id}`;
  return (
    <main className={PAGE}>
      <ThingHeader back="/life/work" backLabel="Workspace" crumbs={[{ label: "Home", href: "/" }, { label: "Work", href: "/life/work" }]} title={title} />
      <div className="mt-4 grid max-w-4xl grid-cols-[minmax(0,1fr)] gap-8">
        <div className="grid gap-2">
          <p className="text-muted-foreground text-xs tracking-wide uppercase">{record.kind === "imagined" ? "Imagined work" : "Work you want"}</p>
          <h1 id="page-title" className="scroll-mt-20 text-3xl font-semibold tracking-tight first-letter:uppercase">{title}</h1>
          {ref ? <div><Button asChild variant="outline" size="sm"><Link href={refHref(ref)}>About this work<ArrowRightIcon /></Link></Button></div> : null}
        </div>
        <States kind={kind} record={record} back={back} />
        <Toward userId={user.id} record={record} back={back} />
        <details className="rounded-xl border p-5">
          <summary className="cursor-pointer font-semibold">{record.kind === "imagined" ? "What it is" : "Why it inspires you"}</summary>
          <div className="mt-4"><RecordForm app="work" kind={kind} record={record} back={back} /></div>
        </details>
        <form action={removeLifeAction.bind(null, record.id)}><Button type="submit" variant="ghost" size="sm">Remove</Button></form>
      </div>
    </main>
  );
}
