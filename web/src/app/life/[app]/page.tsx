import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { appOf, isLifeApp, kindOf, type Kind, type LifeAppId } from "@reflect/core/life";
import { SectionHeader } from "@/components/headers";
import { PAGE } from "@/components/page-header";
import { Segmented } from "@/components/segmented";
import { Button } from "@/components/ui/button";
import { requireUser } from "@/server/session";
import { lifeActs, lifeRecords, type LifeAct, type LifeRecord } from "@/server/life-store";
import { Acts, RecordCard, RecordForm, States } from "../record";
import { Summary } from "../summary";
import { restoreLifeAction } from "../actions";
import { WorkHeader, type WorkTab } from "../work/parts";

export const dynamic = "force-dynamic";
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

export async function generateMetadata({ params }: { params: Promise<{ app: string }> }): Promise<Metadata> {
  const { app } = await params;
  return { title: isLifeApp(app) ? appOf(app).label : "Reflect" };
}

function Standard({ kind }: { kind: Kind }) {
  if (!kind.standard) return null;
  return <p className="bg-muted/50 rounded-lg border px-4 py-3 text-sm">{kind.standard.text} <span className="text-muted-foreground">({kind.standard.source})</span></p>;
}

/** One kind of record in an app: the one record when the kind is kept once, else the list and a way to add one. */
function KindView({ app, kind, records, acts, back }: { app: LifeAppId; kind: Kind; records: LifeRecord[]; acts: LifeAct[]; back: string }) {
  const mine = records.filter((r) => r.kind === kind.id);
  if (kind.one) {
    const r = mine[0];
    return (
      <div className="grid gap-6">
        {r ? <States kind={kind} record={r} back={back} /> : null}
        <RecordForm app={app} kind={kind} {...(r ? { record: r } : {})} back={back} />
        {r ? <Acts kind={kind} record={r} acts={acts.filter((a) => a.record === r.id)} back={back} /> : null}
      </div>
    );
  }
  return (
    <div className="grid gap-6">
      {mine.length ? (
        <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {mine.map((r) => { const last = acts.find((a) => a.record === r.id); return <RecordCard key={r.id} app={app} kind={kind} record={r} {...(last ? { last } : {})} />; })}
        </ul>
      ) : <p className="text-muted-foreground text-sm">No {kind.plural.toLowerCase()} kept yet.</p>}
      <details className="rounded-xl border p-4" open={!mine.length}>
        <summary className="cursor-pointer font-medium">Add {kind.label.toLowerCase()}</summary>
        <div className="mt-4"><RecordForm app={app} kind={kind} back={back} /></div>
      </details>
    </div>
  );
}

/**
 * An app behind the home screen's tiles (`@reflect/core/life`): its overview, where the person stands from their
 * own records, and a tab for each kind of record it keeps. A tile that names one part (Income, Food) opens its tab.
 */
export default async function LifeAppPage({ params, searchParams }: { params: Promise<{ app: string }>; searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const { app } = await params;
  if (!isLifeApp(app)) notFound();
  const sp = await searchParams;
  const a = appOf(app);
  const kind = kindOf(app, one(sp.k));
  const [records, acts] = await Promise.all([lifeRecords(user.id, app), lifeActs(user.id)]);
  const ids = new Set(records.map((r) => r.id));
  const mine = acts.filter((x) => ids.has(x.record));
  const back = `/life/${app}${kind ? `?k=${kind.id}` : ""}`;
  const removed = one(sp.removed);
  const today = new Date().toISOString().slice(0, 10);
  return (
    <main className={PAGE}>
      {app === "work" ? <WorkHeader on={(kind?.id as WorkTab | undefined) ?? null} /> : <SectionHeader back="/" backLabel="Home" title={a.label} tabs={
        <Segmented label={`${a.label} parts`} items={[{ label: "Overview", href: `/life/${app}`, on: !kind }, ...a.kinds.filter((k) => !k.hidden).map((k) => ({ label: k.plural, href: `/life/${app}?k=${k.id}`, on: kind?.id === k.id })), { label: "Calendar", href: `/calendar?tile=${app}`, on: false }]} />
      } />}
      {removed ? (
        <form action={restoreLifeAction.bind(null, removed, back)} className="bg-muted mt-4 flex items-center justify-between gap-3 rounded-lg px-4 py-2 text-sm">
          <span>Removed.</span><Button type="submit" variant="outline" size="sm">Undo</Button>
        </form>
      ) : null}
      <div className="mt-4 grid max-w-4xl gap-6">
        <p className="text-muted-foreground text-sm">{kind ? kind.intro : a.intro}</p>
        {kind ? <>{app === "work" && kind.id === "income" ? <Summary app={app} records={records} acts={mine} today={today} /> : null}<Standard kind={kind} /><KindView app={app} kind={kind} records={records} acts={mine} back={back} /></> : <Summary app={app} records={records} acts={mine} today={today} />}
        {!kind ? <p className="text-muted-foreground text-xs">Everything here is what you state. What you do is kept in your log; reflect on it in your journal.</p> : null}
      </div>
    </main>
  );
}
