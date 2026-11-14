import Link from "next/link";
import { appOf, incomeSummary, titleOf, type LifeAppId } from "@reflect/core/life";
import { cn } from "@/lib/utils";
import type { LifeAct, LifeRecord } from "@/server/life-store";

const money = (n: number) => `$${Math.round(n).toLocaleString("en-US")}`;

function Stat({ label: l, value, note, href }: { label: string; value: string; note?: string; href?: string }) {
  const body = (
    <>
      <span className="text-muted-foreground text-xs">{l}</span>
      <span className="text-lg font-semibold tabular-nums">{value}</span>
      {note ? <span className="text-muted-foreground text-xs">{note}</span> : null}
    </>
  );
  const box = "bg-card flex flex-col gap-0.5 rounded-xl border p-4 shadow-xs";
  return <li>{href ? <Link href={href} className={cn(box, "hover:bg-accent h-full transition")}>{body}</Link> : <div className={cn(box, "h-full")}>{body}</div>}</li>;
}

/**
 * Where the person stands in an app, from their own records only: income against the intended household on Work's
 * income tab, and in a tile with no workspace of its own yet (Legacy) what is under way in each kind. Never a score.
 */
export function Summary({ app, records, acts, today }: { app: LifeAppId; records: LifeRecord[]; acts: LifeAct[]; today: string }) {
  const a = appOf(app);
  const stats: { label: string; value: string; note?: string; href?: string }[] = [];
  if (app === "work") {
    const s = incomeSummary(records);
    stats.push({ label: "Income coming in", value: `${money(s.year)} a year`, href: `/life/work?k=income`, ...(s.unknown ? { note: `${s.unknown} by the hour without hours a week, not counted` } : {}) });
    stats.push(s.need ? { label: "Your intended household needs", value: `${money(s.need)} a year`, note: `Your income covers ${s.share}% of it`, href: `/life/work?k=household` } : { label: "Your intended household", value: "Not stated", note: "Say what it needs to see where you stand", href: `/life/work?k=household` });
    const working = records.filter((r) => r.kind === "work" && r.state === "working");
    stats.push({ label: "Work under way", value: String(working.length), note: `${working.reduce((n, r) => n + (typeof r.body.hours === "number" ? r.body.hours : 0), 0)} hours a week, paid and unpaid`, href: `/life/work?k=work` });
  } else {
    for (const k of a.kinds) {
      const mine = records.filter((r) => r.kind === k.id);
      if (k.one) {
        const r = mine[0];
        const st = k.states?.find((s) => s.id === r?.state);
        stats.push({ label: k.label, value: r ? (st?.label ?? "Stated") : "Not stated", href: `/life/${app}?k=${k.id}` });
        continue;
      }
      const active = mine.filter((r) => k.states?.find((s) => s.id === r.state)?.active);
      const under = k.states?.some((s) => s.active) ? `${active.length} ${active.length === 1 ? "is" : "are"} ${active.map((r) => k.states!.find((s) => s.id === r.state)!.label.toLowerCase()).filter((v, i, l) => l.indexOf(v) === i).join(" or ")}` : undefined;
      const next = k.fields.some((f) => f.key === "day") ? mine.filter((r) => r.state === "coming" && typeof r.body.day === "string" && r.body.day >= today).sort((x, y) => String(x.body.day).localeCompare(String(y.body.day)))[0] : undefined;
      stats.push({ label: k.plural, value: String(mine.length), href: `/life/${app}?k=${k.id}`, ...(next ? { note: `Next: ${titleOf(k, next.body)} on ${next.body.day}` } : under && active.length ? { note: under } : {}) });
    }
  }
  return <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3">{stats.map((s) => <Stat key={s.label} {...s} />)}</ul>;
}
