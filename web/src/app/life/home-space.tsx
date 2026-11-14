import Link from "next/link";
import { homeCost, homeNeeds, type NeedView } from "@reflect/core/home";
import { cn } from "@/lib/utils";
import type { LifeRecord } from "@/server/life-store";
import { Head, SPACE } from "./space";
import { Upcoming } from "./upcoming";

const money = (n: number) => `$${Math.round(n).toLocaleString("en-US")}`;
const BY: Record<string, string> = { kin: "family", polis: "the polis", gifts: "gifts", nature: "nature" };
const lower = (s: string) => s.charAt(0).toLowerCase() + s.slice(1);

function NeedCard({ n }: { n: NeedView }) {
  const how = [n.how, n.by ? `by ${n.by.toLowerCase()}` : null].filter(Boolean).join(" ");
  return (
    <li>
      <Link href={`/life/home?k=${n.id}`} className="bg-card hover:bg-accent flex h-full flex-col gap-2 rounded-xl border p-4 shadow-xs transition">
        <span className="flex items-baseline justify-between gap-2"><span className="font-medium">{n.label}</span><span className={cn("text-xs", n.done === n.of ? "text-foreground" : "text-muted-foreground")}>{n.done} of {n.of} of the standard</span></span>
        <span className="text-sm">{n.stated ? how || "How it is met not said" : <span className="text-muted-foreground">Not stated yet</span>}</span>
        <span className="text-muted-foreground text-xs">{n.covered ? `Covered by ${BY[n.byId!] ?? n.by}` : n.cost !== null ? `${money(n.cost)} a month` : "Cost not given"}</span>
        {n.missing.length ? <ul className="text-muted-foreground mt-auto grid gap-0.5 text-xs">{n.missing.map((m) => <li key={m}>Not yet: {lower(m)}</li>)}</ul> : null}
      </Link>
    </li>
  );
}

/**
 * Home's workspace (Torian, 2026-11-14 17:44Z, 18:11Z): the person's own needs, how each is met and by whom
 * against its standard and what of it is not in place yet; what Home costs them and what others cover, which is what Work
 * reads to find what is left to earn; the firearms they keep; and Home's calendar.
 */
export async function HomeSpace({ userId, records, today }: { userId: string; records: LifeRecord[]; today: string }) {
  const needs = homeNeeds(records);
  const cost = homeCost(needs);
  const guns = records.filter((r) => r.kind === "firearm");
  return (
    <div className={SPACE}>
      <section className="grid gap-3">
        <Head title="Your needs" note="How each is met now and by whom, and what of its standard is not in place yet. Open one to change it." />
        <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{needs.map((n) => <NeedCard key={n.id} n={n} />)}</ul>
      </section>

      <section className="grid gap-3">
        <Head title="What Home asks of Work" note="What your needs cost you, after what others cover. Work reads this to find what is left to earn." />
        <dl className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-xl border p-4"><dt className="text-muted-foreground text-xs">You pay a month</dt><dd className="mt-1 text-2xl font-semibold">{cost.pay !== null ? money(cost.pay) : "Not known"}</dd><dd className="text-muted-foreground mt-1 text-xs">{cost.pay !== null ? `${money(cost.pay * 12)} a year` : "Give what a need costs to see it."}</dd></div>
          <div className="rounded-xl border p-4"><dt className="text-muted-foreground text-xs">Covered by others</dt><dd className="mt-1 text-base font-medium">{cost.covered.length ? cost.covered.map((n) => `${n.label} by ${BY[n.byId!] ?? n.by}`).join(", ") : "None recorded"}</dd></div>
          <div className="rounded-xl border p-4"><dt className="text-muted-foreground text-xs">In Work</dt><dd className="mt-1 text-sm"><Link href="/life/work#now" className="font-medium underline underline-offset-2">See what is left to earn</Link> and the work that covers it.</dd></div>
        </dl>
      </section>

      {guns.length ? (
        <section className="grid gap-3">
          <Head title="Firearms" note="Training and safe storage for each one you keep." />
          <ul className="divide-y rounded-xl border text-sm">
            {guns.map((g) => (
              <li key={g.id}><Link href={`/life/home/${g.id}`} className="hover:bg-accent flex items-baseline justify-between gap-3 px-4 py-2"><span className="font-medium">{String(g.body.name ?? "Firearm")}</span><span className="text-muted-foreground text-xs">{g.body.training === true ? "Trained" : "Not trained yet"}, {g.body.storage === true ? "stored safely" : "storage not in place"}</span></Link></li>
            ))}
          </ul>
        </section>
      ) : null}

      <Upcoming userId={userId} tile="home" today={today} />
    </div>
  );
}
