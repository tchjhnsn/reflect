import Link from "next/link";
import type { ReactNode } from "react";
import { friendsSpace, kinSpace, mateSpace, polisSpace } from "@reflect/core/tile-spaces";
import type { LifeAct, LifeRecord } from "@/server/life-store";
import { Head, SPACE } from "./space";
import { Upcoming } from "./upcoming";

const ago = (days: number | null) => (days === null ? null : days < 1 ? "today" : days < 14 ? `${days} ${days === 1 ? "day" : "days"} ago` : days < 60 ? `${Math.round(days / 7)} weeks ago` : `${Math.round(days / 30)} months ago`);
const CARER: Record<string, string> = { me: "you", family: "family", "paid-help": "paid help", "given-by-church-or-charity": "a church or charity", "no-one-yet": "no one yet" };
const AGAIN: Record<string, string> = { yes: "See them again", no: "Not again", "not-sure": "Not sure about again" };

function Section({ title, note, action, children }: { title: string; note: string; action?: { label: string; href: string }; children: ReactNode }) {
  return (
    <section className="grid gap-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <Head title={title} note={note} />
        {action ? <Link href={action.href} className="text-muted-foreground hover:text-foreground text-sm underline-offset-2 hover:underline">{action.label}</Link> : null}
      </div>
      {children}
    </section>
  );
}

function Rows({ rows, empty }: { rows: { key: string; href: string; main: ReactNode; side?: ReactNode }[]; empty: ReactNode }) {
  if (!rows.length) return <p className="text-muted-foreground text-sm">{empty}</p>;
  return (
    <ul className="divide-y rounded-xl border text-sm">
      {rows.map((r) => <li key={r.key}><Link href={r.href} className="hover:bg-accent flex items-baseline justify-between gap-3 px-4 py-2"><span className="min-w-0 font-medium">{r.main}</span>{r.side ? <span className="text-muted-foreground shrink-0 text-right text-xs">{r.side}</span> : null}</Link></li>)}
    </ul>
  );
}

/** Friends' workspace (Torian, 2026-11-14 17:44Z, 18:11Z): friends, who has gone longest without time together first; groups; Friends' calendar. */
export async function FriendsSpace({ userId, records, acts, today }: { userId: string; records: LifeRecord[]; acts: LifeAct[]; today: string }) {
  const s = friendsSpace(records, acts, today);
  return (
    <div className={SPACE}>
      <Section title="Your friends" note="Longest since time together or a call first." action={{ label: "Add a friend", href: "/life/friends?k=friend" }}>
        <Rows rows={s.close.map((f) => ({ key: f.id, href: `/life/friends/${f.id}`, main: <>{f.name} <span className="text-muted-foreground font-normal">· {f.state}</span></>, side: f.since === null ? "No time together kept" : `Last ${ago(f.since)}` }))} empty="No friends kept yet." />
        {s.others.length ? <p className="text-muted-foreground text-sm">Also kept: {s.others.map((f, i) => <span key={f.id}>{i ? ", " : ""}<Link href={`/life/friends/${f.id}`} className="hover:text-foreground underline-offset-2 hover:underline">{f.name}</Link> ({f.state?.toLowerCase()})</span>)}.</p> : null}
      </Section>
      <Section title="Groups you belong to" note="Teams, clubs, a congregation, and when you last went." action={{ label: "Add a group", href: "/life/friends?k=group" }}>
        <Rows rows={s.groups.map((g) => ({ key: g.id, href: `/life/friends/${g.id}`, main: <>{g.name} <span className="text-muted-foreground font-normal">· {g.state}</span></>, side: g.last ? `Last went ${g.last}` : "Not attended yet" }))} empty="No groups you are a member of yet." />
      </Section>
      <Upcoming userId={userId} tile="friends" today={today} />
    </div>
  );
}

/** Mate's workspace (Torian, 2026-11-14 17:44Z, 18:11Z): what the person is looking for, dates planned and gone on, the household they intend, and Mate's calendar. */
export async function MateSpace({ userId, records, today }: { userId: string; records: LifeRecord[]; today: string }) {
  const s = mateSpace(records, today);
  return (
    <div className={SPACE}>
      <Section title="What you are looking for" note="Your must-haves and dealbreakers, as you wrote them.">
        {s.seeking ? (
          <Link href="/life/mate?k=seeking" className="hover:bg-accent grid gap-3 rounded-xl border p-4 text-sm transition sm:grid-cols-3">
            <span><span className="text-muted-foreground block text-xs">Where you stand</span>{s.seeking.state ?? "Not said"}</span>
            <span><span className="text-muted-foreground block text-xs">Must-haves</span><span className="line-clamp-3 whitespace-pre-line">{s.seeking.must ?? "Not written"}</span></span>
            <span><span className="text-muted-foreground block text-xs">Dealbreakers</span><span className="line-clamp-3 whitespace-pre-line">{s.seeking.never ?? "Not written"}</span></span>
          </Link>
        ) : <p className="text-muted-foreground text-sm">Not written yet. <Link href="/life/mate?k=seeking" className="underline underline-offset-2">Write it</Link>.</p>}
      </Section>
      <Section title="Dates planned" note="Soonest first." action={{ label: "Add a date", href: "/life/mate?k=date" }}>
        <Rows rows={s.planned.map((d) => ({ key: d.id, href: `/life/mate/${d.id}`, main: d.who, side: [d.day, d.where].filter(Boolean).join(", ") || undefined }))} empty="None planned." />
      </Section>
      <Section title="Dates you went on" note="What you learned from each, latest first.">
        <Rows rows={s.went.slice(0, 10).map((d) => ({ key: d.id, href: `/life/mate/${d.id}`, main: <>{d.who}{d.learned ? <span className="text-muted-foreground block truncate text-xs font-normal">{d.learned}</span> : null}</>, side: [d.day, d.again ? AGAIN[d.again] : null].filter(Boolean).join(" · ") || undefined }))} empty="None kept yet." />
      </Section>
      <Section title="What Mate asks of Work" note="The household you intend sets what Work needs to provide.">
        <p className="text-sm">{s.seeking?.household ? <span className="line-clamp-3 whitespace-pre-line">{s.seeking.household}</span> : <span className="text-muted-foreground">No household written in what you are looking for.</span>} <Link href="/life/work?k=household" className="underline underline-offset-2">Your intended household in Work</Link>.</p>
      </Section>
      <Upcoming userId={userId} tile="mate" today={today} />
    </div>
  );
}

/** Kin's workspace (Torian, 2026-11-14 17:44Z, 18:11Z): kin who need care first, the rest, the hours of care the person gives (time Work cannot use), and Kin's calendar. */
export async function KinSpace({ userId, records, acts, today }: { userId: string; records: LifeRecord[]; acts: LifeAct[]; today: string }) {
  const s = kinSpace(records, acts, today);
  return (
    <div className={SPACE}>
      <Section title="Kin who need care" note="Who provides it, the care you gave in four weeks, and your last visit." action={{ label: "Add kin", href: "/life/kin?k=relative" }}>
        <Rows rows={s.care.map((k) => ({ key: k.id, href: `/life/kin/${k.id}`, main: <>{k.name}{k.relation ? <span className="text-muted-foreground font-normal"> · {k.relation}</span> : null}<span className="text-muted-foreground block text-xs font-normal">{k.state}{k.carer ? `, care by ${CARER[k.carer] ?? k.carer}` : ", who provides care not said"}</span></>, side: <>{k.careMinutes ? `${Math.round(k.careMinutes / 6) / 10} hours of care` : "No care logged"}<br />{k.visited ? `Visited ${k.visited}` : "No visit kept"}</> }))} empty="No kin needing care kept." />
      </Section>
      <Section title="Your kin" note="Kin who are independent now, kept so a plan is made before any need starts.">
        <Rows rows={s.others.map((k) => ({ key: k.id, href: `/life/kin/${k.id}`, main: <>{k.name}{k.relation ? <span className="text-muted-foreground font-normal"> · {k.relation}</span> : null}</>, side: [k.age !== null ? `${k.age} years old` : null, k.state].filter(Boolean).join(", ") || undefined }))} empty="No other kin kept." />
      </Section>
      <Section title="What Kin asks of Work" note="The care you give takes time that work cannot.">
        <p className="text-sm">{s.careHours ? <>You gave <span className="font-semibold">{s.careHours} hours a week</span> of care over four weeks.</> : <span className="text-muted-foreground">No care logged these four weeks.</span>} <Link href="/life/work#now" className="underline underline-offset-2">See the hours you have for work</Link>.</p>
      </Section>
      <Upcoming userId={userId} tile="kin" today={today} />
    </div>
  );
}

/** Polis' workspace (Torian, 2026-11-14 17:44Z, 18:11Z): where the person is a citizen, elections coming and duties due, the roles they serve in, and Polis' calendar. */
export async function PolisSpace({ userId, records, today }: { userId: string; records: LifeRecord[]; today: string }) {
  const s = polisSpace(records, today);
  return (
    <div className={SPACE}>
      <Section title="Where you are a citizen" note="Residence, citizenship and whether you are registered to vote.">
        {s.civic ? (
          <Link href="/life/polis?k=civic" className="hover:bg-accent grid gap-3 rounded-xl border p-4 text-sm transition sm:grid-cols-3">
            <span><span className="text-muted-foreground block text-xs">Residence</span>{s.civic.residence ?? "Not stated"}</span>
            <span><span className="text-muted-foreground block text-xs">Citizenship</span>{s.civic.citizenship ?? "Not stated"}{s.civic.state ? `, ${s.civic.state.toLowerCase()}` : ""}</span>
            <span><span className="text-muted-foreground block text-xs">Registered to vote</span>{s.civic.registered ? "Yes" : "Not yet"}</span>
          </Link>
        ) : <p className="text-muted-foreground text-sm">Not stated yet. <Link href="/life/polis?k=civic" className="underline underline-offset-2">Say where</Link>.</p>}
      </Section>
      <Section title="Elections coming" note="Soonest first." action={{ label: "Add an election", href: "/life/polis?k=election" }}>
        <Rows rows={s.elections.map((e) => ({ key: e.id, href: `/life/polis/${e.id}`, main: e.name || "An election", side: e.day ? (e.late ? `${e.day}, passed` : e.day) : "Day not given" }))} empty="None kept." />
      </Section>
      <Section title="Duties due" note="Jury service, taxes, licenses and permits, soonest first." action={{ label: "Add a duty", href: "/life/polis?k=duty" }}>
        <Rows rows={s.duties.map((d) => ({ key: d.id, href: `/life/polis/${d.id}`, main: d.name || "A duty", side: d.day ? (d.late ? `Was due ${d.day}` : `Due ${d.day}`) : "No date" }))} empty="Nothing due." />
      </Section>
      <Section title="Where you serve" note="Offices you hold and roles you are trusted with." action={{ label: "Add a role", href: "/life/polis?k=standing" }}>
        <Rows rows={s.serving.map((r) => ({ key: r.id, href: `/life/polis/${r.id}`, main: r.role, side: r.where ?? undefined }))} empty="No office or role kept." />
      </Section>
      <Upcoming userId={userId} tile="polis" today={today} />
    </div>
  );
}
