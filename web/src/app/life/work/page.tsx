import type { Metadata } from "next";
import Link from "next/link";
import { redirect } from "next/navigation";
import { PenLineIcon, PlusIcon, SearchIcon, SparklesIcon } from "lucide-react";
import { familyOf, ZONES, type Suggestion } from "@reflect/core/work-browse";
import { PAGE } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { requireUser } from "@/server/session";
import { workspace, type Wanted } from "@/server/work-space";
import { CREDIT } from "@/server/work-catalog";
import LifeAppPage from "../[app]/page";
import { Credit, WorkHeader } from "./parts";
import { addOwnAction, hoursAction } from "./space-actions";

export const metadata: Metadata = { title: "Work" };
export const dynamic = "force-dynamic";
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";
const money = (n: number) => `$${Math.round(n).toLocaleString("en-US")}`;
const BY: Record<string, string> = { kin: "family", polis: "the polis", gifts: "gifts", nature: "nature" };
const TYPE: Record<string, string> = { job: "Job", career: "Career", venture: "Venture", company: "Company", other: "Work" };
const STATE: Record<string, string> = { drawn: "Drawn to it", exploring: "Exploring it", pursuing: "Pursuing it", imagined: "Imagined", forming: "Forming", real: "Real now" };
const SELECT = "border-input bg-background h-9 rounded-md border px-3 text-sm";

function WantedCard({ w }: { w: Wanted }) {
  const kind = w.record.kind === "imagined" ? "Imagined" : TYPE[String(w.record.body.type ?? "")] ?? (w.job ? "Job" : "Work");
  return (
    <li>
      <Link href={`/life/work/${w.record.id}`} className="bg-card hover:bg-accent flex h-full flex-col gap-2 rounded-xl border p-4 shadow-xs transition">
        <span className="text-muted-foreground text-xs">{kind}{w.record.state ? ` · ${STATE[w.record.state] ?? w.record.state}` : ""}</span>
        <span className="font-medium leading-snug first-letter:uppercase">{w.title}</span>
        {typeof w.record.body.draws === "string" && w.record.body.draws ? <span className="text-muted-foreground line-clamp-2 text-sm">{w.record.body.draws}</span> : null}
        <span className="text-muted-foreground mt-auto text-xs">
          {w.entries ? `${w.entries} ${w.entries === 1 ? "entry" : "entries"}, last ${w.lastEntry}` : "No entries yet"}
          {w.learning.length ? ` · learning ${w.learning.length} toward it` : ""}
        </span>
      </Link>
    </li>
  );
}

function SuggestionCard({ s, hours }: { s: Suggestion; hours: number | null }) {
  return (
    <li>
      <Link href={`/life/work/jobs/${s.job.code}`} className="bg-card hover:bg-accent flex h-full flex-col gap-2 rounded-xl border p-4 shadow-xs transition">
        <span className="text-muted-foreground text-xs">{familyOf(s.job.code)?.label ?? "Work"}</span>
        <span className="font-medium leading-snug">{s.job.title}</span>
        <span className="text-sm">{s.pay ? `${money(s.pay)} a year${hours && hours < 40 ? ` at ${hours} hours a week` : ""}` : "Pay not published"}{s.job.zone ? <span className="text-muted-foreground"> · {ZONES[s.job.zone]}</span> : null}</span>
        <ul className="text-muted-foreground mt-auto grid gap-0.5 text-xs">{s.why.map((y) => <li key={y}>{y}</li>)}</ul>
      </Link>
    </li>
  );
}

const GRID = "grid gap-3 sm:grid-cols-2 lg:grid-cols-4";

/**
 * Work's workspace (Torian, 2026-11-14 17:44Z): the work this person wants, with what moves them toward each; what
 * their needs leave to earn now, after what family covers, and the time learning takes; and the jobs that fit now
 * and that what they are learning leads to. Browse, income and the rest are the tabs after it.
 */
export default async function WorkPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const sp = await searchParams;
  const k = one(sp.k);
  if (k === "inspired") redirect("/life/work");
  if (k) return LifeAppPage({ params: Promise.resolve({ app: "work" }), searchParams: Promise.resolve(sp) });
  const user = await requireUser();
  const s = await workspace(user.id);
  const needNote = s.need.from === "home" ? "What your needs from Home cost you a year" : s.need.from === "household" ? "What your intended household needs a year" : null;
  return (
    <main className={PAGE}>
      <WorkHeader on="space" />
      <div className="mt-4 grid grid-cols-[minmax(0,1fr)] gap-10">
        <section className="grid gap-3">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold tracking-tight">Work you want</h2>
              <p className="text-muted-foreground text-sm">Jobs, careers and ventures you like. Open one to write what moves you toward it.</p>
            </div>
            <Button asChild variant="outline" size="sm"><Link href="/life/work/browse"><SearchIcon />Find work</Link></Button>
          </div>
          {s.wanted.length ? <ul className={GRID}>{s.wanted.map((w) => <WantedCard key={w.record.id} w={w} />)}</ul> : <p className="text-muted-foreground text-sm">Nothing yet. Browse work and press It inspires me on what you like, or add your own.</p>}
          <form action={addOwnAction} className="flex flex-wrap items-center gap-2">
            <Input name="title" placeholder="Add your own: a job, a career or a venture" aria-label="Name of the work" className="h-9 max-w-sm" required />
            <select name="type" aria-label="Kind of work" className={SELECT} defaultValue="venture">
              <option value="job">Job</option><option value="career">Career</option><option value="venture">Venture</option><option value="other">Other</option>
            </select>
            <Button type="submit" variant="outline" size="sm"><PlusIcon />Add</Button>
            <Link href="/life/work?k=imagined" className="text-muted-foreground hover:text-foreground text-sm underline-offset-2 hover:underline">Imagine work that does not exist yet</Link>
          </form>
        </section>

        <section id="now" className="grid scroll-mt-20 gap-3">
          <div>
            <h2 className="text-lg font-semibold tracking-tight">Right now</h2>
            <p className="text-muted-foreground text-sm">What your other tiles say about the work you need this season.</p>
          </div>
          <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-xl border p-4">
              <dt className="text-muted-foreground text-xs">Needs a year</dt>
              <dd className="mt-1 text-2xl font-semibold">{s.need.pay !== null ? money(s.need.pay) : "Not known"}</dd>
              <dd className="text-muted-foreground mt-1 text-xs">{needNote ?? <>Give what your needs cost in <Link href="/life/home" className="underline underline-offset-2">Home</Link>, or your <Link href="/life/work?k=household" className="underline underline-offset-2">intended household</Link>.</>}{s.need.unknown.length && s.need.from === "home" ? `. No cost given yet for ${s.need.unknown.join(", ").toLowerCase()}.` : ""}</dd>
            </div>
            <div className="rounded-xl border p-4">
              <dt className="text-muted-foreground text-xs">Covered by others</dt>
              <dd className="mt-1 text-base font-medium">{s.need.covered.length ? s.need.covered.map((c) => `${c.label} by ${BY[c.by] ?? c.by}`).join(", ") : "None recorded"}</dd>
              <dd className="text-muted-foreground mt-1 text-xs">From who meets each need in <Link href="/life/home" className="underline underline-offset-2">Home</Link>.</dd>
            </div>
            <div className="rounded-xl border p-4">
              <dt className="text-muted-foreground text-xs">Left to earn a year</dt>
              <dd className="mt-1 text-2xl font-semibold">{s.left !== null ? money(s.left) : "Not known"}</dd>
              <dd className="text-muted-foreground mt-1 text-xs">After {money(s.income.year)} a year coming in (<Link href="/life/work?k=income" className="underline underline-offset-2">income</Link>){s.income.unknown ? `, ${s.income.unknown} not counted` : ""}.</dd>
            </div>
            <div className="rounded-xl border p-4">
              <dt className="text-muted-foreground text-xs">Time</dt>
              <dd className="mt-1 text-sm">{s.learningHours ? <>Learning <span className="font-semibold">{s.learningHours} hours a week</span>, from your sessions in <Link href="/learn" className="underline underline-offset-2">Learn</Link> over four weeks.</> : <>No learning sessions kept in <Link href="/learn" className="underline underline-offset-2">Learn</Link> these four weeks.</>}</dd>
              <dd className="mt-2">
                <form action={hoursAction} className="flex flex-wrap items-center gap-2 text-sm">
                  <label className="flex items-center gap-2"><Input name="hours" type="number" min={0} max={100} defaultValue={s.hours ?? ""} className="h-8 w-16" aria-label="Hours a week you can give to work" />hours a week for work</label>
                  <Button type="submit" variant="outline" size="sm" className="h-8">Save</Button>
                </form>
              </dd>
            </div>
          </dl>
        </section>

        <section className="grid gap-3">
          <div>
            <h2 className="text-lg font-semibold tracking-tight">Work for now</h2>
            <p className="text-muted-foreground text-sm">Jobs you could start without long preparation that cover what is left to earn{s.hours ? ` at ${s.hours} hours a week` : ""}, nearest first to the work you want and what you are learning.</p>
          </div>
          {s.now.length ? <ul className={GRID}>{s.now.map((x) => <SuggestionCard key={x.job.code} s={x} hours={s.hours} />)}</ul> : <p className="text-muted-foreground text-sm">No job covers what is left to earn at those hours. Try more hours, or add work you want.</p>}
        </section>

        <section className="grid gap-3">
          <div>
            <h2 className="text-lg font-semibold tracking-tight">Work for later</h2>
            <p className="text-muted-foreground text-sm">Where what you are learning, or want to learn, leads.</p>
          </div>
          {s.later.length ? <ul className={GRID}>{s.later.map((x) => <SuggestionCard key={x.job.code} s={x} hours={null} />)}</ul> : <p className="text-muted-foreground text-sm">Add what you are learning or want to learn in <Link href="/learn" className="underline underline-offset-2">Learn</Link>, and the work it leads to shows here.</p>}
        </section>

        <section className="grid gap-3">
          <div>
            <h2 className="text-lg font-semibold tracking-tight">What moves you toward it</h2>
            <p className="text-muted-foreground text-sm">Your journal entries toward the work you want, newest first.</p>
          </div>
          {s.entries.length ? (
            <ul className="divide-y rounded-xl border text-sm">
              {s.entries.map((e) => (
                <li key={e.id} className="flex items-baseline justify-between gap-3 px-4 py-2">
                  <Link href={`/journal/w/${e.id}`} className="min-w-0 truncate font-medium hover:underline"><PenLineIcon className="mr-1.5 inline size-3.5" />{e.title}</Link>
                  <span className="text-muted-foreground shrink-0 text-xs"><Link href={`/life/work/${e.towardId}`} className="hover:underline">{e.toward}</Link> · {e.day}</span>
                </li>
              ))}
            </ul>
          ) : <p className={cn("text-muted-foreground text-sm")}>{s.wanted.length ? <>Open work you want and press <span className="font-medium">Write an entry</span>.</> : <><SparklesIcon className="mr-1 inline size-3.5" />Add work you want first.</>}</p>}
        </section>

        <Credit>Jobs from the <a href={CREDIT.onet.href} className="underline underline-offset-2">{CREDIT.onet.name}</a>; pay is the national median from <a href={CREDIT.bls.href} className="underline underline-offset-2">{CREDIT.bls.name}</a>.</Credit>
      </div>
    </main>
  );
}
