import type { Metadata } from "next";
import Link from "next/link";
import { PlusIcon, SearchIcon } from "lucide-react";
import { daily, FAMILIES, formingBatches } from "@reflect/core/work-browse";
import { PAGE } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { requireUser } from "@/server/session";
import { lifeRecords } from "@/server/life-store";
import { companies, CREDIT, filingsThisWeek, fictionWork, jobs, pastWork } from "@/server/work-catalog";
import LifeAppPage from "../[app]/page";
import { CompanyCard, Credit, FilingList, InspirationCard, JobCard, Row, WikiCard, WorkHeader } from "./parts";

export const metadata: Metadata = { title: "Work" };
export const dynamic = "force-dynamic";
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";

/**
 * Work (Torian, 2026-11-14 16:30Z): first a place to browse work and be inspired, "not about income yet; it's
 * about purpose and puzzle and reason". Work that is forming now, jobs that exist and the new tasks forming in them,
 * jobs that have existed, companies young and ended, and work imagined in fiction; each opens its page, where the
 * person can keep it with why it inspires them. Their own work and income are the tabs after.
 */
export default async function WorkPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const sp = await searchParams;
  const k = one(sp.k);
  if (k && k !== "inspired") return LifeAppPage({ params: Promise.resolve({ app: "work" }), searchParams: Promise.resolve(sp) });
  const user = await requireUser();
  const records = await lifeRecords(user.id, "work");
  const inspirations = records.filter((r) => r.kind === "inspiration").reverse();
  if (k === "inspired") {
    return (
      <main className={PAGE}>
        <WorkHeader on="inspired" />
        <div className="mt-4 grid gap-6">
          <p className="text-muted-foreground text-sm">The jobs, companies and imagined work you kept, each with what draws you to it and where that comes from.</p>
          {inspirations.length ? (
            <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{inspirations.map((r) => <InspirationCard key={r.id} r={r} />)}</ul>
          ) : <p className="text-muted-foreground text-sm">Nothing yet. <Link href="/life/work" className="underline underline-offset-2">Browse work</Link> and keep what inspires you.</p>}
        </div>
      </main>
    );
  }
  const today = new Date().toISOString().slice(0, 10);
  const all = companies();
  const batches = formingBatches(all);
  const forming = all.filter((c) => batches.includes(c.batch));
  const [raising, public_] = await Promise.all([filingsThisWeek("D"), filingsThisWeek("S-1")]);
  const imagined = records.filter((r) => r.kind === "imagined");
  return (
    <main className={PAGE}>
      <WorkHeader on="browse" />
      <div className="mt-4 grid grid-cols-[minmax(0,1fr)] gap-10">
        <form action="/life/work/browse" className="flex max-w-xl gap-2" role="search">
          <Input name="q" placeholder="Search jobs, companies and imagined work" aria-label="Search work" className="h-10" />
          <Button type="submit" variant="outline" className="h-10" aria-label="Search"><SearchIcon /></Button>
        </form>
        {inspirations.length ? <Row title="What inspires you" all="/life/work?k=inspired">{inspirations.slice(0, 10).map((r) => <InspirationCard key={r.id} r={r} />)}</Row> : null}
        <Row title="Forming now" note={`Companies in Y Combinator's newest batches: ${batches.join(" and ")}.`} all="/life/work/browse?m=forming">
          {daily(forming, 12, today, "forming").map((c) => <CompanyCard key={c.slug} c={c} />)}
        </Row>
        <div className="grid grid-cols-[minmax(0,1fr)] gap-6 lg:grid-cols-2">
          <section className="grid gap-3">
            <div><h2 className="text-lg font-semibold tracking-tight">Raising money this week</h2><p className="text-muted-foreground text-sm">Companies that filed with the SEC to raise money (Form D) in the last seven days.</p></div>
            <FilingList list={raising.slice(0, 10)} />
          </section>
          <section className="grid gap-3">
            <div><h2 className="text-lg font-semibold tracking-tight">Going public this week</h2><p className="text-muted-foreground text-sm">Companies that filed to sell shares to the public (Form S-1) in the last seven days.</p></div>
            <FilingList list={public_.slice(0, 10)} />
          </section>
        </div>
        <Row title="New work forming inside jobs" note="Tasks that people in these jobs have newly reported doing." all="/life/work/browse?m=jobs&new=1">
          {daily(jobs().filter((j) => j.emerging.length), 12, today, "emerging").map((j) => <JobCard key={j.code} job={j} forming />)}
        </Row>
        <section className="grid gap-3">
          <div><h2 className="text-lg font-semibold tracking-tight">Jobs that exist</h2><p className="text-muted-foreground text-sm">{jobs().length} occupations and what people in them do, by family.</p></div>
          <ul className="flex flex-wrap gap-2">
            {FAMILIES.map((f) => <li key={f.id}><Link href={`/life/work/browse?m=jobs&family=${f.id}`} className="hover:bg-accent block rounded-full border px-3 py-1 text-sm">{f.label}</Link></li>)}
          </ul>
        </section>
        <Row title="Jobs that have existed" note="Work people once did, from gladiators to scribes." all="/life/work/browse?m=past">
          {daily(pastWork().slice(0, 200), 12, today, "past").map((w) => <WikiCard key={w.id} w={w} />)}
        </Row>
        <Row title="Companies that ended" note="Startups that closed or were bought." all="/life/work/browse?m=companies&status=ended">
          {daily(all.filter((c) => c.status === "Inactive" || c.status === "Acquired"), 12, today, "ended").map((c) => <CompanyCard key={c.slug} c={c} />)}
        </Row>
        <Row title="Imagined in fiction" note="Work that stories imagined: jobs and companies from books, films and games." all="/life/work/browse?m=imagined">
          {daily(fictionWork().slice(0, 250), 12, today, "fiction").map((w) => <WikiCard key={w.id} w={w} />)}
        </Row>
        <section className="grid gap-3">
          <div className="flex items-end justify-between gap-3">
            <div><h2 className="text-lg font-semibold tracking-tight">Imagined by you</h2><p className="text-muted-foreground text-sm">Work that does not exist yet.</p></div>
            <Button asChild variant="outline" size="sm"><Link href="/life/work?k=imagined"><PlusIcon />Imagine one</Link></Button>
          </div>
          {imagined.length ? <ul className="flex flex-wrap gap-2">{imagined.map((r) => <li key={r.id}><Link href={`/life/work/${r.id}`} className="hover:bg-accent block rounded-lg border px-3 py-2 text-sm font-medium">{String(r.body.name ?? "")}</Link></li>)}</ul> : null}
        </section>
        <Credit>Jobs from the <a href={CREDIT.onet.href} className="underline underline-offset-2">{CREDIT.onet.name}</a>. Past and imagined work from <a href={CREDIT.wikidata.href} className="underline underline-offset-2">{CREDIT.wikidata.name}</a>. Companies from <a href={CREDIT.yc.href} className="underline underline-offset-2">{CREDIT.yc.name}</a> and filings from <a href={CREDIT.sec.href} className="underline underline-offset-2">{CREDIT.sec.name}</a>.</Credit>
      </div>
    </main>
  );
}
