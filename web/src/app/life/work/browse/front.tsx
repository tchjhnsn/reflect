import Link from "next/link";
import { daily, FAMILIES, formingBatches } from "@reflect/core/work-browse";
import { companies, CREDIT, filingsThisWeek, fictionWork, jobs, pastWork } from "@/server/work-catalog";
import { CompanyCard, Credit, FilingList, JobCard, Row, WikiCard } from "../parts";

/** Browsing work before searching: rows that change daily, from work forming now to work imagined in fiction. */
export async function BrowseFront() {
  const today = new Date().toISOString().slice(0, 10);
  const all = companies();
  const batches = formingBatches(all);
  const forming = all.filter((c) => batches.includes(c.batch));
  const [raising, public_] = await Promise.all([filingsThisWeek("D"), filingsThisWeek("S-1")]);
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-10">
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
        <Credit>Jobs from the <a href={CREDIT.onet.href} className="underline underline-offset-2">{CREDIT.onet.name}</a>. Past and imagined work from <a href={CREDIT.wikidata.href} className="underline underline-offset-2">{CREDIT.wikidata.name}</a>. Companies from <a href={CREDIT.yc.href} className="underline underline-offset-2">{CREDIT.yc.name}</a> and filings from <a href={CREDIT.sec.href} className="underline underline-offset-2">{CREDIT.sec.name}</a>.</Credit>
    </div>
  );
}
