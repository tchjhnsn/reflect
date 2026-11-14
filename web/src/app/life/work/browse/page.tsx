import type { Metadata } from "next";
import Link from "next/link";
import { SearchIcon } from "lucide-react";
import { batchOrder, FAMILIES, formingBatches, industries, ranked, type Company, type Job, type WikiWork } from "@reflect/core/work-browse";
import { PAGE } from "@/components/page-header";
import { Segmented } from "@/components/segmented";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { requireUser } from "@/server/session";
import { lifeRecords } from "@/server/life-store";
import { companies, filingsThisWeek, fictionWork, jobs, pastWork } from "@/server/work-catalog";
import { CompanyCard, FilingList, JobCard, WikiCard, WorkHeader } from "../parts";
import { BrowseFront } from "./front";

export const metadata: Metadata = { title: "Browse work" };
export const dynamic = "force-dynamic";
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v) ?? "";
const PAGE_SIZE = 48;
const MODES = [
  { id: "all", label: "Everything" }, { id: "jobs", label: "Jobs that exist" }, { id: "past", label: "Jobs that have existed" },
  { id: "companies", label: "Companies" }, { id: "forming", label: "Forming now" }, { id: "imagined", label: "Imagined" },
] as const;
type Mode = (typeof MODES)[number]["id"];

const GRID = "grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4";

function Chips({ items }: { items: { label: string; href: string; on: boolean }[] }) {
  return (
    <ul className="flex flex-wrap gap-2">
      {items.map((it) => (
        <li key={it.href}><Link href={it.href} aria-current={it.on ? "true" : undefined} className={cn("block rounded-full border px-3 py-1 text-sm", it.on ? "bg-foreground text-background" : "hover:bg-accent")}>{it.label}</Link></li>
      ))}
    </ul>
  );
}

/** Every catalog of work, searchable and filtered: jobs that exist, jobs that have existed, companies, what is forming and imagined work. */
export default async function BrowseWorkPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  await requireUser();
  const sp = await searchParams;
  const q = one(sp.q).trim();
  const front = !q && !one(sp.m);
  const m = (MODES.find((x) => x.id === one(sp.m))?.id ?? "all") as Mode;
  const n = Math.max(PAGE_SIZE, Math.min(2000, Number(one(sp.n)) || PAGE_SIZE));
  const family = one(sp.family);
  const industry = one(sp.industry);
  const status = one(sp.status);
  const fresh = one(sp.new) === "1";
  const url = (over: Record<string, string | null>) => {
    const p = new URLSearchParams();
    const base: Record<string, string> = { m, q, family, industry, status, new: fresh ? "1" : "" };
    for (const [k, v] of Object.entries({ ...base, ...over })) if (v) p.set(k, v);
    return `/life/work/browse?${p}`;
  };
  const more = (total: number) => (total > n ? <div><Button asChild variant="outline"><Link href={url({ n: String(n + PAGE_SIZE) })} scroll={false}>More</Link></Button></div> : null);

  const findJobs = (list: Job[]) => ranked(list, q, (j) => j.title, (j) => [j.about, ...j.titles, ...j.tasks, ...j.emerging]);
  const findWiki = (list: WikiWork[]) => ranked(list, q, (w) => w.name, (w) => [w.about, w.from]);
  const findCos = (list: Company[]) => ranked(list, q, (c) => c.name, (c) => [c.line, c.about, ...c.industries]);
  const all = companies();

  let body: React.ReactNode;
  if (front) {
    body = <BrowseFront />;
  } else if (m === "all") {
    const sections: { title: string; mode: Mode; count: number; cards: React.ReactNode }[] = [];
    const j = findJobs(jobs());
    sections.push({ title: "Jobs that exist", mode: "jobs", count: j.length, cards: j.slice(0, 8).map((x) => <JobCard key={x.code} job={x} />) });
    const p = findWiki(pastWork());
    sections.push({ title: "Jobs that have existed", mode: "past", count: p.length, cards: p.slice(0, 8).map((x) => <WikiCard key={x.id} w={x} />) });
    const c = findCos(all);
    sections.push({ title: "Companies", mode: "companies", count: c.length, cards: c.slice(0, 8).map((x) => <CompanyCard key={x.slug} c={x} />) });
    const f = findWiki(fictionWork());
    sections.push({ title: "Imagined in fiction", mode: "imagined", count: f.length, cards: f.slice(0, 8).map((x) => <WikiCard key={x.id} w={x} />) });
    const shown = sections.filter((s) => s.count);
    body = shown.length ? (
      <div className="grid gap-8">
        {shown.map((s) => (
          <section key={s.mode} className="grid gap-3">
            <div className="flex items-end justify-between gap-3">
              <h2 className="text-lg font-semibold tracking-tight">{s.title} <span className="text-muted-foreground text-sm font-normal">{s.count}</span></h2>
              {s.count > 8 ? <Link href={url({ m: s.mode })} className="text-muted-foreground hover:text-foreground text-sm">See all</Link> : null}
            </div>
            <ul className={GRID}>{s.cards}</ul>
          </section>
        ))}
      </div>
    ) : <p className="text-muted-foreground text-sm">Nothing matches “{q}”.</p>;
  } else if (m === "jobs") {
    let list = jobs();
    if (family) list = list.filter((j) => j.code.startsWith(family));
    if (fresh) list = list.filter((j) => j.emerging.length);
    list = q ? findJobs(list) : [...list].sort((a, b) => a.title.localeCompare(b.title));
    body = (
      <div className="grid gap-4">
        <Chips items={[{ label: "All families", href: url({ family: null }), on: !family }, ...FAMILIES.map((f) => ({ label: f.label, href: url({ family: f.id }), on: family === f.id }))]} />
        <Chips items={[{ label: "Every job", href: url({ new: null }), on: !fresh }, { label: "With new work forming", href: url({ new: "1" }), on: fresh }]} />
        <p className="text-muted-foreground text-sm">{list.length} jobs</p>
        <ul className={GRID}>{list.slice(0, n).map((j) => <JobCard key={j.code} job={j} forming={fresh} />)}</ul>
        {more(list.length)}
      </div>
    );
  } else if (m === "past" || m === "imagined") {
    const src = m === "past" ? pastWork() : fictionWork();
    const list = q ? findWiki(src) : src;
    let mine: React.ReactNode = null;
    if (m === "imagined") {
      const user = await requireUser();
      const imagined = (await lifeRecords(user.id, "work")).filter((r) => r.kind === "imagined");
      mine = (
        <section className="grid gap-3">
          <div className="flex items-end justify-between gap-3">
            <h2 className="text-lg font-semibold tracking-tight">Imagined by you</h2>
            <Button asChild variant="outline" size="sm"><Link href="/life/work?k=imagined">Imagine one</Link></Button>
          </div>
          {imagined.length ? <ul className="flex flex-wrap gap-2">{imagined.map((r) => <li key={r.id}><Link href={`/life/work/${r.id}`} className="hover:bg-accent block rounded-lg border px-3 py-2 text-sm font-medium">{String(r.body.name ?? "")}</Link></li>)}</ul> : <p className="text-muted-foreground text-sm">Work that does not exist yet, as you picture it.</p>}
        </section>
      );
    }
    body = (
      <div className="grid gap-6">
        {mine}
        <section className="grid gap-3">
          {m === "imagined" ? <h2 className="text-lg font-semibold tracking-tight">Imagined in fiction</h2> : null}
          <p className="text-muted-foreground text-sm">{list.length} {m === "past" ? "kinds of work people once did" : "jobs and companies from books, films and games"}</p>
          <ul className={GRID}>{list.slice(0, n).map((w) => <WikiCard key={w.id} w={w} />)}</ul>
          {more(list.length)}
        </section>
      </div>
    );
  } else if (m === "companies") {
    let list = all;
    if (status === "ended") list = list.filter((c) => c.status === "Inactive" || c.status === "Acquired");
    else if (status === "public") list = list.filter((c) => c.status === "Public");
    else if (status === "active") list = list.filter((c) => c.status === "Active");
    if (industry) list = list.filter((c) => c.industries[0] === industry);
    list = q ? findCos(list) : [...list].sort((a, b) => batchOrder(b.batch) - batchOrder(a.batch));
    body = (
      <div className="grid gap-4">
        <Chips items={[["", "All"], ["active", "Running"], ["public", "Public"], ["ended", "Ended"]].map(([id, label]) => ({ label: label!, href: url({ status: id || null }), on: status === id }))} />
        <Chips items={[{ label: "Every industry", href: url({ industry: null }), on: !industry }, ...industries(all).slice(0, 16).map((i) => ({ label: i.name, href: url({ industry: i.name }), on: industry === i.name }))]} />
        <p className="text-muted-foreground text-sm">{list.length} companies</p>
        <ul className={GRID}>{list.slice(0, n).map((c) => <CompanyCard key={c.slug} c={c} />)}</ul>
        {more(list.length)}
      </div>
    );
  } else {
    const batches = formingBatches(all);
    const list = findCos(all.filter((c) => batches.includes(c.batch)));
    const [raising, public_] = await Promise.all([filingsThisWeek("D"), filingsThisWeek("S-1")]);
    body = (
      <div className="grid gap-8">
        <section className="grid gap-3">
          <div><h2 className="text-lg font-semibold tracking-tight">New companies</h2><p className="text-muted-foreground text-sm">{list.length} companies in Y Combinator&apos;s newest batches: {batches.join(" and ")}.</p></div>
          <ul className={GRID}>{list.slice(0, n).map((c) => <CompanyCard key={c.slug} c={c} />)}</ul>
          {more(list.length)}
        </section>
        <div className="grid grid-cols-[minmax(0,1fr)] gap-6 lg:grid-cols-2">
          <section className="grid gap-3"><h2 className="text-lg font-semibold tracking-tight">Raising money this week</h2><FilingList list={raising.slice(0, 40)} /></section>
          <section className="grid gap-3"><h2 className="text-lg font-semibold tracking-tight">Going public this week</h2><FilingList list={public_.slice(0, 40)} /></section>
        </div>
      </div>
    );
  }

  return (
    <main className={PAGE}>
      <WorkHeader on="browse" />
      <div className="mt-4 grid grid-cols-[minmax(0,1fr)] gap-6">
        <form action="/life/work/browse" className="flex max-w-xl gap-2" role="search">
          {front ? null : <input type="hidden" name="m" value={m} />}
          <Input name="q" defaultValue={q} placeholder="Search jobs, companies and imagined work" aria-label="Search work" className="h-10" />
          <Button type="submit" variant="outline" className="h-10" aria-label="Search"><SearchIcon /></Button>
        </form>
        <div className="-mx-1 overflow-x-auto px-1"><Segmented label="Kinds of work" items={MODES.map((x) => ({ label: x.label, href: url({ m: x.id, family: null, industry: null, status: null, new: null }), on: !front && m === x.id }))} /></div>
        {body}
      </div>
    </main>
  );
}
