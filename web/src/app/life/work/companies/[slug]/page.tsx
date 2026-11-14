import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { BuildingIcon } from "lucide-react";
import { daily } from "@reflect/core/work-browse";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { lifeRecords } from "@/server/life-store";
import { companies, companyBySlug, CREDIT } from "@/server/work-catalog";
import { CompanyCard, Credit, EntryTop, Inspires, inspirationsOf, Part } from "../../parts";

export const dynamic = "force-dynamic";
export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  return { title: companyBySlug((await params).slug)?.name ?? "Company" };
}

const STATUS: Record<string, string> = { Active: "Running", Inactive: "Closed", Acquired: "Bought", Public: "Public" };

/** A company: what it set out to do, when it began, where it stands now and companies near it. */
export default async function CompanyPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const c = companyBySlug(slug);
  if (!c) notFound();
  const user = await requireUser();
  const kept = inspirationsOf(await lifeRecords(user.id, "work")).get(`company:${slug}`) ?? null;
  const near = !c.industries[0] ? [] : daily(companies().filter((x) => x.slug !== slug && x.industries[0] === c.industries[0]), 4, slug, "near");
  const facts: [string, string][] = [
    ["Began", c.batch !== "Unspecified" ? `Y Combinator, ${c.batch}` : "Y Combinator"],
    ["Now", STATUS[c.status] ?? c.status],
    ...(c.team ? [["People", String(c.team)] as [string, string]] : []),
    ...(c.where ? [["Where", c.where] as [string, string]] : []),
  ];
  return (
    <main className={PAGE}>
      <EntryTop kind="Companies" kindHref="/life/work/browse?m=companies" title={c.name}>
        {c.line ? <p className="text-muted-foreground text-base">{c.line}</p> : null}
      </EntryTop>
      <div className="mt-6 grid max-w-4xl grid-cols-[minmax(0,1fr)] gap-8">
        <div className="flex flex-wrap items-start gap-6">
          {c.logo ? <img src={c.logo} alt="" className="bg-muted size-20 shrink-0 rounded-xl object-contain" /> : <span className="bg-muted flex size-20 shrink-0 items-center justify-center rounded-xl"><BuildingIcon className="size-8" /></span>}
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            {facts.flatMap(([k, v]) => [<dt key={`${k}-t`} className="text-muted-foreground">{k}</dt>, <dd key={`${k}-d`}>{v}</dd>])}
          </dl>
        </div>
        {c.about ? <Part title="What it does"><p className="leading-relaxed">{c.about}</p></Part> : null}
        {c.industries.length ? (
          <ul className="flex flex-wrap gap-2">{c.industries.map((i) => <li key={i}><Link href={`/life/work/browse?m=companies&industry=${encodeURIComponent(i)}`} className="hover:bg-accent block rounded-full border px-3 py-1 text-sm">{i}</Link></li>)}</ul>
        ) : null}
        <Inspires target={`company:${slug}`} title={c.name} kept={kept} />
        {near.length ? <Part title={`Other companies in ${c.industries[0]}`}><ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{near.map((x) => <CompanyCard key={x.slug} c={x} />)}</ul></Part> : null}
        <Credit>From <a href={`https://www.ycombinator.com/companies/${slug}`} className="underline underline-offset-2">{CREDIT.yc.name}</a>.</Credit>
      </div>
    </main>
  );
}
