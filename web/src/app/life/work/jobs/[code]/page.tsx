import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { familyOf, INTERESTS, ZONES } from "@reflect/core/work-browse";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { lifeRecords } from "@/server/life-store";
import { CREDIT, jobByCode } from "@/server/work-catalog";
import { Credit, EntryTop, Inspires, inspirationsOf, JobCard, Part } from "../../parts";
import { Toward } from "../../toward";

export const dynamic = "force-dynamic";
export async function generateMetadata({ params }: { params: Promise<{ code: string }> }): Promise<Metadata> {
  return { title: jobByCode((await params).code)?.title ?? "Job" };
}

/** A job that exists: what people in it do, the new work forming in it, what it asks of you and the jobs near it. */
export default async function JobPage({ params }: { params: Promise<{ code: string }> }) {
  const { code } = await params;
  const job = jobByCode(code);
  if (!job) notFound();
  const user = await requireUser();
  const kept = inspirationsOf(await lifeRecords(user.id, "work")).get(`job:${code}`) ?? null;
  const fam = familyOf(code);
  const related = job.related.map(jobByCode).filter((j) => j !== null).slice(0, 8);
  return (
    <main className={PAGE}>
      <EntryTop kind="Jobs that exist" kindHref="/life/work/browse?m=jobs" title={job.title}>
        {fam ? <p className="text-sm"><Link href={`/life/work/browse?m=jobs&family=${fam.id}`} className="text-muted-foreground underline-offset-2 hover:underline">{fam.label}</Link></p> : null}
      </EntryTop>
      <div className="mt-6 grid max-w-4xl grid-cols-[minmax(0,1fr)] gap-8">
        <p className="text-base leading-relaxed">{job.about}</p>
        {job.emerging.length ? (
          <Part title="New work forming in it">
            <p className="text-muted-foreground text-sm">Tasks that people in this job have newly reported doing.</p>
            <ul className="grid list-disc gap-1 pl-5">{job.emerging.map((t) => <li key={t}>{t}</li>)}</ul>
          </Part>
        ) : null}
        {job.tasks.length ? <Part title="What people in it do"><ul className="grid list-disc gap-1 pl-5">{job.tasks.map((t) => <li key={t}>{t}</li>)}</ul></Part> : null}
        <div className="grid gap-6 sm:grid-cols-2">
          {job.interests.length ? (
            <Part title="The work it favours">
              <ul className="grid gap-2 text-sm">{job.interests.map((i) => <li key={i}><span className="font-medium">{i}</span><span className="text-muted-foreground block">{INTERESTS[i] ?? ""}</span></li>)}</ul>
            </Part>
          ) : null}
          {job.zone ? <Part title="Preparation"><p className="text-sm">{ZONES[job.zone]}</p></Part> : null}
          {job.pay ? <Part title="Pay"><p className="text-sm">${job.pay.toLocaleString("en-US")} a year, the national median{job.people ? `, across ${job.people.toLocaleString("en-US")} people in this work` : ""}.</p></Part> : null}
        </div>
        {job.titles.length ? <Part title="What people call it"><p className="text-muted-foreground text-sm">{job.titles.join(", ")}</p></Part> : null}
        <Inspires target={`job:${code}`} title={job.title} kept={kept} />
        {kept ? <Toward userId={user.id} record={kept} back={`/life/work/jobs/${code}`} /> : null}
        {related.length ? <Part title="Jobs near it"><ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{related.map((j) => <JobCard key={j.code} job={j} />)}</ul></Part> : null}
        <Credit>From the <a href={`https://www.onetonline.org/link/summary/${code}`} className="underline underline-offset-2">{CREDIT.onet.name}</a>{job.pay ? <>; pay from <a href={CREDIT.bls.href} className="underline underline-offset-2">{CREDIT.bls.name}</a></> : null}.</Credit>
      </div>
    </main>
  );
}
