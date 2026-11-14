import type { ReactNode } from "react";
import Link from "next/link";
import { BuildingIcon, ChevronRightIcon, SparklesIcon } from "lucide-react";
import { ORIGINS } from "@reflect/core/origins";
import { familyOf, parseRef, refHref, type Company, type Filing, type Job, type WikiWork } from "@reflect/core/work-browse";
import { SectionHeader, ThingHeader } from "@/components/headers";
import { Segmented } from "@/components/segmented";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { LifeRecord } from "@/server/life-store";
import { inspireAction } from "./inspire-action";

export type WorkTab = "space" | "browse" | "work" | "income" | "household" | "credential";

/** Work's header: the person's own workspace first, then browsing work, then the work they do, income and credentials. */
export function WorkHeader({ on }: { on: WorkTab | null }) {
  return (
    <SectionHeader back="/" backLabel="Home" title="Work" tabs={
      <Segmented label="Work parts" items={[
        { label: "Workspace", href: "/life/work", on: on === "space" },
        { label: "Browse", href: "/life/work/browse", on: on === "browse" },
        { label: "Work you do", href: "/life/work?k=work", on: on === "work" },
        { label: "Income", href: "/life/work?k=income", on: on === "income" },
        { label: "Intended household", href: "/life/work?k=household", on: on === "household" },
        { label: "Credentials", href: "/life/work?k=credential", on: on === "credential" },
      ]} />
    } />
  );
}

/** A titled row of cards that scrolls sideways, with a link to all of them. */
export function Row({ title, note, all, children }: { title: string; note?: string; all?: string; children: ReactNode }) {
  return (
    <section className="grid gap-3">
      <div className="flex items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
          {note ? <p className="text-muted-foreground text-sm">{note}</p> : null}
        </div>
        {all ? <Link href={all} className="text-muted-foreground hover:text-foreground flex shrink-0 items-center gap-0.5 text-sm">See all<ChevronRightIcon className="size-4" /></Link> : null}
      </div>
      <ul className="-mx-1 flex snap-x gap-3 overflow-x-auto px-1 pb-2 [&>li]:w-60 [&>li]:shrink-0 [&>li]:snap-start">{children}</ul>
    </section>
  );
}

const CARD = "bg-card hover:bg-accent flex h-full flex-col gap-2 overflow-hidden rounded-xl border shadow-xs transition";

export function JobCard({ job, forming }: { job: Job; forming?: boolean }) {
  return (
    <li>
      <Link href={`/life/work/jobs/${job.code}`} className={cn(CARD, "p-4")}>
        <span className="text-muted-foreground text-xs">{familyOf(job.code)?.label ?? "Work"}</span>
        <span className="font-medium leading-snug">{job.title}</span>
        <span className="text-muted-foreground line-clamp-3 text-sm">{forming && job.emerging[0] ? `New: ${job.emerging[0]}` : job.about}</span>
      </Link>
    </li>
  );
}

export function WikiCard({ w }: { w: WikiWork }) {
  const href = w.kind === "past" ? `/life/work/past/${w.id}` : `/life/work/imagined/${w.id}`;
  return (
    <li>
      <Link href={href} className={CARD}>
        {w.image ? <img src={`${w.image}?width=480`} alt="" loading="lazy" className="bg-muted aspect-[4/3] w-full object-cover" /> : <span className="bg-muted text-muted-foreground flex aspect-[4/3] w-full items-center justify-center text-3xl font-semibold capitalize">{w.name.slice(0, 1)}</span>}
        <span className="grid gap-1 px-4 pb-4">
          <span className="font-medium capitalize leading-snug">{w.name}</span>
          {w.from ? <span className="text-muted-foreground text-xs">From {w.from}</span> : null}
          <span className="text-muted-foreground line-clamp-2 text-sm first-letter:uppercase">{w.about}</span>
        </span>
      </Link>
    </li>
  );
}

export function CompanyCard({ c }: { c: Company }) {
  return (
    <li>
      <Link href={`/life/work/companies/${c.slug}`} className={cn(CARD, "p-4")}>
        <span className="flex items-center gap-3">
          {c.logo ? <img src={c.logo} alt="" loading="lazy" className="bg-muted size-10 shrink-0 rounded-md object-contain" /> : <span className="bg-muted flex size-10 shrink-0 items-center justify-center rounded-md"><BuildingIcon className="size-5" /></span>}
          <span className="min-w-0">
            <span className="block truncate font-medium">{c.name}</span>
            <span className="text-muted-foreground block truncate text-xs">{c.batch}{c.status !== "Active" ? ` · ${c.status}` : ""}</span>
          </span>
        </span>
        <span className="text-muted-foreground line-clamp-3 text-sm">{c.line || c.about}</span>
      </Link>
    </li>
  );
}

export function FilingList({ list }: { list: Filing[] }) {
  if (!list.length) return <p className="text-muted-foreground text-sm">None could be read from the SEC just now.</p>;
  return (
    <ul className="divide-y rounded-xl border text-sm">
      {list.map((f) => (
        <li key={`${f.name}-${f.day}`} className="flex items-baseline justify-between gap-3 px-4 py-2">
          <span className="min-w-0 truncate font-medium">{f.name}</span>
          <span className="text-muted-foreground shrink-0 text-xs">{f.where ? `${f.where} · ` : ""}{f.day}</span>
        </li>
      ))}
    </ul>
  );
}

export function InspirationCard({ r }: { r: LifeRecord }) {
  const ref = typeof r.body.ref === "string" ? parseRef(r.body.ref) : null;
  return (
    <li>
      <Link href={ref ? refHref(ref) : `/life/work/${r.id}`} className={cn(CARD, "p-4")}>
        <span className="text-muted-foreground flex items-center gap-1 text-xs"><SparklesIcon className="size-3.5" />Inspires you</span>
        <span className="font-medium capitalize leading-snug">{String(r.body.title ?? "")}</span>
        {r.body.draws ? <span className="text-muted-foreground line-clamp-3 text-sm">{String(r.body.draws)}</span> : null}
      </Link>
    </li>
  );
}

/**
 * Keeping something as an inspiration: what draws you to it, where that comes from (the seven origins), and for the
 * sake of what. The same questions as why you are learning a course, asked of work.
 */
export function Inspires({ target, title, kept }: { target: string; title: string; kept: LifeRecord | null }) {
  const on = typeof kept?.body.origins === "string" ? kept.body.origins.split(",") : [];
  return (
    <section id="inspires" className="grid scroll-mt-20 gap-3 rounded-xl border p-5">
      <h2 className="flex items-center gap-2 text-lg font-semibold"><SparklesIcon className="size-5" />{kept ? "Why it inspires you" : "Does this inspire you?"}</h2>
      <p className="text-muted-foreground text-sm">Say what draws you to it and where that comes from. It is kept with what inspires you, and over time shows what keeps pulling you.</p>
      <form action={inspireAction.bind(null, target, title)} className="grid gap-4">
        <label className="grid gap-1.5 text-sm font-medium">What draws me to it
          <Textarea name="draws" rows={3} defaultValue={typeof kept?.body.draws === "string" ? kept.body.draws : ""} />
        </label>
        <fieldset className="grid gap-1.5">
          <legend className="text-sm font-medium">Where that comes from</legend>
          <div className="flex flex-wrap gap-2">
            {ORIGINS.map((o) => (
              <label key={o.id} className="has-[:checked]:bg-foreground has-[:checked]:text-background cursor-pointer rounded-full border px-3 py-1 text-sm">
                <input type="checkbox" name="origins" value={o.id} defaultChecked={on.includes(o.id)} className="sr-only" />{o.label}
              </label>
            ))}
          </div>
        </fieldset>
        <label className="grid gap-1.5 text-sm font-medium">For the sake of what
          <Textarea name="end" rows={2} defaultValue={typeof kept?.body.end === "string" ? kept.body.end : ""} />
        </label>
        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit">{kept ? "Save" : "It inspires me"}</Button>
          {kept ? <Link href={`/life/work/${kept.id}`} className="text-muted-foreground text-sm underline-offset-2 hover:underline">Open it with what inspires you</Link> : null}
        </div>
      </form>
    </section>
  );
}

/** The person's inspirations, by what they point at. */
export function inspirationsOf(records: LifeRecord[]): Map<string, LifeRecord> {
  return new Map(records.filter((r) => r.kind === "inspiration" && typeof r.body.ref === "string").map((r) => [r.body.ref as string, r]));
}

export function Credit({ children }: { children: ReactNode }) {
  return <p className="text-muted-foreground text-xs">{children}</p>;
}

/** The top of a catalog entry's page: where it sits in Work, what kind of work it is, its name, and a link to say it inspires you. */
export function EntryTop({ kind, kindHref, title, children }: { kind: string; kindHref: string; title: string; children?: ReactNode }) {
  return (
    <>
      <ThingHeader back={kindHref} backLabel={kind} crumbs={[{ label: "Home", href: "/" }, { label: "Work", href: "/life/work" }, { label: kind, href: kindHref }]} title={title} action={<Button asChild size="sm" variant="outline"><a href="#inspires"><SparklesIcon />It inspires me</a></Button>} />
      <div className="mt-4 grid gap-2">
        <p className="text-muted-foreground text-xs tracking-wide uppercase">{kind}</p>
        <h1 id="page-title" className="scroll-mt-20 text-3xl font-semibold tracking-tight first-letter:uppercase">{title}</h1>
        {children}
      </div>
    </>
  );
}

/** A titled part of an entry's page. */
export function Part({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="grid content-start gap-2">
      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}
