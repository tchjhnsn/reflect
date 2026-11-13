import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { PlusIcon } from "lucide-react";
import { DOMAINS, isDomain } from "@reflect/core/book";
import { KINDS, workTitle } from "@reflect/core/works";
import { Button } from "@/components/ui/button";
import { PAGE, PageHeader } from "@/components/page-header";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { requireUser } from "@/server/session";
import { domainView, idealsOf } from "@/server/book-store";
import { learnIn } from "@/server/subject-domain-store";
import { STANCES } from "@reflect/core/claims";
import { AddSubject, RemoveSubject } from "../../subjects";
import { BeliefLine, groups } from "../../belief";
import { MarkLater } from "../../mark-later";

export const metadata: Metadata = { title: "Journal" };
export const dynamic = "force-dynamic";

const day = (iso: string) => new Date(iso).toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" });

/** One domain of a life: the works written in it, then what you believe there, ideals first. */
export default async function DomainPage({ params }: { params: Promise<{ domain: string }> }) {
  const user = await requireUser();
  const { domain } = await params;
  if (!isDomain(domain)) notFound();
  const d = DOMAINS.find((x) => x.id === domain)!;
  const [{ chapter, works }, ideals, learn] = await Promise.all([domainView(user.id, domain), idealsOf(user.id), learnIn(user.id, domain).catch(() => ({ subjects: [], unplaced: [] }))]);
  // Claims under a placed subject show with their subject; the rest of the chapter shows below.
  const fromSubjects = new Set(learn.subjects.flatMap((s) => s.claims.map((c) => c.id)));
  const beliefs = groups(chapter.entries.filter((e) => !fromSubjects.has(e.id)));
  return (
    <main className={PAGE}>
      <PageHeader back="/journal" backLabel="Journal" title={d.title}>
        <Button asChild><Link href={`/journal/new?domain=${domain}`}><PlusIcon />Create</Link></Button>
      </PageHeader>
      <p className="text-muted-foreground mt-2 text-sm">{d.about}</p>

      <section className="mt-8">
        <h2 className="text-muted-foreground text-xs tracking-widest uppercase">Works</h2>
        {works.length ? (
          <div className="mt-3 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {works.map((w) => (
              <Link key={w.id} href={`/journal/w/${w.id}`} className="group">
                <Card className="group-hover:border-foreground/30 h-full gap-2 py-5 transition-colors">
                  <CardHeader className="px-5">
                    <CardDescription className="text-xs">{KINDS[w.kind].label} · {day(w.savedAt)}</CardDescription>
                    <CardTitle className="text-base leading-snug">{workTitle(w)}</CardTitle>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        ) : <p className="text-muted-foreground mt-3 text-sm">Nothing written here yet.</p>}
      </section>

      <section className="mt-12">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-muted-foreground text-xs tracking-widest uppercase">From Learn</h2>
          <AddSubject domain={domain} subjects={learn.unplaced} />
        </div>
        {learn.subjects.length ? (
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            {learn.subjects.map((s) => (
              <Card key={s.id} className="gap-3 py-5">
                <CardHeader className="flex flex-row items-center justify-between px-5">
                  <CardTitle className="font-serif text-lg">{s.name}</CardTitle>
                  <RemoveSubject id={s.id} name={s.name} />
                </CardHeader>
                <div className="space-y-3 px-5 text-sm">
                  {s.claims.length ? (
                    <ul className="space-y-2">
                      {s.claims.map((c) => (
                        <li key={c.id}>
                          <p className="font-serif text-base leading-snug">{c.text}</p>
                          <p className="text-muted-foreground text-xs">{STANCES.find((x) => x.id === c.stance)?.text ?? "No stance yet"}</p>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                  {s.books.length ? (
                    <ul className="text-muted-foreground space-y-0.5">
                      {s.books.map((b) => <li key={b.id}><Link href={`/learn?book=${b.id}`} className="hover:text-foreground">{b.title}</Link>{b.authors.length ? `, ${b.authors[0]}` : ""}</li>)}
                    </ul>
                  ) : null}
                  {!s.claims.length && !s.books.length ? <p className="text-muted-foreground">Nothing kept under it yet.</p> : null}
                </div>
              </Card>
            ))}
          </div>
        ) : <p className="text-muted-foreground mt-3 text-sm">{learn.unplaced.length ? "Add a subject from Learn and its claims and books show here." : "No subjects in Learn yet. Add books or claims in Learn and their subjects can be placed here."}</p>}
      </section>

      <section className="mt-12 max-w-2xl">
        <h2 className="text-muted-foreground text-xs tracking-widest uppercase">What you believe here</h2>
        {beliefs.length ? beliefs.map((g) => (
          <div key={g.id} className="mt-6">
            <h3 className="font-serif text-xl font-semibold">{g.title}</h3>
            <p className="text-muted-foreground text-xs">{g.about}</p>
            <ul className="mt-3 space-y-5">
              {g.entries.map((e) => (
                <li key={e.id} className="space-y-2">
                  <BeliefLine e={e} />
                  {g.id === "unmarked" ? <MarkLater claimId={e.id} domain={domain} ideals={ideals} /> : null}
                </li>
              ))}
            </ul>
          </div>
        )) : <p className="text-muted-foreground mt-3 text-sm">Select a passage in anything you write here and mark it as a belief: an ideal, or a practical belief that serves one.</p>}
      </section>
    </main>
  );
}
