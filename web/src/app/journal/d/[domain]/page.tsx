import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeftIcon, PlusIcon } from "lucide-react";
import { DOMAINS, isDomain } from "@reflect/core/book";
import { KINDS, workTitle } from "@reflect/core/works";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { requireUser } from "@/server/session";
import { domainView, idealsOf } from "@/server/book-store";
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
  const [{ chapter, works }, ideals] = await Promise.all([domainView(user.id, domain), idealsOf(user.id)]);
  const beliefs = groups(chapter.entries);
  return (
    <main className="mx-auto max-w-4xl px-6 py-6">
      <header className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Button asChild variant="ghost" size="icon" aria-label="Journal"><Link href="/journal"><ArrowLeftIcon /></Link></Button>
          <h1 className="font-serif text-2xl font-semibold tracking-tight">{d.title}</h1>
        </div>
        <Button asChild><Link href={`/journal/new?domain=${domain}`}><PlusIcon />Create</Link></Button>
      </header>
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
