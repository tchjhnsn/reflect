import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeftIcon, BookCopyIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { requireUser } from "@/server/session";
import { bookOf } from "@/server/book-store";
import { compileEditionAction } from "../actions";
import { Place } from "./place";
import { BeliefLine, groups } from "../belief";

export const metadata: Metadata = { title: "The book of your life" };
export const dynamic = "force-dynamic";

/** The book of your life: what you hold, chapter by chapter through the domains of a life, compiled live. */
export default async function BookPage() {
  const user = await requireUser();
  const { chapters, unplaced } = await bookOf(user.id).catch(() => ({ chapters: [], unplaced: [] }));
  const any = chapters.some((c) => c.entries.length);
  return (
    <main className="mx-auto max-w-2xl px-6 py-6">
      <header className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Button asChild variant="ghost" size="icon" aria-label="Journal"><Link href="/journal"><ArrowLeftIcon /></Link></Button>
          <h1 className="font-serif text-2xl font-semibold tracking-tight">The book of your life</h1>
        </div>
        {any ? <form action={compileEditionAction}><Button variant="outline" size="sm"><BookCopyIcon />Compile an edition</Button></form> : null}
      </header>
      <p className="text-muted-foreground mt-3 text-sm">What you believe, domain by domain: your ideals, and the practical beliefs by which you live toward them. Select a passage in anything you write and mark it as a belief to add it.</p>
      <div className="mt-10 space-y-12">
        {chapters.map((c, i) => (
          <section key={c.domain}>
            <p className="text-muted-foreground text-xs tracking-widest uppercase">Chapter {i + 1}</p>
            <h2 className="font-serif text-2xl font-semibold"><Link href={`/journal/d/${c.domain}`} className="hover:underline underline-offset-4">{c.title}</Link></h2>
            {c.entries.length ? groups(c.entries).map((g) => (
              <div key={g.id} className="mt-5">
                <h3 className="text-muted-foreground text-xs tracking-widest uppercase">{g.title}</h3>
                <ul className="mt-3 space-y-5">{g.entries.map((e) => <li key={e.id}><BeliefLine e={e} /></li>)}</ul>
              </div>
            )) : <p className="text-muted-foreground mt-2 text-sm italic">{c.about} Nothing here yet.</p>}
          </section>
        ))}
      </div>
      {unplaced.length ? (
        <section className="mt-16 border-t pt-8">
          <h2 className="text-sm font-medium">Not yet in a chapter</h2>
          <p className="text-muted-foreground text-sm">Beliefs you keep in Learn. Place each in the domain of your life it belongs to.</p>
          <ul className="mt-4 space-y-5">
            {unplaced.map((e) => (
              <li key={e.id} className="space-y-2">
                <p className="font-serif text-lg leading-relaxed">{e.text}</p>
                <Place claimId={e.id} />
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </main>
  );
}
