import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeftIcon, BookCopyIcon } from "lucide-react";
import type { BookEntry } from "@reflect/core/book";
import { Button } from "@/components/ui/button";
import { requireUser } from "@/server/session";
import { bookOf } from "@/server/book-store";
import { compileEditionAction } from "../actions";
import { Place } from "./place";

export const metadata: Metadata = { title: "The book of your life" };
export const dynamic = "force-dynamic";

const day = (iso: string) => new Date(iso.length === 10 ? `${iso}T12:00:00` : iso).toLocaleDateString(undefined, { month: "long", year: "numeric" });

function Entry({ e }: { e: BookEntry }) {
  const from = [
    e.holders.length ? `with ${e.holders.map((h) => (h.citation ? `${h.name}, ${h.citation}` : h.name)).join("; ")}` : null,
  ].filter(Boolean);
  return (
    <li className="space-y-1">
      <p className={e.state === "once" ? "text-muted-foreground font-serif text-lg leading-relaxed line-through decoration-1" : "font-serif text-lg leading-relaxed"}>{e.text}</p>
      <p className="text-muted-foreground text-xs">
        {e.state === "held" ? `Held since ${day(e.since)}` : `I once held this, ${day(e.since)} to ${day(e.until!)}`}
        {from.length ? ` · ${from.join(" · ")}` : null}
        {e.source ? <> · from <Link href={`/journal/w/${e.source.workId}`} className="underline underline-offset-2">{e.source.title}</Link></> : null}
      </p>
    </li>
  );
}

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
      <p className="text-muted-foreground mt-3 text-sm">What you hold, gathered from your journal and your claims. Select a sentence in anything you write and mark it &ldquo;I hold this&rdquo; to add it.</p>
      <div className="mt-10 space-y-12">
        {chapters.map((c, i) => (
          <section key={c.domain}>
            <p className="text-muted-foreground text-xs tracking-widest uppercase">Chapter {i + 1}</p>
            <h2 className="font-serif text-2xl font-semibold">{c.title}</h2>
            {c.entries.length ? <ul className="mt-4 space-y-5">{c.entries.map((e) => <Entry key={e.id} e={e} />)}</ul> : <p className="text-muted-foreground mt-2 text-sm italic">{c.about} Nothing here yet.</p>}
          </section>
        ))}
      </div>
      {unplaced.length ? (
        <section className="mt-16 border-t pt-8">
          <h2 className="text-sm font-medium">Not yet in a chapter</h2>
          <p className="text-muted-foreground text-sm">Beliefs you keep in Learn. Place each where it belongs in your life.</p>
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
