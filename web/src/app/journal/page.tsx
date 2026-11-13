import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeftIcon, BookOpenIcon, DownloadIcon, PlusIcon } from "lucide-react";
import { KINDS, workTitle } from "@reflect/core/works";
import { DOMAINS } from "@reflect/core/book";
import { CHAT_PROMPT } from "@reflect/core/chat-journal";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { requireUser } from "@/server/session";
import { listWorks } from "@/server/work-store";
import { listEntries } from "@/server/journal-store";
import { ImportButton } from "./import";

export const metadata: Metadata = { title: "Journal" };
export const dynamic = "force-dynamic";

const day = (iso: string) => new Date(`${iso.slice(0, 10)}T12:00:00`).toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" });
const excerpt = (body: string) => body.replace(/^#+\s.*$/gm, "").replace(/^\s*(?:[-*+]|\d+\.)\s+/gm, "").replace(/[*_>`]/g, "").replace(/\s+/g, " ").trim().slice(0, 160);

/** The journal: the library of everything the person writes, and a way to start something new. */
export default async function JournalPage() {
  const user = await requireUser();
  const [works, entries] = await Promise.all([listWorks(user.id).catch(() => []), listEntries(user.id)]);
  // Entries written before the editor, kept as they were (chats live in the chat history).
  const earlier = entries.filter((e) => e.promptId !== CHAT_PROMPT && e.parts.length);
  return (
    <main className="mx-auto max-w-4xl px-6 py-6">
      <header className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Button asChild variant="ghost" size="icon" aria-label="Home"><Link href="/"><ArrowLeftIcon /></Link></Button>
          <h1 className="text-2xl font-semibold tracking-tight">Journal</h1>
        </div>
        <div className="flex items-center gap-2">
          <Button asChild variant="ghost" size="icon" aria-label="Export"><a href="/api/journal/export" download><DownloadIcon /></a></Button>
          <Button asChild variant="ghost"><Link href="/journal/book"><BookOpenIcon />Book</Link></Button>
          <ImportButton />
          <Button asChild><Link href="/journal/new"><PlusIcon />Create</Link></Button>
        </div>
      </header>
      <nav aria-label="Domains" className="mt-6 flex flex-wrap gap-1.5">
        {DOMAINS.map((d) => {
          const n = works.filter((w) => w.domain === d.id).length;
          return (
            <Link key={d.id} href={`/journal/d/${d.id}`} className="hover:border-foreground/30 rounded-full border px-3 py-1 text-sm transition-colors">
              {d.title}{n ? <span className="text-muted-foreground ml-1.5 text-xs">{n}</span> : null}
            </Link>
          );
        })}
      </nav>
      {works.length || earlier.length ? (
        <section className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {works.map((w) => (
            <Link key={w.id} href={`/journal/w/${w.id}`} className="group">
              <Card className="group-hover:border-foreground/30 h-full gap-2 py-5 transition-colors">
                <CardHeader className="px-5">
                  <CardDescription className="text-xs">{KINDS[w.kind].label}{w.domain ? ` in ${DOMAINS.find((d) => d.id === w.domain)!.title}` : ""} · {day(w.savedAt)}</CardDescription>
                  <CardTitle className="text-base leading-snug">{workTitle(w)}</CardTitle>
                </CardHeader>
                {excerpt(w.body) ? <CardContent className="text-muted-foreground line-clamp-3 px-5 text-sm">{excerpt(w.body)}</CardContent> : null}
              </Card>
            </Link>
          ))}
          {earlier.map((e) => (
            <Link key={e.id} href={`/journal/entries/${e.id}`} className="group">
              <Card className="group-hover:border-foreground/30 h-full gap-2 py-5 transition-colors">
                <CardHeader className="px-5">
                  <CardDescription className="text-xs">Journal entry · {day(e.localDate)}</CardDescription>
                  <CardTitle className="line-clamp-2 text-base leading-snug">{e.parts[0]!.text.split("\n")[0]}</CardTitle>
                </CardHeader>
              </Card>
            </Link>
          ))}
        </section>
      ) : (
        <div className="mt-24 text-center">
          <p className="text-muted-foreground">Your journal holds everything you write: entries, essays, letters, plans.</p>
          <Button asChild className="mt-4"><Link href="/journal/new"><PlusIcon />Create your first</Link></Button>
        </div>
      )}
    </main>
  );
}
