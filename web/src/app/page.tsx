import type { Metadata } from "next";
import Link from "next/link";
import { FolderIcon, PlusIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Segmented } from "@/components/segmented";
import { cn } from "@/lib/utils";
import { requireUser } from "@/server/session";
import { learnView } from "@/server/learn-store";
import { listEntries } from "@/server/journal-store";
import { CHAT_PROMPT } from "@reflect/core/chat-journal";
import { GLYPHS, hrefOf, MINE, OURS, toneOf, type Facts, type Tone } from "./home-tiles";

export const metadata: Metadata = { title: "Reflect" };
export const dynamic = "force-dynamic";

const FILL: Record<Tone, string> = { action: "fill-foreground", thought: "fill-neutral-400", unstated: "fill-neutral-200" };
const DOT: Record<Tone, string> = { action: "bg-foreground", thought: "bg-neutral-400", unstated: "bg-neutral-200" };
const SAY: Record<Tone, string> = { action: "in action", thought: "in thought", unstated: "unstated" };

/** Home: your life as objects. MINE is the private vault; OURS, the shared spaces, is not built yet. */
export default async function Home({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const user = await requireUser();
  const ours = (await searchParams).view === "ours";
  const [learn, all] = await Promise.all([learnView(user.id), listEntries(user.id, 40)]);
  const recent = all.filter((e) => e.promptId !== CHAT_PROMPT);
  const facts: Facts = {
    learn: { underway: learn.books.some((b) => b.status === "reading"), stated: learn.books.length > 0 || learn.known.length > 0 },
    journal: { stated: recent.length > 0, open: recent.some((e) => !e.closedAt) },
  };
  const tiles = ours ? OURS : MINE;
  return (
    <main className="mx-auto flex min-h-screen max-w-5xl flex-col px-6 py-6">
      <header className="flex items-center justify-between">
        <Button asChild variant="outline" size="icon" aria-label="Ask Reflect">
          <Link href="/chat"><PlusIcon /></Link>
        </Button>
        <Segmented label="Whose" items={[{ label: "Mine", href: "/", on: !ours }, { label: "Ours", href: "/?view=ours", on: ours }]} />
        <Button asChild variant="ghost" size="icon" aria-label="Your journal" title="Journal">
          <Link href="/journal"><FolderIcon /></Link>
        </Button>
      </header>
      {ours ? <p className="text-muted-foreground mt-6 text-center text-sm">Shared spaces (a marriage, a household, a village) are not built yet.</p> : null}
      <ul className="mt-8 grid grid-cols-3 gap-4 sm:grid-cols-4 lg:grid-cols-6">
        {tiles.map((t) => {
          const tone = ours ? "unstated" : toneOf(t, facts);
          const href = ours ? null : hrefOf(t);
          const body = (
            <>
              <svg viewBox="0 0 100 100" className="size-14" aria-hidden="true"><path d={GLYPHS[t.id]} className={FILL[tone]} fillRule="evenodd" /></svg>
              <span className="text-sm font-medium">{t.label}</span>
            </>
          );
          const box = "flex aspect-square flex-col items-center justify-center gap-3 rounded-xl border bg-card shadow-xs";
          return (
            <li key={t.id}>
              {href ? (
                <Link href={href} className={cn(box, "hover:bg-accent transition")} aria-label={`${t.label}, ${SAY[tone]}`}>{body}</Link>
              ) : (
                <span className={cn(box, "text-muted-foreground")} aria-label={`${t.label}, ${SAY[tone]}, not built yet`}>{body}</span>
              )}
            </li>
          );
        })}
      </ul>
      <footer className="text-muted-foreground mt-auto flex justify-center gap-6 pt-8 text-xs">
        {(["unstated", "thought", "action"] as const).map((t) => (
          <span key={t} className="flex items-center gap-2"><i className={cn("inline-block size-2.5 rounded-full", DOT[t])} />{SAY[t].replace(/^./, (c) => c.toUpperCase())}</span>
        ))}
      </footer>
    </main>
  );
}
