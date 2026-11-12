import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeftIcon } from "lucide-react";
import { CHAT_PROMPT } from "@reflect/core/chat-journal";
import { Button } from "@/components/ui/button";
import { LocalTime } from "@/components/local-time";
import { requireUser } from "@/server/session";
import { listEntries } from "@/server/journal-store";

export const metadata: Metadata = { title: "Journal entry" };
export const dynamic = "force-dynamic";

/** An entry written before the editor, shown as it was kept. */
export default async function EntryPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const { id } = await params;
  const entry = (await listEntries(user.id)).find((e) => e.id === id && e.promptId !== CHAT_PROMPT);
  if (!entry) notFound();
  return (
    <main className="mx-auto max-w-2xl px-6 py-6">
      <header className="flex items-center gap-3">
        <Button asChild variant="ghost" size="icon" aria-label="Journal"><Link href="/journal"><ArrowLeftIcon /></Link></Button>
        <p className="text-muted-foreground text-sm">Journal entry · {new Date(`${entry.localDate}T12:00:00`).toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric", year: "numeric" })}</p>
      </header>
      <article className="mt-10 space-y-6 font-serif text-lg leading-relaxed">
        {entry.parts.map((p) => (
          <div key={p.seq}>
            <p className="text-muted-foreground font-sans text-xs"><LocalTime iso={p.createdAt} only="time" /></p>
            <p className="whitespace-pre-wrap">{p.text}</p>
          </div>
        ))}
      </article>
    </main>
  );
}
