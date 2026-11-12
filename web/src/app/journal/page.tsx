import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeftIcon, DownloadIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { requireUser } from "@/server/session";
import { listEntries } from "@/server/journal-store";
import { CHAT_PROMPT } from "@reflect/core/chat-journal";
import { EntryCard } from "./entry";
import { Write } from "./write";

export const metadata: Metadata = { title: "Journal" };
export const dynamic = "force-dynamic";

/** The journal: the person's own entries, newest first, and the whole of it to export (docs/design/REFLECT_UI_2026-11-12.md section 1). */
export default async function JournalPage() {
  const user = await requireUser();
  // Chats are kept apart, in the chat history (/chat/history).
  const entries = (await listEntries(user.id)).filter((e) => e.promptId !== CHAT_PROMPT);
  return (
    <main className="mx-auto max-w-3xl px-6 py-6">
      <header className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Button asChild variant="ghost" size="icon" aria-label="Home"><Link href="/"><ArrowLeftIcon /></Link></Button>
          <h1 className="text-2xl font-semibold tracking-tight">Journal</h1>
        </div>
        <Button asChild variant="outline" size="sm">
          <a href="/api/journal/export" download><DownloadIcon />Export</a>
        </Button>
      </header>
      <Card className="mt-6 py-5">
        <CardContent className="px-5">
          <Write placeholder="What happened today, in your own words..." />
        </CardContent>
      </Card>
      <section className="mt-6 space-y-4">
        {entries.length ? entries.map((e) => <EntryCard key={e.id} entry={e} />) : <p className="text-muted-foreground text-center text-sm">No entries yet. What you write here is yours alone and never published.</p>}
      </section>
    </main>
  );
}
