import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeftIcon, DownloadIcon, PlusIcon } from "lucide-react";
import { phaseOf, STATE_LABEL } from "@reflect/core/items";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { requireUser } from "@/server/session";
import { listEntries } from "@/server/journal-store";
import { listItems } from "@/server/item-store";
import { CHAT_PROMPT } from "@reflect/core/chat-journal";
import { EntryCard } from "./entry";
import { Write } from "./write";

export const metadata: Metadata = { title: "Journal" };
export const dynamic = "force-dynamic";

/** The journal: the person's own entries, newest first, and the whole of it to export (docs/design/REFLECT_UI_2026-11-12.md section 1). */
export default async function JournalPage() {
  const user = await requireUser();
  // Chats are kept apart, in the chat history (/chat/history).
  const [all, items] = await Promise.all([listEntries(user.id), listItems(user.id).catch(() => [])]);
  const entries = all.filter((e) => e.promptId !== CHAT_PROMPT);
  const active = items.filter((x) => x.item.state !== "dropped" && x.item.state !== "done");
  const rest = items.filter((x) => !active.includes(x));
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
      <section className="mt-8">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-medium">Reflecting</h2>
          <Button asChild variant="ghost" size="sm"><Link href="/journal/items/new"><PlusIcon />Observe something</Link></Button>
        </div>
        {items.length ? (
          <Card className="gap-0 py-1">
            <ul className="divide-y">
              {[...active, ...rest].map(({ id, item }) => (
                <li key={id}>
                  <Link href={`/journal/items/${id}`} className="hover:bg-muted/50 flex items-center gap-3 px-5 py-3">
                    <span className={item.state === "dropped" || item.state === "done" ? "text-muted-foreground flex-1 text-sm" : "flex-1 text-sm"}>{item.statement}</span>
                    <span className="text-muted-foreground hidden text-xs sm:inline">{phaseOf(item.state) ?? ""}</span>
                    <Badge variant={item.state === "dropped" ? "outline" : "secondary"}>{STATE_LABEL[item.state]}</Badge>
                  </Link>
                </li>
              ))}
            </ul>
          </Card>
        ) : (
          <p className="text-muted-foreground text-sm">Nothing yet. Reflect on an entry below or on something in your log: state what you observe, then reflect, examine and plan.</p>
        )}
      </section>
      <h2 className="mt-8 mb-3 text-sm font-medium">Entries</h2>
      <section className="space-y-4">
        {entries.length ? entries.map((e) => <EntryCard key={e.id} entry={e} />) : <p className="text-muted-foreground text-center text-sm">No entries yet. What you write here is yours alone and never published.</p>}
      </section>
    </main>
  );
}
