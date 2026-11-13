import type { Metadata } from "next";
import Link from "next/link";
import { PlusIcon } from "lucide-react";
import { CHAT_PROMPT, readChatPart } from "@reflect/core/chat-journal";
import { Button } from "@/components/ui/button";
import { PAGE, PageHeader } from "@/components/page-header";
import { Card } from "@/components/ui/card";
import { requireUser } from "@/server/session";
import { listEntries } from "@/server/journal-store";

export const metadata: Metadata = { title: "Chat history" };
export const dynamic = "force-dynamic";

const when = (iso: string) => new Date(iso).toLocaleString(undefined, { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

/** Past chats, newest first, apart from the journal. Each opens read-only. */
export default async function ChatHistoryPage() {
  const user = await requireUser();
  const chats = (await listEntries(user.id, 500)).filter((e) => e.promptId === CHAT_PROMPT);
  return (
    <main className={PAGE}>
      <PageHeader back="/chat" backLabel="Back to the chat" title="Chat history">
        <Button asChild size="sm"><Link href="/chat"><PlusIcon />New chat</Link></Button>
      </PageHeader>
      <div className="max-w-3xl">
      {chats.length ? (
        <Card className="mt-6 gap-0 py-0">
          {chats.map((c, i) => {
            const first = c.parts.map((p) => readChatPart(p.text)).find((m) => m.who === "you")?.text ?? "A chat";
            return (
              <Link key={c.id} href={`/chat/history/${c.id}`} className={`hover:bg-accent flex items-center justify-between gap-4 px-4 py-3 transition ${i > 0 ? "border-t" : ""}`}>
                <span className="min-w-0 truncate text-sm font-medium">{first}</span>
                <span className="text-muted-foreground shrink-0 text-xs">{c.parts[0] ? when(c.parts[0].createdAt) : c.localDate} · {c.parts.length} messages</span>
              </Link>
            );
          })}
        </Card>
      ) : (
        <p className="text-muted-foreground mt-10 text-center text-sm">No chats yet. Each chat you have is kept here.</p>
      )}
      </div>
    </main>
  );
}
