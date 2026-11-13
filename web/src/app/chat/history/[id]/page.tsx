import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { CHAT_PROMPT, isChatId } from "@reflect/core/chat-journal";
import { PAGE, PageHeader } from "@/components/page-header";
import { Card, CardContent } from "@/components/ui/card";
import { requireUser } from "@/server/session";
import { listEntries } from "@/server/journal-store";
import { Transcript } from "../transcript";

export const metadata: Metadata = { title: "A past chat" };
export const dynamic = "force-dynamic";

/** One past chat, read-only. Continuing a past chat is not built. */
export default async function PastChatPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const { id } = await params;
  if (!isChatId(id)) notFound();
  const chat = (await listEntries(user.id, 500)).find((e) => e.id === id && e.promptId === CHAT_PROMPT);
  if (!chat) notFound();
  const day = new Date(`${chat.localDate}T12:00:00`).toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric", year: "numeric" });
  return (
    <main className={PAGE}>
      <PageHeader back="/chat/history" backLabel="Chat history" title={day}></PageHeader>
      <div className="max-w-3xl">
      <Card className="mt-6 py-5">
        <CardContent className="px-5"><Transcript entry={chat} /></CardContent>
      </Card>
      </div>
    </main>
  );
}
