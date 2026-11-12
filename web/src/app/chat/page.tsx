import type { Metadata } from "next";
import { requireUser } from "@/server/session";
import { currentName } from "@/server/person-store";
import { initialOf, Nav } from "../nav";
import { Chat } from "./chat";

export const metadata: Metadata = { title: "Chat" };
export const dynamic = "force-dynamic";

/** The plus button's chat (docs/design/REFLECT_UI_2026-11-12.md section 5). It is not saved to the journal yet. */
export default async function ChatPage() {
  const user = await requireUser();
  return (
    <main className="page">
      <Nav current="chat" eyebrow="Chat" initial={initialOf(await currentName(user.id))} />
      <Chat />
    </main>
  );
}
