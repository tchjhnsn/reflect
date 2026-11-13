import type { Metadata } from "next";
import { requireUser } from "@/server/session";
import { Chat } from "./chat";

export const metadata: Metadata = { title: "Ask" };
export const dynamic = "force-dynamic";

/** The plus button's chat (docs/design/REFLECT_UI_2026-11-12.md section 5), in shadcn's look with AI Elements. It is not saved to the journal yet. */
export default async function ChatPage() {
  await requireUser();
  return <Chat />;
}
