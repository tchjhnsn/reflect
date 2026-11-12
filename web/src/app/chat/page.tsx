import type { Metadata } from "next";
import Link from "next/link";
import { requireUser } from "@/server/session";
import { mono } from "../fonts";
import { Chat } from "./chat";

export const metadata: Metadata = { title: "Ask" };
export const dynamic = "force-dynamic";

/** The plus button's chat (docs/design/REFLECT_UI_2026-11-12.md section 5). It is not saved to the journal yet. */
export default async function ChatPage() {
  await requireUser();
  return (
    <main className={`nd ${mono.className}`}>
      <div className="nd-bar" aria-hidden="true" />
      <header className="nd-head">
        <div className="nd-head-l">
          <Link href="/" className="nd-quiet">&#8592; HOME</Link>
          <h1 className="nd-title">ASK</h1>
        </div>
      </header>
      <Chat />
    </main>
  );
}
