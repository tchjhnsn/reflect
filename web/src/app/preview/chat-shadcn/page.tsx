import type { Metadata } from "next";
import { requireUser } from "@/server/session";
import { ShadcnChat } from "./shadcn-chat";

export const metadata: Metadata = { title: "Ask (shadcn preview)" };
export const dynamic = "force-dynamic";

export default async function ShadcnChatPage() {
  await requireUser();
  return <ShadcnChat />;
}
