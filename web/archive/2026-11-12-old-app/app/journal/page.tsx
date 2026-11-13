import type { Metadata } from "next";
import { requireUser } from "@/server/session";
import { currentName } from "@/server/person-store";
import { listEntries } from "@/server/journal-store";
import { chosenPacks, usedPrompts } from "@/server/me-store";
import { Journal } from "./journal";

export const metadata: Metadata = { title: "Your journal" };
export const dynamic = "force-dynamic";

export default async function JournalPage() {
  const user = await requireUser();
  const [name, entries, packs, used] = await Promise.all([currentName(user.id), listEntries(user.id), chosenPacks(user.id), usedPrompts(user.id)]);
  return <Journal name={name} entries={entries} packs={packs} used={used} />;
}
