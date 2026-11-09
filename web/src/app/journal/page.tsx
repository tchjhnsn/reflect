import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { requireUser } from "@/server/session";
import { currentName, hasOnboarded } from "@/server/person-store";
import { listEntries } from "@/server/journal-store";
import { Journal } from "./journal";

export const metadata: Metadata = { title: "Your journal" };
export const dynamic = "force-dynamic";

export default async function JournalPage() {
  const user = await requireUser();
  if (!(await hasOnboarded(user.id))) redirect("/welcome");
  const [name, entries] = await Promise.all([currentName(user.id), listEntries(user.id)]);
  return <Journal name={name} entries={entries} />;
}
