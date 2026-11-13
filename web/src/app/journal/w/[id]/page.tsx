import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { requireUser } from "@/server/session";
import { getWork } from "@/server/work-store";
import { idealsOf } from "@/server/book-store";
import { subjectNames } from "@/server/subject-domain-store";
import { Editor } from "./editor";

export const metadata: Metadata = { title: "Journal" };
export const dynamic = "force-dynamic";

/** One work, open to write in. */
export default async function WorkPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const { id } = await params;
  const [work, ideals, subjects] = await Promise.all([getWork(user.id, id), idealsOf(user.id).catch(() => []), subjectNames(user.id).catch(() => [])]);
  if (!work) notFound();
  return <Editor work={work} ideals={ideals} subjects={subjects} />;
}
