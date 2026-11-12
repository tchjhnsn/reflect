import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { requireUser } from "@/server/session";
import { getWork } from "@/server/work-store";
import { Editor } from "./editor";

export const metadata: Metadata = { title: "Journal" };
export const dynamic = "force-dynamic";

/** One work, open to write in. */
export default async function WorkPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const { id } = await params;
  const work = await getWork(user.id, id);
  if (!work) notFound();
  return <Editor work={work} />;
}
