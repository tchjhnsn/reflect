import LifeRecordPage from "../../[app]/[id]/page";

import type { Metadata } from "next";

export const metadata: Metadata = { title: "Work" };
export const dynamic = "force-dynamic";

/** Work's own records (what you imagined, what inspires you), opened like any record of a life app. */
export default async function WorkRecordPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return LifeRecordPage({ params: Promise.resolve({ app: "work", id }) });
}
