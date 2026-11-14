import type { Metadata } from "next";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { logActs } from "@/server/log-store";
import { JournalHeader } from "../header";
import { LogDays } from "./days";

export const metadata: Metadata = { title: "Log" };
export const dynamic = "force-dynamic";

/** The log, a lens in the journal (Torian, 2026-11-14 17:54Z): what you did in the tiles, kept by the apps, never written by you. */
export default async function LogPage() {
  const user = await requireUser();
  const acts = await logActs(user.id);
  return (
    <main className={PAGE}>
      <JournalHeader on="log" />
      <div className="max-w-3xl">
        <p className="text-muted-foreground mt-2 text-sm">What you did, kept by your apps as you go. You never write here; reflect on any of it in your journal.</p>
        <div className="mt-6">
          {acts.length ? <LogDays acts={acts} /> : <p className="text-muted-foreground mt-10 text-center text-sm">Nothing yet. Start a book in Learn, or keep something in any of your apps, and it shows here.</p>}
        </div>
      </div>
    </main>
  );
}
