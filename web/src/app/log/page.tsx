import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeftIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { requireUser } from "@/server/session";
import { logActs } from "@/server/log-store";
import { LogDays } from "./days";

export const metadata: Metadata = { title: "Log" };
export const dynamic = "force-dynamic";

/** The log: what you did in the tiles, kept by the apps, never written by you. Learn only so far. */
export default async function LogPage() {
  const user = await requireUser();
  const acts = await logActs(user.id);
  return (
    <main className="mx-auto max-w-3xl px-6 py-6">
      <header className="flex items-center gap-3">
        <Button asChild variant="ghost" size="icon" aria-label="Home"><Link href="/"><ArrowLeftIcon /></Link></Button>
        <h1 className="text-2xl font-semibold tracking-tight">Log</h1>
      </header>
      <p className="text-muted-foreground mt-2 text-sm">What you did, kept by your apps as you go. You never write here; reflect on any of it in your journal.</p>
      <div className="mt-6">
        {acts.length ? <LogDays acts={acts} /> : <p className="text-muted-foreground mt-10 text-center text-sm">Nothing yet. Start a book or keep a claim in Learn and it shows here.</p>}
      </div>
    </main>
  );
}
