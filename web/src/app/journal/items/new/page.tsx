import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeftIcon } from "lucide-react";
import type { ItemSource } from "@reflect/core/items";
import type { Act } from "@reflect/core/log";
import { actText } from "@reflect/core/log";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { requireUser } from "@/server/session";
import { entryText } from "@/server/journal-store";
import { Observe } from "./observe";

export const metadata: Metadata = { title: "Observe" };
export const dynamic = "force-dynamic";

function readAct(raw: string | undefined): Act | null {
  if (!raw) return null;
  try {
    const a = JSON.parse(raw) as Act;
    return typeof a.at === "string" && a.app === "learn" && typeof a.kind === "string" && typeof a.object === "string" ? a : null;
  } catch {
    return null;
  }
}

/** Observe: state what is, from an entry, an act in the log, or from nothing. */
export default async function NewItemPage({ searchParams }: { searchParams: Promise<{ entry?: string; act?: string }> }) {
  const user = await requireUser();
  const q = await searchParams;
  const act = readAct(q.act);
  const entry = q.entry ? await entryText(user.id, q.entry) : null;
  const from: ItemSource | null = act ? { kind: "act", act } : entry && q.entry ? { kind: "entry", id: q.entry } : null;
  return (
    <main className="mx-auto max-w-2xl px-6 py-6">
      <header className="flex items-center gap-3">
        <Button asChild variant="ghost" size="icon" aria-label="Journal"><Link href="/journal"><ArrowLeftIcon /></Link></Button>
        <h1 className="text-2xl font-semibold tracking-tight">Observe</h1>
      </header>
      <Card className="mt-6 py-5">
        <CardHeader className="px-5">
          <CardTitle className="text-base">What do you observe?</CardTitle>
          <CardDescription>State what is: something you believe, something you noticed, something you mean to do. Reflect, Examine and Plan come after.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 px-5">
          {act ? <blockquote className="border-l-2 pl-3 text-sm">{actText(act)} <span className="text-muted-foreground">(from your log)</span></blockquote> : null}
          {entry ? <blockquote className="text-muted-foreground line-clamp-4 border-l-2 pl-3 text-sm whitespace-pre-wrap">{entry}</blockquote> : null}
          <Observe from={from} />
        </CardContent>
      </Card>
    </main>
  );
}
