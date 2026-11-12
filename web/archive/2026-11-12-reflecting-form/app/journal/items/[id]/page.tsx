import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeftIcon } from "lucide-react";
import { eventText, PHASES, phaseOf, STATE_LABEL } from "@reflect/core/items";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LocalTime } from "@/components/local-time";
import { cn } from "@/lib/utils";
import { requireUser } from "@/server/session";
import { getItem } from "@/server/item-store";
import { entryText } from "@/server/journal-store";
import { SourceLine } from "../source";
import { Moves } from "./moves";

export const metadata: Metadata = { title: "Reflecting" };
export const dynamic = "force-dynamic";

/** One item: where it sits in Observe, Reflect, Examine, Plan, Act, what can come next, and its whole history. */
export default async function ItemPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const { id } = await params;
  const item = await getItem(user.id, id).catch(() => null);
  if (!item) notFound();
  const phase = phaseOf(item.state);
  const at = phase ? PHASES.indexOf(phase) : -1;
  const entry = item.from?.kind === "entry" ? await entryText(user.id, item.from.id) : null;
  return (
    <main className="mx-auto max-w-2xl px-6 py-6">
      <header className="flex items-center gap-3">
        <Button asChild variant="ghost" size="icon" aria-label="Journal"><Link href="/journal"><ArrowLeftIcon /></Link></Button>
        <h1 className="text-2xl font-semibold tracking-tight">Reflecting</h1>
      </header>
      <ol className="mt-6 grid grid-cols-5 gap-1.5" aria-label="Where it is">
        {PHASES.map((p, i) => (
          <li key={p} className="space-y-1.5">
            <div className={cn("h-1 rounded-full", i < at ? "bg-foreground/60" : i === at ? "bg-foreground" : "bg-muted")} />
            <p className={cn("text-xs", i === at ? "font-medium" : "text-muted-foreground")}>{p}</p>
          </li>
        ))}
      </ol>
      <Card className="mt-5 gap-3 py-5">
        <CardHeader className="px-5">
          <div className="flex items-start justify-between gap-3">
            <CardTitle className="text-lg leading-snug">{item.statement}</CardTitle>
            <Badge variant={item.state === "dropped" ? "outline" : "secondary"} className="shrink-0">{STATE_LABEL[item.state]}</Badge>
          </div>
          <SourceLine from={item.from} entry={entry} />
          {item.plan ? <p className="text-sm"><span className="text-muted-foreground">Plan: </span>{item.plan}{item.scheduledOn ? <span className="text-muted-foreground"> · {item.scheduledOn}</span> : null}</p> : null}
        </CardHeader>
        <CardContent className="px-5 pt-2">
          <Moves id={id} item={item} />
        </CardContent>
      </Card>
      <section className="mt-6">
        <h2 className="text-muted-foreground mb-2 text-xs font-medium tracking-wide uppercase">History</h2>
        <ol className="space-y-2 border-l pl-4">
          {[...item.events].reverse().map((e, i) => (
            <li key={i} className="text-sm">
              <span className="text-muted-foreground mr-2 text-xs"><LocalTime iso={e.at} /></span>
              {eventText(e)}
            </li>
          ))}
        </ol>
      </section>
    </main>
  );
}
