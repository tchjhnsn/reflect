"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { BookOpenIcon, BookmarkIcon, CheckIcon, CircleSlashIcon, MessageSquareQuoteIcon, PlusIcon, RefreshCwIcon, SparklesIcon } from "lucide-react";
import { actText, logByDay, type Act, type ActKind } from "@reflect/core/log";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const ICON: Record<ActKind, typeof PlusIcon> = { intended: PlusIcon, started: BookOpenIcon, finished: CheckIcon, put_down: CircleSlashIcon, cited: BookmarkIcon, kept_claim: MessageSquareQuoteIcon, changed_stance: RefreshCwIcon, rated: SparklesIcon };
const dayName = (d: string) => new Date(`${d}T12:00:00`).toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric", year: "numeric" });
const time = (a: Act) => (a.at.length === 10 ? "" : new Date(a.at).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }));

/** The acts grouped by the person's own day (the browser's time zone). Intentions are quieter than acts. */
export function LogDays({ acts }: { acts: Act[] }) {
  // Days and times are the browser's, so they are drawn after mounting: the server does not know the time zone.
  const [offset, setOffset] = useState<number | null>(null);
  useEffect(() => setOffset(-new Date().getTimezoneOffset()), []);
  if (offset === null) return null;
  const days = logByDay(acts, offset);
  return (
    <div className="space-y-4">
      {days.map((d) => (
        <Card key={d.day} className="gap-3 py-5">
          <CardHeader className="px-5"><CardTitle className="text-sm">{dayName(d.day)}</CardTitle></CardHeader>
          <CardContent className="px-5">
            <ul className="space-y-2.5">
              {d.acts.map((a, i) => {
                const Icon = ICON[a.kind];
                return (
                  <li key={i} className={cn("flex items-start gap-3 text-sm", a.kind === "intended" && "text-muted-foreground")}>
                    <Icon className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
                    <span className="flex-1">{actText(a)}</span>
                    <span className="text-muted-foreground shrink-0 text-xs">{time(a) || "Learn"}</span>
                    <Link href={`/journal/items/new?act=${encodeURIComponent(JSON.stringify(a))}`} className="text-muted-foreground hover:text-foreground shrink-0 text-xs underline-offset-2 hover:underline">Reflect</Link>
                  </li>
                );
              })}
            </ul>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
