"use client";

import { useState, useTransition } from "react";
import Link from "next/link";
import type { Entry } from "@reflect/core/journal";
import { LocalTime } from "@/components/local-time";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { closeEntryAction } from "./actions";
import { Write } from "./write";

const day = (iso: string) => new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric", year: "numeric" });

/** One entry: its parts in order, and while open, a way to add to it or close it. */
export function EntryCard({ entry }: { entry: Entry }) {
  const [adding, setAdding] = useState(false);
  const [pending, start] = useTransition();
  const open = !entry.closedAt;
  return (
    <Card className="gap-3 py-5">
      <CardHeader className="px-5">
        <CardTitle className="flex items-center gap-2 text-sm">
          {day(entry.localDate)}
          {open ? <Badge variant="secondary">Open</Badge> : null}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 px-5">
        {entry.parts.map((p) => (
          <div key={p.seq} className="space-y-1">
            <p className="text-muted-foreground text-xs"><LocalTime iso={p.createdAt} only="time" /></p>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{p.text}</p>
          </div>
        ))}
        {open ? (
          adding ? (
            <Write entryId={entry.id} placeholder="Add to this entry..." onDone={() => setAdding(false)} />
          ) : (
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => setAdding(true)}>Add to it</Button>
              <Button size="sm" variant="ghost" disabled={pending} onClick={() => start(async () => void (await closeEntryAction(entry.id)))}>Close entry</Button>
              <Button asChild size="sm" variant="ghost"><Link href={`/journal/items/new?entry=${entry.id}`}>Reflect on this</Link></Button>
            </div>
          )
        ) : (
          <Button asChild size="sm" variant="outline"><Link href={`/journal/items/new?entry=${entry.id}`}>Reflect on this</Link></Button>
        )}
      </CardContent>
    </Card>
  );
}
