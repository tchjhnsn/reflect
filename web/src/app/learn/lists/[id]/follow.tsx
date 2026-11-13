"use client";

import { useState, useTransition } from "react";
import { ListOrderedIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { followListAction } from "../../actions";

/** Put a list's books on Want to read in the list's order, after the books already there. */
export function FollowList({ id, missing }: { id: string; /** How many of its books are not on the person's shelves. */ missing: number }) {
  const [note, setNote] = useState<string | null>(null);
  const [pending, run] = useTransition();
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button variant="outline" size="sm" disabled={pending || missing === 0} onClick={() => run(async () => {
        const n = await followListAction(id);
        setNote(`Added ${n} ${n === 1 ? "book" : "books"} to Want to read, in this order.`);
      })}><ListOrderedIcon />{pending ? "Adding..." : missing === 0 ? "All on your shelves" : "Read in this order"}</Button>
      {note ? <span className="text-muted-foreground text-xs" role="status">{note}</span> : null}
    </div>
  );
}
