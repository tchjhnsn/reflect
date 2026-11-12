import Link from "next/link";
import { actText } from "@reflect/core/log";
import type { ItemSource } from "@reflect/core/items";

/** Where an item was stated from, as one quiet line. */
export function SourceLine({ from, entry }: { from: ItemSource | null; entry?: string | null }) {
  if (!from) return null;
  if (from.kind === "act") return <p className="text-muted-foreground text-xs">From your log: <Link href="/log" className="underline underline-offset-2">{actText(from.act)}</Link></p>;
  return <p className="text-muted-foreground line-clamp-2 text-xs">From your journal{entry ? <>: &ldquo;{entry}&rdquo;</> : null}</p>;
}
