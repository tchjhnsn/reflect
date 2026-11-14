import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { itemListById } from "@reflect/core/item-lists";
import { mediumOf } from "@reflect/core/ways";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PAGE } from "@/components/page-header";
import { requireUser } from "@/server/session";
import { listItems } from "@/server/ways-store";
import { addFromItemListAction } from "../../../actions";
import { LearnHeader } from "../../../header";

export const metadata: Metadata = { title: "List" };
export const dynamic = "force-dynamic";

/** A list beyond books, in its order: each item with its medium, the ones already kept marked and opened. */
export default async function ItemListPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const id = (await params).id;
  const list = itemListById(id);
  if (!list) notFound();
  const kept = new Map((await listItems(user.id).catch(() => [])).map((i) => [`${i.medium}|${i.title.toLowerCase()}`, i.id]));
  const left = list.items.filter((i) => !kept.has(`${i.medium}|${i.title.toLowerCase()}`)).length;
  return (
    <main className={PAGE}>
      <LearnHeader on="browse" />
      <section className="mt-8 space-y-5">
        <div className="space-y-1">
          <p className="text-muted-foreground text-sm"><Link href="/learn/lists" className="hover:underline">Lists</Link></p>
          <h2 className="text-2xl font-semibold tracking-tight">{list.name}</h2>
          <p className="text-muted-foreground text-sm">{list.about}</p>
          <p className="text-muted-foreground text-xs">{list.source}</p>
        </div>
        {left ? (
          <form action={addFromItemListAction}>
            <input type="hidden" name="list" value={list.id} />
            <Button size="sm">Add the {left} not kept to Want to</Button>
          </form>
        ) : <p className="text-sm">You keep every item on this list.</p>}
        <ol className="space-y-2">
          {list.items.map((i, n) => {
            const mine = kept.get(`${i.medium}|${i.title.toLowerCase()}`);
            return (
              <li key={i.title} className="flex flex-wrap items-center gap-3 rounded-md border p-3">
                <span className="text-muted-foreground w-6 text-right text-sm tabular-nums">{n + 1}</span>
                <span className="min-w-0 flex-1">
                  <span className="block text-sm font-medium">{mine ? <Link href={`/learn/items/${mine}`} className="hover:underline">{i.title}</Link> : i.title}</span>
                  <span className="text-muted-foreground block text-xs">{i.by}</span>
                </span>
                <Badge variant="secondary">{mediumOf(i.medium).label}</Badge>
                {mine ? <Button asChild size="sm" variant="outline"><Link href={`/learn/items/${mine}`}>Kept</Link></Button> : (
                  <form action={addFromItemListAction}>
                    <input type="hidden" name="list" value={list.id} />
                    <input type="hidden" name="title" value={i.title} />
                    <Button size="sm" variant="outline">Want to</Button>
                  </form>
                )}
              </li>
            );
          })}
        </ol>
      </section>
    </main>
  );
}
