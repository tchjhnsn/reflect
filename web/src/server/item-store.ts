// Items, kept as events (migration 0014; docs/design/REFLECT_SYSTEM_2026-11-12.md section 3). Every move is
// checked by the core's statechart against the item's events before it is added; the item's state is always
// the core's fold, never stored. Events are sealed with the person's key and hash-chained per item.
import { createHash, randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { apply, fold, type Item, type ItemEvent, type ItemMove } from "@reflect/core/items";
import { open, seal } from "./crypto";

type Row = { item_id: string; seq: number; kind: string; body_ct: string; hash: string };

const chain = (prev: string, itemId: string, seq: number, kind: string, bodyCt: string) =>
  createHash("sha256").update(`${prev}\n${itemId}\n${seq}\n${kind}\n${bodyCt}`).digest("hex");

const eventOf = (userId: string, r: Row): ItemEvent => ({ ...(JSON.parse(open(userId, r.body_ct)) as object), kind: r.kind } as ItemEvent);

function itemsOf(userId: string, rows: Row[]): Map<string, ItemEvent[]> {
  const byItem = new Map<string, ItemEvent[]>();
  for (const r of rows) byItem.set(r.item_id, [...(byItem.get(r.item_id) ?? []), eventOf(userId, r)]);
  return byItem;
}

/** Every item the person has, newest change first. An item whose events no longer fold is left out, not shown wrong. */
export async function listItems(userId: string): Promise<{ id: string; item: Item }[]> {
  const rows = (await db().query("SELECT item_id, seq, kind, body_ct, hash FROM item_event WHERE user_id = $1 ORDER BY item_id, seq", [userId])).rows as Row[];
  const out: { id: string; item: Item }[] = [];
  for (const [id, events] of itemsOf(userId, rows)) {
    try {
      const item = fold(events);
      if (item) out.push({ id, item });
    } catch { /* a broken history is not shown as if it were whole */ }
  }
  return out.sort((a, b) => (b.item.events.at(-1)!.at < a.item.events.at(-1)!.at ? -1 : 1));
}

export async function getItem(userId: string, itemId: string): Promise<Item | null> {
  if (!/^[0-9a-f-]{36}$/i.test(itemId)) return null;
  const rows = (await db().query("SELECT item_id, seq, kind, body_ct, hash FROM item_event WHERE user_id = $1 AND item_id = $2 ORDER BY seq", [userId, itemId])).rows as Row[];
  return fold(rows.map((r) => eventOf(userId, r)));
}

/**
 * Add one move to an item, or make a new item when `itemId` is null (the move must then be `stated`). The
 * statechart decides; a refused move throws with its reason. Two moves racing for the same place both read
 * the same history, and the second insert hits the primary key and is refused.
 */
export async function addMove(userId: string, itemId: string | null, move: ItemMove): Promise<string> {
  const id = itemId ?? randomUUID();
  const rows = itemId ? ((await db().query("SELECT item_id, seq, kind, body_ct, hash FROM item_event WHERE user_id = $1 AND item_id = $2 ORDER BY seq", [userId, id])).rows as Row[]) : [];
  if (itemId && !rows.length) throw new Error("No such item.");
  const event = { ...move, at: new Date().toISOString() } as ItemEvent;
  apply(fold(rows.map((r) => eventOf(userId, r))), event);
  const { kind, ...body } = event;
  const bodyCt = seal(userId, JSON.stringify(body));
  const seq = rows.length;
  const hash = chain(rows.at(-1)?.hash ?? "", id, seq, kind, bodyCt);
  await db().query("INSERT INTO item_event (item_id, seq, user_id, kind, body_ct, hash) VALUES ($1, $2, $3, $4, $5, $6)", [id, seq, userId, kind, bodyCt, hash]);
  return id;
}

/** Whether an item's events still chain: each hash follows from the one before. */
export async function chainHolds(userId: string, itemId: string): Promise<boolean> {
  const rows = (await db().query("SELECT item_id, seq, kind, body_ct, hash FROM item_event WHERE user_id = $1 AND item_id = $2 ORDER BY seq", [userId, itemId])).rows as Row[];
  return rows.every((r, i) => r.seq === i && r.hash === chain(rows[i - 1]?.hash ?? "", r.item_id, r.seq, r.kind, r.body_ct));
}
