"use server";

import { revalidatePath } from "next/cache";
import { readMove, type ItemSource } from "@reflect/core/items";
import type { Act } from "@reflect/core/log";
import { requireUser } from "@/server/session";
import { entryText } from "@/server/journal-store";
import { logActs } from "@/server/log-store";
import { addMove } from "@/server/item-store";

export type MoveResult = { error: string } | { ok: true; id: string };

const sameAct = (a: Act, b: Act) => a.at === b.at && a.app === b.app && a.kind === b.kind && a.object === b.object && (a.detail ?? null) === (b.detail ?? null);

/** Whether the source is the person's own: an entry they wrote, or an act their records show. */
async function checkSource(userId: string, from: ItemSource | null): Promise<boolean> {
  if (!from) return true;
  if (from.kind === "entry") return (await entryText(userId, from.id)) !== null;
  return (await logActs(userId)).some((a) => sameAct(a, from.act));
}

/** A new item: what the person observes, stated from an entry, an act in the log, or nothing. */
export async function stateItem(text: string, from: ItemSource | null): Promise<MoveResult> {
  const user = await requireUser();
  if (!(await checkSource(user.id, from))) return { error: "That is not in your journal or your log." };
  try {
    const id = await addMove(user.id, null, { kind: "stated", text, ...(from ? { from } : {}) });
    revalidatePath("/journal");
    return { ok: true, id };
  } catch (e) {
    return { error: (e as Error).message };
  }
}

/** One move on an item. The statechart decides; a refusal comes back with its reason. */
export async function moveItem(itemId: string, move: unknown): Promise<MoveResult> {
  const user = await requireUser();
  const m = readMove(move);
  if (!m || m.kind === "stated") return { error: "That is not a move." };
  try {
    await addMove(user.id, itemId, m);
  } catch (e) {
    return { error: (e as Error).message };
  }
  revalidatePath("/journal");
  revalidatePath(`/journal/items/${itemId}`);
  return { ok: true, id: itemId };
}
