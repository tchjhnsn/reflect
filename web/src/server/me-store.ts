import type { PoolClient } from "@neondatabase/serverless";
import { db } from "@/db/pool";
import type { IsoDate } from "@reflect/core/person";
import { isPackId } from "@reflect/core/packs";
import { open, seal } from "./crypto";

export const FACT_KEYS = ["name", "date_of_birth", "occupation", "marital_status", "seeing_someone", "children"] as const;
export type FactKey = (typeof FACT_KEYS)[number];
export const isFactKey = (k: string): k is FactKey => (FACT_KEYS as readonly string[]).includes(k);

/** One stated value and the dates it held. */
export type Dated = { value: string; validFrom: IsoDate; validTo: IsoDate | null };
export type Belief = { id: string; kind: "belief" | "value"; text: string; validFrom: IsoDate; validTo: IsoDate | null };

export const HOLD_SUBJECTS = ["partnership", "children"] as const;
export type HoldSubject = (typeof HOLD_SUBJECTS)[number];
export const isHoldSubject = (k: string): k is HoldSubject => (HOLD_SUBJECTS as readonly string[]).includes(k);

export type Me = {
  facts: Record<FactKey, Dated[]>;
  holds: Record<HoldSubject, Dated[]>;
  closestFriend: Dated[];
  beliefs: Belief[];
  packs: string[];
};

const day = (v: Date | string | null): IsoDate | null =>
  v === null ? null : ((typeof v === "string" ? v : v.toISOString()).slice(0, 10) as IsoDate);

async function tx<T>(fn: (c: PoolClient) => Promise<T>): Promise<T> {
  const client = await db().connect();
  try {
    await client.query("BEGIN");
    const out = await fn(client);
    await client.query("COMMIT");
    return out;
  } catch (e) {
    await client.query("ROLLBACK");
    throw e;
  } finally {
    client.release();
  }
}

export async function chosenPacks(userId: string): Promise<string[]> {
  const r = await db().query(
    "SELECT DISTINCT ON (pack_id) pack_id, chosen FROM pack_choice WHERE user_id = $1 ORDER BY pack_id, id DESC",
    [userId],
  );
  return (r.rows as { pack_id: string; chosen: boolean }[]).filter((x) => x.chosen && isPackId(x.pack_id)).map((x) => x.pack_id);
}

export async function usedPrompts(userId: string): Promise<string[]> {
  const r = await db().query("SELECT DISTINCT prompt_id FROM journal_entry WHERE user_id = $1 AND prompt_id IS NOT NULL", [userId]);
  return (r.rows as { prompt_id: string }[]).map((x) => x.prompt_id);
}

export async function loadMe(userId: string): Promise<Me> {
  const [facts, rels, beliefs, packs, holds] = await Promise.all([
    db().query("SELECT key, value_ct, valid_from::text AS f, valid_to::text AS t FROM person_fact WHERE user_id = $1 ORDER BY valid_from DESC, id DESC", [userId]),
    db().query("SELECT name_ct, valid_from::text AS f, valid_to::text AS t FROM person_relationship WHERE user_id = $1 AND kind = 'closest_friend' ORDER BY valid_from DESC, id DESC", [userId]),
    db().query("SELECT id::text, kind, text_ct, valid_from::text AS f, valid_to::text AS t FROM person_belief WHERE user_id = $1 ORDER BY id DESC", [userId]),
    chosenPacks(userId),
    db().query("SELECT subject, value_ct, valid_from::text AS f, valid_to::text AS t FROM person_hold WHERE user_id = $1 ORDER BY valid_from DESC, id DESC", [userId]),
  ]);
  const out: Me = {
    facts: { name: [], date_of_birth: [], occupation: [], marital_status: [], seeing_someone: [], children: [] },
    holds: { partnership: [], children: [] },
    closestFriend: [],
    beliefs: [],
    packs,
  };
  for (const r of holds.rows as { subject: string; value_ct: string; f: string; t: string | null }[]) {
    if (isHoldSubject(r.subject)) out.holds[r.subject].push({ value: open(userId, r.value_ct), validFrom: day(r.f)!, validTo: day(r.t) });
  }
  for (const r of facts.rows as { key: string; value_ct: string; f: string; t: string | null }[]) {
    if (isFactKey(r.key)) out.facts[r.key].push({ value: open(userId, r.value_ct), validFrom: day(r.f)!, validTo: day(r.t) });
  }
  for (const r of rels.rows as { name_ct: string; f: string; t: string | null }[]) {
    out.closestFriend.push({ value: open(userId, r.name_ct), validFrom: day(r.f)!, validTo: day(r.t) });
  }
  for (const r of beliefs.rows as { id: string; kind: "belief" | "value"; text_ct: string; f: string; t: string | null }[]) {
    out.beliefs.push({ id: r.id, kind: r.kind, text: open(userId, r.text_ct), validFrom: day(r.f)!, validTo: day(r.t) });
  }
  return out;
}

/** A new value closes the open one; nothing is overwritten. An empty value only closes it. */
export async function stateFactNow(userId: string, key: FactKey, value: string | null, at: IsoDate): Promise<void> {
  await tx(async (c) => {
    await c.query("UPDATE person_fact SET valid_to = $3 WHERE user_id = $1 AND key = $2 AND valid_to IS NULL", [userId, key, at]);
    if (value !== null) {
      await c.query("INSERT INTO person_fact (user_id, key, value_ct, valid_from, source) VALUES ($1, $2, $3, $4, 'stated')", [userId, key, seal(userId, value), at]);
    }
  });
}

export async function stateFriendNow(userId: string, name: string | null, at: IsoDate): Promise<void> {
  await tx(async (c) => {
    await c.query("UPDATE person_relationship SET valid_to = $2 WHERE user_id = $1 AND kind = 'closest_friend' AND valid_to IS NULL", [userId, at]);
    if (name !== null) {
      await c.query("INSERT INTO person_relationship (user_id, kind, name_ct, valid_from, source) VALUES ($1, 'closest_friend', $2, $3, 'stated')", [userId, seal(userId, name), at]);
    }
  });
}

export async function addBelief(userId: string, kind: "belief" | "value", text: string, at: IsoDate): Promise<void> {
  await db().query("INSERT INTO person_belief (user_id, kind, text_ct, valid_from, source) VALUES ($1, $2, $3, $4, 'stated')", [userId, kind, seal(userId, text), at]);
}

export async function letGo(userId: string, beliefId: string, at: IsoDate): Promise<void> {
  if (!/^\d+$/.test(beliefId)) return;
  await db().query("UPDATE person_belief SET valid_to = $3 WHERE user_id = $1 AND id = $2 AND valid_to IS NULL", [userId, beliefId, at]);
}

export async function choosePack(userId: string, packId: string, chosen: boolean): Promise<void> {
  if (!isPackId(packId)) throw new Error("no such pack");
  await db().query("INSERT INTO pack_choice (user_id, pack_id, chosen) VALUES ($1, $2, $3)", [userId, packId, chosen]);
}

export async function stateHoldNow(userId: string, subject: HoldSubject, value: string | null, at: IsoDate): Promise<void> {
  await tx(async (c) => {
    await c.query("UPDATE person_hold SET valid_to = $3 WHERE user_id = $1 AND subject = $2 AND valid_to IS NULL", [userId, subject, at]);
    if (value !== null) {
      await c.query("INSERT INTO person_hold (user_id, subject, value_ct, valid_from, source) VALUES ($1, $2, $3, $4, 'stated')", [userId, subject, seal(userId, value), at]);
    }
  });
}
