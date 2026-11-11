import { createHash, randomBytes, randomUUID } from "node:crypto";
import type { PoolClient } from "@neondatabase/serverless";
import { db } from "@/db/pool";
import { open, seal } from "./crypto";

/**
 * A legacy release carried out by the app (Torian, 2026-11-11). The person keeps a plan: trustees who
 * may confirm their death, how many must confirm, the days of waiting in which the person or any
 * trustee can stop it, and each recipient's part. Each trustee and recipient has a link. Nothing is
 * shown to a recipient until enough trustees have confirmed and the wait has passed; before then a
 * recipient's link says nothing, not even that something waits.
 */

export type LegacyItem = { title: string; text: string };
export type PlanInput = {
  confirmations: number;
  waitDays: number;
  trustees: { name: string; contact: string }[];
  gifts: { to: string; contact: string; items: LegacyItem[] }[];
};
export type ReleaseState =
  | { state: "none" }
  | { state: "confirming"; confirmed: number; needed: number }
  | { state: "waiting"; confirmed: number; needed: number; releasesAt: string }
  | { state: "released"; releasedAt: string };

const LIMIT = { name: 200, contact: 320, title: 200, text: 100_000 };

const newToken = () => randomBytes(24).toString("base64url");
const hashOf = (token: string) => createHash("sha256").update(token).digest("hex");

export function planProblems(p: PlanInput): string[] {
  const out: string[] = [];
  if (!Number.isInteger(p.confirmations) || p.confirmations < 1) out.push("confirmations is a whole number, 1 or more");
  if (!Number.isInteger(p.waitDays) || p.waitDays < 0 || p.waitDays > 365) out.push("waitDays is a whole number of days from 0 to 365");
  if (!p.trustees.length) out.push("name at least one trustee who may confirm your death");
  if (p.confirmations > p.trustees.length) out.push(`the plan asks ${p.confirmations} confirmations of ${p.trustees.length} trustee(s)`);
  for (const t of p.trustees) if (!t.name?.trim() || !t.contact?.trim() || t.name.length > LIMIT.name || t.contact.length > LIMIT.contact) out.push("each trustee needs a name and a contact");
  for (const g of p.gifts) {
    if (!g.to?.trim() || !g.contact?.trim() || g.to.length > LIMIT.name || g.contact.length > LIMIT.contact) out.push("each gift needs a person and a contact");
    if (!g.items?.length) out.push(`the gift to ${g.to} has nothing in it`);
    for (const i of g.items ?? []) if (!i.title?.trim() || typeof i.text !== "string" || i.title.length > LIMIT.title || i.text.length > LIMIT.text) out.push(`an item for ${g.to} needs a title and its text`);
  }
  return out;
}

async function tx<T>(f: (c: PoolClient) => Promise<T>): Promise<T> {
  const client = await db().connect();
  try {
    await client.query("BEGIN");
    const r = await f(client);
    await client.query("COMMIT");
    return r;
  } catch (e) {
    await client.query("ROLLBACK");
    throw e;
  } finally {
    client.release();
  }
}

type ReleaseRow = { id: string; started_at: Date | null; wait_days: number; stopped_at: Date | null; released: boolean; releases_at: Date | null };

async function openRelease(c: PoolClient | ReturnType<typeof db>, userId: string): Promise<ReleaseRow | null> {
  const r = await c.query(
    `SELECT id, started_at, wait_days, stopped_at,
            (started_at IS NOT NULL AND now() >= started_at + make_interval(days => wait_days)) AS released,
            started_at + make_interval(days => wait_days) AS releases_at
       FROM legacy_release WHERE user_id = $1 AND stopped_at IS NULL`,
    [userId],
  );
  return (r.rows[0] as ReleaseRow | undefined) ?? null;
}

async function stateOf(c: PoolClient | ReturnType<typeof db>, userId: string): Promise<ReleaseState> {
  const rel = await openRelease(c, userId);
  if (!rel) return { state: "none" };
  if (rel.released) return { state: "released", releasedAt: rel.releases_at!.toISOString() };
  const plan = (await c.query("SELECT confirmations FROM legacy_plan WHERE user_id = $1", [userId])).rows[0] as { confirmations: number } | undefined;
  const confirmed = Number(((await c.query("SELECT count(*) AS n FROM legacy_confirmation WHERE release_id = $1", [rel.id])).rows[0] as { n: string }).n);
  const needed = plan?.confirmations ?? 1;
  if (rel.started_at) return { state: "waiting", confirmed, needed, releasesAt: rel.releases_at!.toISOString() };
  return { state: "confirming", confirmed, needed };
}

/** Keep a new plan. Old trustees and gifts are revoked, never deleted. Refused while a release is open. */
export async function keepPlan(userId: string, p: PlanInput): Promise<{ trustees: { name: string; token: string }[]; gifts: number }> {
  const problems = planProblems(p);
  if (problems.length) throw new Error(problems.join("; "));
  return tx(async (c) => {
    if (await openRelease(c, userId)) throw new Error("a release is under way; stop it before changing the plan");
    await c.query(
      `INSERT INTO legacy_plan (user_id, confirmations, wait_days) VALUES ($1, $2, $3)
       ON CONFLICT (user_id) DO UPDATE SET confirmations = $2, wait_days = $3, updated_at = now()`,
      [userId, p.confirmations, p.waitDays],
    );
    await c.query("UPDATE legacy_trustee SET revoked_at = now() WHERE user_id = $1 AND revoked_at IS NULL", [userId]);
    await c.query("UPDATE legacy_gift SET revoked_at = now() WHERE user_id = $1 AND revoked_at IS NULL", [userId]);
    const trustees = [];
    for (const t of p.trustees) {
      const token = newToken();
      await c.query("INSERT INTO legacy_trustee (id, user_id, name_ct, contact_ct, token_hash, token_ct) VALUES ($1, $2, $3, $4, $5, $6)", [
        randomUUID(), userId, seal(userId, t.name.trim()), seal(userId, t.contact.trim()), hashOf(token), seal(userId, token),
      ]);
      trustees.push({ name: t.name.trim(), token });
    }
    for (const g of p.gifts) {
      const token = newToken();
      await c.query("INSERT INTO legacy_gift (id, user_id, recipient_ct, contact_ct, items_ct, token_hash, token_ct) VALUES ($1, $2, $3, $4, $5, $6, $7)", [
        randomUUID(), userId, seal(userId, g.to.trim()), seal(userId, g.contact.trim()), seal(userId, JSON.stringify(g.items)), hashOf(token), seal(userId, token),
      ]);
    }
    return { trustees, gifts: p.gifts.length };
  });
}

type TrusteeRow = { id: string; user_id: string; name_ct: string; contact_ct: string; token_ct: string };
type GiftRow = { user_id: string; recipient_ct: string; contact_ct: string; items_ct: string; token_ct: string };

async function personName(userId: string): Promise<string> {
  return ((await db().query(`SELECT name FROM "user" WHERE id = $1`, [userId])).rows[0] as { name: string } | undefined)?.name || "Someone";
}

async function giftsOf(userId: string): Promise<{ to: string; contact: string; token: string }[]> {
  const rows = (await db().query("SELECT user_id, recipient_ct, contact_ct, items_ct, token_ct FROM legacy_gift WHERE user_id = $1 AND revoked_at IS NULL ORDER BY created_at", [userId])).rows as GiftRow[];
  return rows.map((g) => ({ to: open(userId, g.recipient_ct), contact: open(userId, g.contact_ct), token: open(userId, g.token_ct) }));
}

/** The person's own view: the plan, each trustee's link, who receives something, and any release. */
export async function legacyStatus(userId: string) {
  const plan = (await db().query("SELECT confirmations, wait_days FROM legacy_plan WHERE user_id = $1", [userId])).rows[0] as { confirmations: number; wait_days: number } | undefined;
  const trustees = ((await db().query("SELECT id, user_id, name_ct, contact_ct, token_ct FROM legacy_trustee WHERE user_id = $1 AND revoked_at IS NULL ORDER BY created_at", [userId])).rows as TrusteeRow[]).map(
    (t) => ({ name: open(userId, t.name_ct), contact: open(userId, t.contact_ct), token: open(userId, t.token_ct) }),
  );
  const gifts = (await giftsOf(userId)).map(({ to, contact }) => ({ to, contact }));
  return { plan: plan ? { confirmations: plan.confirmations, waitDays: plan.wait_days } : null, trustees, gifts, release: await stateOf(db(), userId) };
}

async function trusteeByToken(token: string): Promise<TrusteeRow | null> {
  if (!token || token.length > 100) return null;
  return ((await db().query("SELECT id, user_id, name_ct, contact_ct, token_ct FROM legacy_trustee WHERE token_hash = $1 AND revoked_at IS NULL", [hashOf(token)])).rows[0] as TrusteeRow | undefined) ?? null;
}

/** What a trustee sees at their link. Recipients' links appear only once the release has happened. */
export async function trusteeView(token: string) {
  const t = await trusteeByToken(token);
  if (!t) return null;
  const release = await stateOf(db(), t.user_id);
  const rel = await openRelease(db(), t.user_id);
  const mine = rel ? (await db().query("SELECT 1 FROM legacy_confirmation WHERE release_id = $1 AND trustee_id = $2", [rel.id, t.id])).rowCount === 1 : false;
  return {
    person: await personName(t.user_id),
    trustee: open(t.user_id, t.name_ct),
    release,
    confirmedByYou: mine,
    recipients: release.state === "released" ? await giftsOf(t.user_id) : [],
  };
}

/** A trustee confirms the person's death. The release starts once enough trustees have. */
export async function confirmDeath(token: string): Promise<ReleaseState> {
  const t = await trusteeByToken(token);
  if (!t) throw new Error("this link is not valid");
  return tx(async (c) => {
    await c.query("SELECT pg_advisory_xact_lock(hashtext($1))", [t.user_id]);
    let rel = await openRelease(c, t.user_id);
    const plan = (await c.query("SELECT confirmations, wait_days FROM legacy_plan WHERE user_id = $1", [t.user_id])).rows[0] as { confirmations: number; wait_days: number };
    if (!rel) {
      await c.query("INSERT INTO legacy_release (id, user_id, wait_days) VALUES ($1, $2, $3)", [randomUUID(), t.user_id, plan.wait_days]);
      rel = (await openRelease(c, t.user_id))!;
    }
    await c.query("INSERT INTO legacy_confirmation (release_id, trustee_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", [rel.id, t.id]);
    const n = Number(((await c.query("SELECT count(*) AS n FROM legacy_confirmation WHERE release_id = $1", [rel.id])).rows[0] as { n: string }).n);
    if (!rel.started_at && n >= plan.confirmations) await c.query("UPDATE legacy_release SET started_at = now() WHERE id = $1", [rel.id]);
    return stateOf(c, t.user_id);
  });
}

/** Stop a release before it happens: the person, or any trustee. A release that has happened cannot be stopped. */
export async function stopRelease(by: { userId: string } | { trusteeToken: string }): Promise<void> {
  let userId: string;
  let who: string;
  if ("userId" in by) {
    userId = by.userId;
    who = "person";
  } else {
    const t = await trusteeByToken(by.trusteeToken);
    if (!t) throw new Error("this link is not valid");
    userId = t.user_id;
    who = `trustee:${t.id}`;
  }
  await tx(async (c) => {
    const rel = await openRelease(c, userId);
    if (!rel) return;
    if (rel.released) throw new Error("the release has already happened");
    await c.query("UPDATE legacy_release SET stopped_at = now(), stopped_by = $2 WHERE id = $1", [rel.id, who]);
  });
}

/** What a recipient sees at their link: their part once released, otherwise nothing at all. */
export async function receive(token: string): Promise<{ from: string; items: LegacyItem[] } | null> {
  if (!token || token.length > 100) return null;
  const g = (await db().query("SELECT user_id, recipient_ct, contact_ct, items_ct, token_ct FROM legacy_gift WHERE token_hash = $1 AND revoked_at IS NULL", [hashOf(token)])).rows[0] as GiftRow | undefined;
  if (!g) return null;
  if ((await stateOf(db(), g.user_id)).state !== "released") return null;
  return { from: await personName(g.user_id), items: JSON.parse(open(g.user_id, g.items_ct)) as LegacyItem[] };
}
