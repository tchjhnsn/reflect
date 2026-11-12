import { randomUUID } from "node:crypto";
import { cleanClaim, currentStance, isStance, STANCES, type Holder, type Stance, type StanceLine } from "@reflect/core/claims";
import { db } from "@/db/pool";
import { open, seal, tag } from "./crypto";
import { subjectId, tx } from "./learn-store";

/** Claims and the person's stance on each (migration 0012). Stances are new rows; the latest counts. */

export type ClaimOut = { id: string; text: string; holders: Holder[]; subject: { id: string; name: string } | null; stance: StanceLine | null; history: StanceLine[] };

/** Keep a claim, or find it again by its words, and record the person's stance on it. */
export async function keepClaim(userId: string, c: { text: string; holders?: Holder[]; subject?: string; stance: string; note?: string }): Promise<string> {
  if (!isStance(c.stance)) throw new Error(`a stance is one of ${STANCES.map((s) => s.id).join(", ")}`);
  const { text, holders } = cleanClaim(c.text, c.holders ?? []);
  const note = c.note?.trim();
  if (note && note.length > 5000) throw new Error("the note is too long");
  return tx(async (q) => {
    const sid = c.subject?.trim() ? await subjectId(q, userId, c.subject) : null;
    const t = tag(userId, text.toLowerCase());
    const id = randomUUID();
    await q.query("INSERT INTO learn_claim (id, user_id, subject_id, text_tag, text_ct, holders_ct) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (user_id, text_tag) DO NOTHING", [id, userId, sid, t, seal(userId, text), seal(userId, JSON.stringify(holders))]);
    const claimId = (await q.query("SELECT id FROM learn_claim WHERE user_id = $1 AND text_tag = $2", [userId, t])).rows[0].id as string;
    await q.query("INSERT INTO learn_claim_stance (id, user_id, claim_id, stance, note_ct) VALUES ($1, $2, $3, $4, $5)", [randomUUID(), userId, claimId, c.stance, note ? seal(userId, note) : null]);
    return claimId;
  });
}

/** A new stance on a claim the person already keeps. */
export async function setStance(userId: string, claimId: string, stance: string, note?: string): Promise<void> {
  if (!isStance(stance)) throw new Error(`a stance is one of ${STANCES.map((s) => s.id).join(", ")}`);
  if (!/^[0-9a-f-]{36}$/.test(claimId)) throw new Error("no such claim");
  const own = await db().query("SELECT 1 FROM learn_claim WHERE id = $1 AND user_id = $2", [claimId, userId]);
  if (!own.rows[0]) throw new Error("no such claim");
  await db().query("INSERT INTO learn_claim_stance (id, user_id, claim_id, stance, note_ct) VALUES ($1, $2, $3, $4, $5)", [randomUUID(), userId, claimId, stance, note?.trim() ? seal(userId, note.trim()) : null]);
}

export async function listClaims(userId: string): Promise<ClaimOut[]> {
  const claims = await db().query("SELECT c.id, c.text_ct, c.holders_ct, c.subject_id, s.name_ct FROM learn_claim c LEFT JOIN learn_subject s ON s.id = c.subject_id WHERE c.user_id = $1 ORDER BY c.created_at", [userId]);
  const lines = await db().query("SELECT claim_id, stance, note_ct, created_at FROM learn_claim_stance WHERE user_id = $1 ORDER BY created_at, id", [userId]);
  const by = new Map<string, StanceLine[]>();
  for (const l of lines.rows) {
    const line: StanceLine = { stance: l.stance as Stance, at: new Date(l.created_at as string).toISOString(), ...(l.note_ct ? { note: open(userId, l.note_ct as string) } : {}) };
    by.set(l.claim_id as string, [...(by.get(l.claim_id as string) ?? []), line]);
  }
  return claims.rows.map((c) => {
    const history = by.get(c.id as string) ?? [];
    return {
      id: c.id as string,
      text: open(userId, c.text_ct as string),
      holders: JSON.parse(open(userId, c.holders_ct as string)) as Holder[],
      subject: c.subject_id ? { id: c.subject_id as string, name: open(userId, c.name_ct as string) } : null,
      stance: currentStance(history),
      history,
    };
  });
}
