import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { bindingProblems, cleanCue, type Binding, type Cue, type CuePoint, type Reading } from "@reflect/core/bindings";
import { open, seal, tag } from "./crypto";
import { entryText } from "./journal-store";

/**
 * Keep one reading of an entry: each element's weight and what it was attached to. Cues are kept once per
 * person and found again by a keyed hash. All bindings of a reading share one time. One transaction.
 */
export async function keepReading(userId: string, entryId: string, bindings: Binding[]): Promise<string> {
  const text = await entryText(userId, entryId);
  if (text === null) throw new Error("no such entry");
  const problems = bindingProblems(bindings, text);
  if (problems.length) throw new Error(problems.join("; "));
  const client = await db().connect();
  try {
    await client.query("BEGIN");
    const cueIds = new Map<string, string>();
    for (const c of bindings.flatMap((b) => b.cues).map(cleanCue)) {
      const t = tag(userId, `${c.kind}:${c.value.toLowerCase()}`);
      if (cueIds.has(t)) continue;
      await client.query(
        "INSERT INTO cue (id, user_id, kind, value_ct, value_tag) VALUES ($1, $2, $3, $4, $5) ON CONFLICT (user_id, kind, value_tag) DO NOTHING",
        [randomUUID(), userId, c.kind, seal(userId, c.value), t],
      );
      const row = (await client.query("SELECT id FROM cue WHERE user_id = $1 AND kind = $2 AND value_tag = $3", [userId, c.kind, t])).rows[0] as { id: string };
      cueIds.set(t, row.id);
    }
    const readAt = ((await client.query("SELECT now() AS t")).rows[0] as { t: Date }).t;
    for (const b of bindings) {
      const id = randomUUID();
      await client.query("INSERT INTO binding (id, entry_id, user_id, element, weight, read_at) VALUES ($1, $2, $3, $4, $5, $6)", [id, entryId, userId, b.element, b.weight, readAt]);
      const ids = new Set(b.cues.map(cleanCue).map((c) => cueIds.get(tag(userId, `${c.kind}:${c.value.toLowerCase()}`))!));
      for (const cueId of ids) await client.query("INSERT INTO binding_cue (binding_id, cue_id) VALUES ($1, $2)", [id, cueId]);
    }
    await client.query("COMMIT");
    return readAt.toISOString();
  } catch (e) {
    await client.query("ROLLBACK");
    throw e;
  } finally {
    client.release();
  }
}

type Row = { binding_id: string; element: string; weight: number; read_at: Date; kind: Cue["kind"]; value_ct: string };

/** Every reading of an entry, oldest first; the first is the entry as felt then. */
export async function readings(userId: string, entryId: string): Promise<Reading[]> {
  const rows = (await db().query(
    `SELECT b.id AS binding_id, b.element, b.weight, b.read_at, c.kind, c.value_ct
     FROM binding b JOIN binding_cue bc ON bc.binding_id = b.id JOIN cue c ON c.id = bc.cue_id
     WHERE b.user_id = $1 AND b.entry_id = $2 ORDER BY b.read_at, b.element, b.id, c.kind`,
    [userId, entryId],
  )).rows as Row[];
  const out = new Map<string, Reading>();
  const byBinding = new Map<string, Binding>();
  for (const r of rows) {
    const at = r.read_at.toISOString();
    let reading = out.get(at);
    if (!reading) out.set(at, (reading = { readAt: at, bindings: [] }));
    let b = byBinding.get(r.binding_id);
    if (!b) {
      byBinding.set(r.binding_id, (b = { element: r.element, weight: r.weight, cues: [] }));
      reading.bindings.push(b);
    }
    b.cues.push({ kind: r.kind, value: open(userId, r.value_ct) });
  }
  return [...out.values()];
}

/** Every binding of one cue across the person's entries, in the order the entries were written. */
export async function cueHistory(userId: string, cue: Cue): Promise<CuePoint[]> {
  const c = cleanCue(cue);
  const rows = (await db().query(
    `SELECT b.entry_id, e.local_date::text AS local_date, b.element, b.weight, b.read_at
     FROM cue c JOIN binding_cue bc ON bc.cue_id = c.id JOIN binding b ON b.id = bc.binding_id JOIN journal_entry e ON e.id = b.entry_id
     WHERE c.user_id = $1 AND c.kind = $2 AND c.value_tag = $3 ORDER BY e.created_at, b.read_at, b.element`,
    [userId, c.kind, tag(userId, `${c.kind}:${c.value.toLowerCase()}`)],
  )).rows as { entry_id: string; local_date: string; element: string; weight: number; read_at: Date }[];
  return rows.map((r) => ({ entryId: r.entry_id, localDate: r.local_date, element: r.element, weight: r.weight, readAt: r.read_at.toISOString() }));
}

/** The entries a cue brings back, oldest first. */
export async function onThisCue(userId: string, cue: Cue): Promise<string[]> {
  return [...new Set((await cueHistory(userId, cue)).map((p) => p.entryId))];
}
