import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import {
  bindingProblems, cleanAnchor, conceptKey,
  type Anchor, type Binding, type Concept, type ConceptKind, type ConceptPoint, type EntitySort, type Reading, type Role,
} from "@reflect/core/bindings";
import { open, seal, tag } from "./crypto";
import { entryText } from "./journal-store";

/**
 * Keep one reading of an entry: each element's weight and the concepts it was attached to, with the
 * entry's words for each. Concepts are kept once per person and found again by a keyed hash. All
 * bindings of a reading share one time. One transaction.
 */
export async function keepReading(userId: string, entryId: string, bindings: Binding[]): Promise<string> {
  const text = await entryText(userId, entryId);
  if (text === null) throw new Error("no such entry");
  const problems = bindingProblems(bindings, text);
  if (problems.length) throw new Error(problems.join("; "));
  const client = await db().connect();
  try {
    await client.query("BEGIN");
    const ids = new Map<string, string>();
    for (const { concept: c } of bindings.flatMap((b) => b.on).map(cleanAnchor)) {
      const t = tag(userId, conceptKey(c));
      if (ids.has(t)) continue;
      await client.query(
        "INSERT INTO concept (id, user_id, kind, sort, name_ct, key_tag) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (user_id, key_tag) DO NOTHING",
        [randomUUID(), userId, c.kind, c.sort ?? null, seal(userId, c.name), t],
      );
      ids.set(t, ((await client.query("SELECT id FROM concept WHERE user_id = $1 AND key_tag = $2", [userId, t])).rows[0] as { id: string }).id);
    }
    const readAt = ((await client.query("SELECT now() AS t")).rows[0] as { t: Date }).t;
    for (const b of bindings) {
      const id = randomUUID();
      await client.query("INSERT INTO binding (id, entry_id, user_id, element, weight, read_at) VALUES ($1, $2, $3, $4, $5, $6)", [id, entryId, userId, b.element, b.weight, readAt]);
      for (const [i, a] of b.on.map(cleanAnchor).entries()) {
        await client.query(
          "INSERT INTO binding_concept (binding_id, concept_id, position, words_ct, role) VALUES ($1, $2, $3, $4, $5)",
          [id, ids.get(tag(userId, conceptKey(a.concept)))!, i, a.words ? seal(userId, a.words) : null, a.role ?? null],
        );
      }
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

type Row = {
  binding_id: string; element: string; weight: number; read_at: Date;
  kind: ConceptKind; sort: EntitySort | null; name_ct: string; words_ct: string | null; role: Role | null;
};

/** Every reading of an entry, oldest first; the first is the entry as felt then. */
export async function readings(userId: string, entryId: string): Promise<Reading[]> {
  const rows = (await db().query(
    `SELECT b.id AS binding_id, b.element, b.weight, b.read_at, c.kind, c.sort, c.name_ct, bc.words_ct, bc.role
     FROM binding b JOIN binding_concept bc ON bc.binding_id = b.id JOIN concept c ON c.id = bc.concept_id
     WHERE b.user_id = $1 AND b.entry_id = $2 ORDER BY b.read_at, b.element, b.id, bc.position`,
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
      byBinding.set(r.binding_id, (b = { element: r.element, weight: r.weight, on: [] }));
      reading.bindings.push(b);
    }
    const concept: Concept = { kind: r.kind, name: open(userId, r.name_ct), ...(r.sort ? { sort: r.sort } : {}) };
    const anchor: Anchor = { concept, ...(r.words_ct ? { words: open(userId, r.words_ct) } : {}), ...(r.role ? { role: r.role } : {}) };
    b.on.push(anchor);
  }
  return [...out.values()];
}

/** Every binding of one concept across the person's entries, in the order the entries were written. */
export async function conceptHistory(userId: string, concept: Concept): Promise<ConceptPoint[]> {
  const rows = (await db().query(
    `SELECT b.entry_id, e.local_date::text AS local_date, b.element, b.weight, b.read_at, bc.role
     FROM concept c JOIN binding_concept bc ON bc.concept_id = c.id JOIN binding b ON b.id = bc.binding_id JOIN journal_entry e ON e.id = b.entry_id
     WHERE c.user_id = $1 AND c.key_tag = $2 ORDER BY e.created_at, b.read_at, b.element`,
    [userId, tag(userId, conceptKey(concept))],
  )).rows as { entry_id: string; local_date: string; element: string; weight: number; read_at: Date; role: Role | null }[];
  return rows.map((r) => ({
    entryId: r.entry_id, localDate: r.local_date, element: r.element, weight: r.weight, readAt: r.read_at.toISOString(),
    ...(r.role ? { role: r.role } : {}),
  }));
}

/** The entries a concept brings back, oldest first. */
export async function onThisConcept(userId: string, concept: Concept): Promise<string[]> {
  return [...new Set((await conceptHistory(userId, concept)).map((p) => p.entryId))];
}
