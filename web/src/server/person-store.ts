import type { PoolClient } from "@neondatabase/serverless";
import { db } from "@/db/pool";
import type { IsoDate, PersonState } from "@reflect/core/person";
import { open, seal } from "./crypto";

/** Write what onboarding produced in one transaction. Idempotent: a second call for a person who finished does nothing. */
export async function saveOnboarding(userId: string, person: PersonState, at: IsoDate): Promise<boolean> {
  const client: PoolClient = await db().connect();
  try {
    await client.query("BEGIN");
    const first = await client.query("INSERT INTO onboarding_done (user_id) VALUES ($1) ON CONFLICT DO NOTHING", [userId]);
    if (first.rowCount === 0) {
      await client.query("ROLLBACK");
      return false;
    }
    for (const f of person.facts.filter((x) => x.validTo === null)) {
      await client.query("UPDATE person_fact SET valid_to = $3 WHERE user_id = $1 AND key = $2 AND valid_to IS NULL", [userId, f.key, at]);
      await client.query(
        "INSERT INTO person_fact (user_id, key, value_ct, valid_from, source) VALUES ($1, $2, $3, $4, 'stated')",
        [userId, f.key, seal(userId, f.value), f.validFrom],
      );
    }
    for (const r of person.relationships.filter((x) => x.validTo === null)) {
      await client.query("UPDATE person_relationship SET valid_to = $3 WHERE user_id = $1 AND kind = $2 AND valid_to IS NULL", [userId, r.kind, at]);
      await client.query(
        "INSERT INTO person_relationship (user_id, kind, name_ct, valid_from, source) VALUES ($1, $2, $3, $4, 'stated')",
        [userId, r.kind, seal(userId, r.name), r.validFrom],
      );
    }
    for (const h of person.holds.filter((x) => x.validTo === null)) {
      await client.query("UPDATE person_hold SET valid_to = $3 WHERE user_id = $1 AND subject = $2 AND valid_to IS NULL", [userId, h.subject, at]);
      await client.query(
        "INSERT INTO person_hold (user_id, subject, value_ct, valid_from, source) VALUES ($1, $2, $3, $4, 'stated')",
        [userId, h.subject, seal(userId, h.value), h.validFrom],
      );
    }
    await client.query("COMMIT");
    return true;
  } catch (e) {
    await client.query("ROLLBACK");
    throw e;
  } finally {
    client.release();
  }
}

export async function hasOnboarded(userId: string): Promise<boolean> {
  const r = await db().query("SELECT 1 FROM onboarding_done WHERE user_id = $1", [userId]);
  return (r.rowCount ?? 0) > 0;
}

export async function currentName(userId: string): Promise<string | null> {
  const r = await db().query("SELECT value_ct FROM person_fact WHERE user_id = $1 AND key = 'name' AND valid_to IS NULL", [userId]);
  const row = r.rows[0] as { value_ct: string } | undefined;
  return row ? open(userId, row.value_ct) : null;
}
