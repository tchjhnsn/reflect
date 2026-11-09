import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { EMPTY_PERSON, stateFact, stateRelationship, type IsoDate } from "@/core/person";
import { currentName, hasOnboarded, saveOnboarding } from "./person-store";
import { open } from "./crypto";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);
const AT = "2026-11-09" as IsoDate;

async function newUser(): Promise<string> {
  const id = randomUUID();
  await db().query(
    `INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, '', $2, false, now(), now())`,
    [id, `${id}@example.test`],
  );
  return id;
}

const person = stateRelationship(stateFact(EMPTY_PERSON, "name", "Ada", AT), "Sam", "closest_friend", AT);

describe.skipIf(!live)("person store on Neon", () => {
  afterAll(() => db().end());

  it("writes each record once, encrypted, and refuses a second onboarding", async () => {
    const user = await newUser();
    expect(await saveOnboarding(user, person, AT)).toBe(true);
    expect(await saveOnboarding(user, person, AT)).toBe(false);
    expect(await hasOnboarded(user)).toBe(true);
    expect(await currentName(user)).toBe("Ada");
    const rows = await db().query("SELECT value_ct FROM person_fact WHERE user_id = $1", [user]);
    expect(rows.rowCount).toBe(1);
    const ct = (rows.rows[0] as { value_ct: string }).value_ct;
    expect(ct).not.toContain("Ada");
    expect(open(user, ct)).toBe("Ada");
    const friend = await db().query("SELECT name_ct FROM person_relationship WHERE user_id = $1", [user]);
    expect((friend.rows[0] as { name_ct: string }).name_ct).not.toContain("Sam");
  });

  it("refuses edits and deletes but allows closing a row once", async () => {
    const user = await newUser();
    await saveOnboarding(user, person, AT);
    await expect(db().query("UPDATE person_fact SET value_ct = 'x' WHERE user_id = $1", [user])).rejects.toThrow(/never edited/);
    await expect(db().query("DELETE FROM person_fact WHERE user_id = $1", [user])).rejects.toThrow(/append-only/);
    await db().query("UPDATE person_relationship SET valid_to = '2026-12-01' WHERE user_id = $1", [user]);
    await expect(db().query("UPDATE person_relationship SET valid_to = '2027-01-01' WHERE user_id = $1", [user])).rejects.toThrow(/never edited/);
  });

  it("keeps one person's records from another's key", async () => {
    const a = await newUser();
    const b = await newUser();
    await saveOnboarding(a, person, AT);
    const ct = ((await db().query("SELECT value_ct FROM person_fact WHERE user_id = $1", [a])).rows[0] as { value_ct: string }).value_ct;
    expect(() => open(b, ct)).toThrow();
  });
});
