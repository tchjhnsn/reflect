import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import type { IsoDate } from "@/core/person";
import { addBelief, choosePack, chosenPacks, letGo, loadMe, stateFactNow, stateFriendNow } from "./me-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, '', $2, false, now(), now())`, [id, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("me store on Neon", () => {
  afterAll(() => db().end());

  it("keeps every dated value of a fact and the friend", async () => {
    const u = await newUser();
    await stateFactNow(u, "occupation", "Teacher", "2026-01-01" as IsoDate);
    await stateFactNow(u, "occupation", "Nurse", "2026-11-09" as IsoDate);
    await stateFriendNow(u, "Sam", "2026-11-09" as IsoDate);
    const me = await loadMe(u);
    expect(me.facts.occupation.map((f) => [f.value, f.validTo])).toEqual([["Nurse", null], ["Teacher", "2026-11-09"]]);
    expect(me.closestFriend[0]?.value).toBe("Sam");
  });

  it("keeps beliefs and values apart from facts and lets one go without deleting it", async () => {
    const u = await newUser();
    await addBelief(u, "value", "Honesty matters.", "2026-11-09" as IsoDate);
    await addBelief(u, "belief", "People can change.", "2026-11-09" as IsoDate);
    let me = await loadMe(u);
    const honesty = me.beliefs.find((b) => b.kind === "value");
    expect(honesty?.text).toBe("Honesty matters.");
    await letGo(u, honesty!.id, "2026-11-10" as IsoDate);
    me = await loadMe(u);
    expect(me.beliefs.find((b) => b.id === honesty!.id)?.validTo).toBe("2026-11-10");
    expect(me.beliefs).toHaveLength(2);
    await expect(db().query("DELETE FROM person_belief WHERE user_id = $1", [u])).rejects.toThrow(/append-only/);
  });

  it("takes the latest pack choice", async () => {
    const u = await newUser();
    await choosePack(u, "body", true);
    await choosePack(u, "decisions", true);
    await choosePack(u, "body", false);
    expect(await chosenPacks(u)).toEqual(["decisions"]);
    await expect(choosePack(u, "nope", true)).rejects.toThrow();
  });
});
