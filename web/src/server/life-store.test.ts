import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { addLifeRecord, keepLifeAct, lifeActs, lifeRecord, lifeRecords, lifeStateChanges, removeLifeRecord, restoreLifeRecord, saveLifeRecord, setLifeState } from "./life-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

describe.skipIf(!live)("the life apps' records, on Neon", () => {
  afterAll(() => db().end());

  it("keeps records sealed, with states and acts kept as they happen", async () => {
    const id = randomUUID();
    await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, 'Ann', $2, false, now(), now())`, [id, `${id}@example.test`]);
    await expect(addLifeRecord(id, "friends", "friend", {})).rejects.toThrow(/name/);
    await expect(addLifeRecord(id, "nowhere", "friend", { name: "Sam" })).rejects.toThrow();
    const sam = await addLifeRecord(id, "friends", "friend", { name: "Sam", met: "At school", junk: "x" });
    expect(await lifeRecord(id, sam)).toMatchObject({ app: "friends", kind: "friend", state: "acquaintance", body: { name: "Sam", met: "At school" } });
    const raw = await db().query("SELECT body_ct FROM life_record WHERE id = $1", [sam]);
    expect(raw.rows[0].body_ct).not.toContain("Sam");

    await setLifeState(id, sam, "close");
    await setLifeState(id, sam, "close");
    await expect(setLifeState(id, sam, "married")).rejects.toThrow();
    expect((await lifeStateChanges(id)).map((s) => s.state).sort()).toEqual(["acquaintance", "close"]);

    await saveLifeRecord(id, sam, { name: "Samuel" });
    expect((await lifeRecord(id, sam))!.body).toEqual({ name: "Samuel" });

    await keepLifeAct(id, sam, { kind: "saw", day: "2026-11-10", note: "Coffee" });
    await expect(keepLifeAct(id, sam, { kind: "workout", day: "2026-11-10" })).rejects.toThrow();
    await expect(keepLifeAct(id, sam, { kind: "saw", day: "2999-01-01" })).rejects.toThrow();
    expect(await lifeActs(id, sam)).toMatchObject([{ kind: "saw", day: "2026-11-10", note: "Coffee", minutes: null }]);

    const food = await addLifeRecord(id, "home", "food", { how: "stocked", pantry: "on" });
    await expect(addLifeRecord(id, "home", "food", {})).rejects.toThrow(/one/);
    expect((await lifeRecords(id, "home")).map((r) => r.id)).toEqual([food]);

    await removeLifeRecord(id, sam);
    expect(await lifeRecord(id, sam)).toBeNull();
    expect(await lifeActs(id)).toEqual([]);
    await restoreLifeRecord(id, sam);
    expect((await lifeRecords(id)).length).toBe(2);
  });
});
