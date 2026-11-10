import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { asFeltThen, presentMeaning } from "@reflect/core/bindings";
import type { IsoDate } from "@reflect/core/person";
import { cueHistory, keepReading, onThisCue, readings } from "./binding-store";
import { createEntry } from "./journal-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, '', $2, false, now(), now())`, [id, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("bindings on Neon", () => {
  afterAll(() => db().end());

  it("keeps what each element was attached to, and a cue's history across entries", async () => {
    const user = await newUser();
    const sad = await createEntry(user, "Dad is sick. We used to fish at the lake.", "2020-06-01" as IsoDate, null);
    await keepReading(user, sad, [
      { element: "emotion", weight: 0.95, cues: [{ kind: "person", value: "Dad" }, { kind: "place", value: "the lake" }] },
      { element: "temporality", weight: 0.6, cues: [{ kind: "phrase", value: "used to fish" }] },
    ]);
    const glad = await createEntry(user, "Took my son to the lake today.", "2026-06-01" as IsoDate, null);
    await keepReading(user, glad, [{ element: "emotion", weight: 0.2, cues: [{ kind: "place", value: "The  Lake" }] }]);

    const r = await readings(user, sad);
    expect(asFeltThen(r)?.bindings.find((b) => b.element === "emotion")).toEqual({ element: "emotion", weight: expect.closeTo(0.95, 5), cues: [{ kind: "person", value: "Dad" }, { kind: "place", value: "the lake" }] });
    const history = await cueHistory(user, { kind: "place", value: "the lake" });
    expect(history.map((p) => [p.localDate, p.element])).toEqual([["2020-06-01", "emotion"], ["2026-06-01", "emotion"]]);
    expect(await onThisCue(user, { kind: "place", value: "THE LAKE" })).toEqual([sad, glad]);
    expect(presentMeaning(history, new Date().toISOString()).emotion!.strength).toBeGreaterThan(1.5);

    const raw = await db().query("SELECT value_ct FROM cue WHERE user_id = $1", [user]);
    expect(raw.rows.every((x: { value_ct: string }) => !x.value_ct.includes("lake"))).toBe(true);
    await expect(db().query("UPDATE binding SET weight = 0 WHERE entry_id = $1", [sad])).rejects.toThrow(/written once/);
  });

  it("refuses a phrase that is not in the entry, and another person's entry", async () => {
    const user = await newUser();
    const other = await newUser();
    const id = await createEntry(user, "Quiet morning.", "2026-11-10" as IsoDate, null);
    await expect(keepReading(user, id, [{ element: "emotion", weight: 0.5, cues: [{ kind: "phrase", value: "loud night" }] }])).rejects.toThrow(/not in the entry/);
    await expect(keepReading(other, id, [{ element: "emotion", weight: 0.5, cues: [{ kind: "time", value: "morning" }] }])).rejects.toThrow(/no such entry/);
  });
});
