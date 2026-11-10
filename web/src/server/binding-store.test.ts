import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { asFeltThen, presentMeaning, type Concept } from "@reflect/core/bindings";
import type { IsoDate } from "@reflect/core/person";
import { conceptHistory, keepReading, onThisConcept, readings } from "./binding-store";
import { createEntry } from "./journal-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, '', $2, false, now(), now())`, [id, `${id}@example.test`]);
  return id;
}

const dad: Concept = { kind: "entity", name: "Dad", sort: "person" };
const lake: Concept = { kind: "entity", name: "the lake", sort: "place" };
const fish: Concept = { kind: "event", name: "fish" };

describe.skipIf(!live)("bindings on Neon", () => {
  afterAll(() => db().end());

  it("keeps what each element was attached to (nouns, verbs, qualities) and a concept's history across entries", async () => {
    const user = await newUser();
    const sad = await createEntry(user, "Dad is sick. We used to fish at the lake, quiet mornings.", "2020-06-01" as IsoDate, null);
    await keepReading(user, sad, [
      { element: "emotion", weight: 0.95, on: [{ concept: dad, words: "Dad", role: "agent" }, { concept: fish, words: "fish" }, { concept: lake, words: "the lake", role: "place" }] },
      { element: "temporality", weight: 0.6, on: [{ concept: { kind: "quality", name: "quiet" }, words: "quiet", role: "manner" }, { concept: { kind: "time", name: "morning" } }] },
    ]);
    const glad = await createEntry(user, "Took my son fishing at the lake today.", "2026-06-01" as IsoDate, null);
    await keepReading(user, glad, [{ element: "emotion", weight: 0.2, on: [{ concept: { kind: "event", name: "FISH" }, words: "fishing" }, { concept: { ...lake, name: "The  Lake" }, words: "the lake" }] }]);

    const r = await readings(user, sad);
    expect(asFeltThen(r)?.bindings.find((b) => b.element === "emotion")).toEqual({
      element: "emotion", weight: expect.closeTo(0.95, 5),
      on: [{ concept: dad, words: "Dad", role: "agent" }, { concept: fish, words: "fish" }, { concept: lake, words: "the lake", role: "place" }],
    });
    const history = await conceptHistory(user, fish);
    expect(history.map((p) => [p.localDate, p.element])).toEqual([["2020-06-01", "emotion"], ["2026-06-01", "emotion"]]);
    expect(await onThisConcept(user, { ...lake, name: "THE LAKE" })).toEqual([sad, glad]);
    expect(presentMeaning(history, new Date().toISOString()).emotion!.strength).toBeGreaterThan(1.5);

    const raw = await db().query("SELECT name_ct FROM concept WHERE user_id = $1", [user]);
    expect(raw.rows.every((x: { name_ct: string }) => !x.name_ct.includes("lake"))).toBe(true);
    await expect(db().query("UPDATE binding SET weight = 0 WHERE entry_id = $1", [sad])).rejects.toThrow(/written once/);
  });

  it("refuses words that are not in the entry, and another person's entry", async () => {
    const user = await newUser();
    const other = await newUser();
    const id = await createEntry(user, "Quiet morning.", "2026-11-10" as IsoDate, null);
    await expect(keepReading(user, id, [{ element: "emotion", weight: 0.5, on: [{ concept: { kind: "quality", name: "loud" }, words: "loud" }] }])).rejects.toThrow(/not in the entry/);
    await expect(keepReading(other, id, [{ element: "emotion", weight: 0.5, on: [{ concept: { kind: "time", name: "morning" } }] }])).rejects.toThrow(/no such entry/);
  });
});
