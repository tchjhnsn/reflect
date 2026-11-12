import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { addMove, chainHolds, getItem, listItems } from "./item-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(name: string): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, $2, $3, false, now(), now())`, [id, name, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("items on Neon", () => {
  afterAll(() => db().end());

  it("keeps an item as sealed, chained events the statechart checks; never edited; another person cannot add to it", async () => {
    const user = await newUser("Item Test");
    const id = await addMove(user, null, { kind: "stated", text: "I put books down too easily.", from: { kind: "act", act: { at: "2026-11-12T09:00:00Z", app: "learn", kind: "put_down", object: "Ulysses" } } });
    await expect(addMove(user, id, { kind: "planned", text: "Finish it." })).rejects.toThrow(/observed cannot be planned/);
    await addMove(user, id, { kind: "reflected", outcome: "held" });
    await addMove(user, id, { kind: "examine" });
    await addMove(user, id, { kind: "examine_step", step: "ground", text: "Finishing is a habit of will." });
    const item = (await getItem(user, id))!;
    expect(item.state).toBe("examining");
    expect(item.from).toMatchObject({ kind: "act" });
    expect(item.notes[0]?.text).toBe("Finishing is a habit of will.");
    expect((await listItems(user)).map((x) => x.id)).toEqual([id]);
    expect(await chainHolds(user, id)).toBe(true);
    const raw = (await db().query("SELECT body_ct FROM item_event WHERE item_id = $1", [id])).rows.map((r) => r.body_ct as string).join(" ");
    expect(raw).not.toContain("Ulysses");
    await expect(db().query("UPDATE item_event SET kind = 'done' WHERE item_id = $1", [id])).rejects.toThrow(/append-only/);
    const other = await newUser("Other");
    await expect(addMove(other, id, { kind: "examine" })).rejects.toThrow(/No such item/);
  });
});
