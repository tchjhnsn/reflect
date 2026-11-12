import { afterAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import { db } from "@/db/pool";
import { createWork, getWork, importWork, listWorks, saveWork } from "./work-store";
import { exportJournal } from "./journal-store";

const live = Boolean(process.env.DATABASE_URL && process.env.REFLECT_DATA_KEY);

async function newUser(name: string): Promise<string> {
  const id = randomUUID();
  await db().query(`INSERT INTO "user" (id, name, email, "emailVerified", "createdAt", "updatedAt") VALUES ($1, $2, $3, false, now(), now())`, [id, name, `${id}@example.test`]);
  return id;
}

describe.skipIf(!live)("works on Neon", () => {
  afterAll(() => db().end());

  it("starts from a template, keeps each changed save as a sealed version, never edited, and only for its owner", async () => {
    const user = await newUser("Work Test");
    const id = await createWork(user, "essay");
    expect((await getWork(user, id))?.body).toContain("## Thesis");
    expect((await saveWork(user, id, "On friendship", "Friends are made.")).saved).toBe(true);
    expect((await saveWork(user, id, "On friendship", "Friends are made.  ")).saved).toBe(false);
    const w = (await getWork(user, id))!;
    expect(w).toMatchObject({ title: "On friendship", body: "Friends are made.", kind: "essay", versions: 2 });
    expect((await listWorks(user)).map((x) => x.id)).toEqual([id]);
    const raw = (await db().query("SELECT title_ct, body_ct FROM work_version WHERE work_id = $1", [id])).rows.map((r) => `${r.title_ct} ${r.body_ct}`).join(" ");
    expect(raw).not.toContain("friendship");
    await expect(db().query("UPDATE work_version SET seq = 9 WHERE work_id = $1", [id])).rejects.toThrow(/append-only/);
    const other = await newUser("Other");
    expect(await getWork(other, id)).toBeNull();
    await expect(saveWork(other, id, "x", "y")).rejects.toThrow(/No such work/);
  });

  it("imports a file as a work and exports every work with the journal", async () => {
    const user = await newUser("Import Test");
    const id = await importWork(user, "on-friends.md", "# On friends\n\nThey are made, not found.");
    expect(await getWork(user, id)).toMatchObject({ kind: "entry", template: "import", title: "On friends", body: "They are made, not found." });
    await expect(importWork(user, "empty.txt", "  \n")).rejects.toThrow(/empty/);
    const b = await exportJournal(user);
    expect(b.works?.map((w) => w.title)).toEqual(["On friends"]);
  });
});
