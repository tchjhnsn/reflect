// The behaviour every host's Journal must have. Each host runs it against itself.
import { expect, it } from "vitest";
import type { IsoDate } from "./person.ts";
import { FRAMES } from "./reflect.ts";
import type { Journal } from "./store.ts";

const DAY = "2026-11-10" as IsoDate;

export function journalContract(make: () => Promise<Journal>): void {
  it("keeps an entry's parts in order until it closes", async () => {
    const j = await make();
    const id = await j.createEntry("Rain at the park.", DAY, null);
    await j.addPart(id, "We went anyway.");
    const [e] = await j.listEntries();
    expect(e?.parts.map((p) => p.text)).toEqual(["Rain at the park.", "We went anyway."]);
    expect(e?.localDate).toBe(DAY);
    expect(await j.entryText(id)).toBe("Rain at the park.\n\nWe went anyway.");
    await j.closeEntry(id);
    await expect(j.addPart(id, "late")).rejects.toThrow(/closed/);
    expect((await j.listEntries())[0]?.closedAt).not.toBeNull();
  });

  it("lists newest first and refuses an unknown entry", async () => {
    const j = await make();
    const a = await j.createEntry("first", DAY, null);
    const b = await j.createEntry("second", DAY, "p1");
    const list = await j.listEntries();
    expect(list.slice(0, 2).map((e) => e.id)).toEqual([b, a]);
    expect(list[0]?.promptId).toBe("p1");
    await expect(j.addPart("00000000-0000-0000-0000-000000000000", "x")).rejects.toThrow(/no such entry/);
    expect(await j.entryText("00000000-0000-0000-0000-000000000000")).toBeNull();
  });

  it("keeps answers with the frame's own question", async () => {
    const j = await make();
    const id = await j.createEntry("Called Dad.", DAY, null);
    const frame = FRAMES[0]!;
    await expect(j.startReflection(id, "no-such-frame")).rejects.toThrow(/no such frame/);
    const r = await j.startReflection(id, frame.id);
    await j.keepAnswer(r, 0, "I felt close to him.");
    await expect(j.keepAnswer(r, 99, "x")).rejects.toThrow(/no such question/);
    const [e] = await j.listEntries();
    expect(e?.reflections?.[0]?.answers[0]).toMatchObject({ seq: 0, question: frame.questions[0]!.text, text: "I felt close to him." });
  });
}
