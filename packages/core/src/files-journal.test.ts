import { mkdtemp, readFile, readdir, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { filesJournal, headExtras, parseEntryFile } from "./files-journal.ts";
import { journalContract } from "./journal-contract.ts";
import { parseMemory, verifyMemory } from "./memory.ts";

const dirs: string[] = [];
async function folder(): Promise<string> {
  const d = await mkdtemp(join(tmpdir(), "reflect-"));
  dirs.push(d);
  return d;
}
afterAll(async () => Promise.all(dirs.map((d) => rm(d, { recursive: true, force: true }))));

describe("journal in a folder", () => {
  let tick = 0;
  journalContract(async () => filesJournal(await folder(), "p-test", { now: () => new Date(Date.UTC(2026, 10, 10, 0, 0, tick++)) }));

  it("logs every write in a chain that holds", async () => {
    const root = await folder();
    const j = filesJournal(root, "p-test");
    const id = await j.createEntry("one", "2026-11-10" as never, null);
    await j.addPart(id, "two");
    await j.keepLabels(id, [{ element: "emotion", weight: 0.9, decision: "confirmed" }], []);
    const lines = parseMemory(await readFile(join(root, "memory.jsonl"), "utf8"));
    expect(lines.map((l) => l.op)).toEqual(["create", "append", "create"]);
    expect(verifyMemory(lines)).toEqual([]);
    const broken = lines.map((l, i) => (i === 1 ? { ...l, path: "elsewhere" } : l));
    expect(verifyMemory(broken)).not.toEqual([]);
    expect(await readdir(join(root, "people/p-test/entries"))).toEqual([`2026-11-10-${id}.md`]);
  });

  it("reads an entry the person wrote by hand, with no front matter", async () => {
    const root = await folder();
    await mkdir(join(root, "people/p-test/entries"), { recursive: true });
    await writeFile(join(root, "people/p-test/entries/2026-11-09-walk.md"), "Walked by the river.\n");
    const [e] = await filesJournal(root, "p-test").listEntries();
    expect(e).toMatchObject({ id: "2026-11-09-walk", localDate: "2026-11-09", closedAt: null });
    expect(e?.parts.map((p) => p.text)).toEqual(["Walked by the river."]);
  });

  it("round-trips parts with their times", () => {
    const text = "---\nid: x\nlocal_date: 2026-11-10\ncreated_at: 2026-11-10T01:00:00.000Z\n---\n\nA\n\n<!-- part 1 at 2026-11-10T02:00:00.000Z -->\n\nB\n";
    expect(parseEntryFile("2026-11-10-x.md", text).parts).toEqual([
      { seq: 0, text: "A", createdAt: "2026-11-10T01:00:00.000Z" },
      { seq: 1, text: "B", createdAt: "2026-11-10T02:00:00.000Z" },
    ]);
  });

  it("keeps front matter the person added (Obsidian tags, aliases) on later writes", async () => {
    const root = await folder();
    await mkdir(join(root, "journal"), { recursive: true });
    await writeFile(join(root, "journal/2026-11-09-walk.md"), "---\ntags:\n  - river\nid: walk\naliases: [stroll]\n---\n\nWalked.\n");
    expect(headExtras(await readFile(join(root, "journal/2026-11-09-walk.md"), "utf8"))).toEqual(["tags:", "  - river", "aliases: [stroll]"]);
    const j = filesJournal(root, "p-test", { layout: "vault" });
    await j.addPart("walk", "Then home.");
    const text = await readFile(join(root, "journal/2026-11-09-walk.md"), "utf8");
    expect(text).toContain("tags:\n  - river\naliases: [stroll]\n---");
    expect(parseEntryFile("2026-11-09-walk.md", text).parts.map((p) => p.text)).toEqual(["Walked.", "Then home."]);
  });
});
