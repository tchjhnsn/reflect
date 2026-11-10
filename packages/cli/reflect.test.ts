import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { main } from "./reflect.ts";

const dirs: string[] = [];
afterAll(async () => Promise.all(dirs.map((d) => rm(d, { recursive: true, force: true }))));

async function run(...args: string[]): Promise<{ code: number; out: string[] }> {
  const out: string[] = [];
  const code = await main(args, (s) => out.push(s));
  return { code, out };
}

describe("reflect CLI", () => {
  it("starts an instance, writes, and checks it", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    expect((await run("init", "--name", "Tester", "--dir", dir)).code).toBe(0);
    expect(await readFile(join(dir, ".github/workflows/reflect.yml"), "utf8")).toContain("tchjhnsn/reflect/action@main");
    await expect(run("init", "--name", "Again", "--dir", dir)).rejects.toThrow(/already/);
    const id = (await run("write", "Walked to the river.", "--date", "2026-11-10", "--dir", dir)).out[0]!;
    await run("add", id, "It was cold.", "--dir", dir);
    await run("close", id, "--dir", dir);
    await expect(run("add", id, "late", "--dir", dir)).rejects.toThrow(/closed/);
    expect((await run("list", "--dir", dir)).out[0]).toContain("closed");
    expect(await run("check", "--dir", dir)).toEqual({ code: 0, out: ["ok"] });

    await writeFile(join(dir, "journal", "2026-11-09-hand.md"), "By hand.\n");
    const before = await run("check", "--dir", dir);
    expect(before.out).toContain("written outside reflect: journal/2026-11-09-hand.md");
    await run("check", "--adopt", "--dir", dir);
    expect(await run("check", "--dir", dir)).toEqual({ code: 0, out: ["ok"] });
  });

  it("fails on a grant that names entries", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    const { mkdir } = await import("node:fs/promises");
    await mkdir(join(dir, "grants"));
    await writeFile(join(dir, "grants", "mentura.json"), JSON.stringify({ to: "mentura", paths: ["intents", "journal"] }));
    const r = await run("check", "--dir", dir);
    expect(r.code).toBe(1);
    expect(r.out.join("\n")).toMatch(/can never name journal/);
  });

  it("reads an entry into a draft, and keeps it as the person left it", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    process.env.REFLECT_READER_STUB = "1";
    try {
      await run("init", "--name", "Tester", "--dir", dir);
      const id = (await run("write", "I was always the independent one.", "--dir", dir)).out[0]!;
      expect((await run("read", "--pending", "--dir", dir)).out).toHaveLength(1);
      expect((await run("read", "--pending", "--dir", dir)).out).toHaveLength(0);
      const path = join(dir, "reflections", `${id}.draft.json`);
      const draft = JSON.parse(await readFile(path, "utf8"));
      expect(draft.chosen).toEqual(["emotion", "attachment"]);
      expect(draft.self_terms[0]).toMatchObject({ quote: "the independent one", model_basis: "imagination", basis: "imagination", keep: true });
      expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);
      draft.chosen = ["attachment", "will"];
      draft.self_terms[0].basis = "will";
      await writeFile(path, JSON.stringify(draft));
      expect((await run("keep", "--all", "--dir", dir)).out).toEqual([`kept: ${id} (attachment, will)`]);
      const refl = JSON.parse(await readFile(join(dir, "reflections", `${id}.json`), "utf8"));
      const decisions = Object.fromEntries(refl.labelings[0].labels.map((l: { element: string; decision: string }) => [l.element, l.decision]));
      expect(decisions).toMatchObject({ emotion: "rejected", attachment: "confirmed", will: "added", reason: "absent" });
      expect(refl.labelings[0].terms[0]).toMatchObject({ modelBasis: "imagination", basis: "will", decision: "corrected" });
      expect((await run("list", "--dir", dir)).out).toHaveLength(1);
      expect((await run("read", "--pending", "--dir", dir)).out).toHaveLength(0);
      expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);
    } finally {
      delete process.env.REFLECT_READER_STUB;
    }
  });

  it("exports a bundle, imports it into another instance, and refuses writes after moving", async () => {
    const a = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    const b = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(a, b);
    await run("init", "--name", "Tester", "--dir", a);
    const id = (await run("write", "Rain.", "--date", "2026-11-01", "--dir", a)).out[0]!;
    await run("add", id, "We went anyway.", "--dir", a);
    const file = join(a, "..", `${id}.bundle.json`);
    dirs.push(file);
    await run("export", "--out", file, "--moving-to", "reflect-app", "--dir", a);
    await expect(run("write", "more", "--dir", a)).rejects.toThrow(/moved to reflect-app/);
    expect((await run("check", "--dir", a)).out).toEqual(["ok"]);
    await run("init", "--name", "Tester", "--dir", b);
    expect((await run("import", file, "--dir", b)).code).toBe(0);
    await expect(run("import", file, "--dir", b)).rejects.toThrow(/already has entries/);
    expect((await run("list", "--dir", b)).out[0]).toContain("Rain.");
    expect((await run("check", "--dir", b)).out).toEqual(["ok"]);
  });

  it("lays out a vault any editor can open", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    expect(JSON.parse(await readFile(join(dir, "reflect.json"), "utf8")).layout).toBe("vault");
    expect((await readdir(dir)).sort()).toEqual([".github", ".gitignore", "concepts", "intents", "journal", "library", "memory.jsonl", "person.json", "reflect.json", "reflections", "telos", "ventures"]);
    expect((await readdir(join(dir, "ventures"))).sort()).toEqual(["community", "friendship", "household", "work"]);
    expect(await readFile(join(dir, "ventures", "household", "README.md"), "utf8")).toContain("# Household");
    // A note the person adds anywhere in the vault's folders is a record written outside reflect.
    await writeFile(join(dir, "library", "meditations.md"), "# Meditations\n");
    expect((await run("check", "--dir", dir)).out).toEqual(["written outside reflect: library/meditations.md", "ok"]);
    // Obsidian's own settings are not records.
    const { mkdir } = await import("node:fs/promises");
    await mkdir(join(dir, ".obsidian"));
    await writeFile(join(dir, ".obsidian", "app.json"), "{}");
    await run("check", "--adopt", "--dir", dir);
    expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);
  });

  it("sparks: a noun to write about, kept as an entry that names the noun as a concept", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    const prompt = (await run("spark", "--minutes", "15", "--dir", dir)).out[0]!.split("\n");
    expect(prompt[1]).toMatch(/^Write for 15 minutes, without stopping, about /);
    expect(prompt[2]).toContain("--minutes 15");
    expect((await run("spark", "--sentence")).out[0]).toContain("Write one sentence about");
    await expect(run("spark", "--minutes", "3")).rejects.toThrow(/--minutes is one of 1, 5, 10, 15/);
    await expect(run("spark", "--minutes", "5", "--sentence")).rejects.toThrow(/not both/);

    const a = (await run("spark", "kitchen table", "The leaf folds down when we eat alone.", "--minutes", "5", "--date", "2026-11-10", "--dir", dir)).out[0]!;
    const b = (await run("spark", "kitchen table", "Flour in the grain.", "--sentence", "--date", "2026-11-11", "--dir", dir)).out[0]!;
    const note = await readFile(join(dir, "concepts", "kitchen table.md"), "utf8");
    expect(note).toContain("kind: entity");
    const file = await readFile(join(dir, "journal", `2026-11-10-${a}.md`), "utf8");
    expect(file).toContain("prompt_id: spark/5m");
    expect(file).toContain('concept: "[[kitchen table]]"');
    expect(await readFile(join(dir, "journal", `2026-11-11-${b}.md`), "utf8")).toContain("spark: sentence");
    // A spark is kept closed: the time is up.
    await expect(run("add", a, "more", "--dir", dir)).rejects.toThrow(/closed/);
    expect((await run("list", "--dir", dir)).out).toHaveLength(2);
    expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);
  });

  it("still works on an instance in the older people/<id> layout", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await writeFile(join(dir, "reflect.json"), JSON.stringify({ reflect: 1, person: "p-0000abcd", host: "files", encrypt: null }));
    await writeFile(join(dir, "memory.jsonl"), "");
    await run("check", "--adopt", "--dir", dir);
    const id = (await run("write", "Old place.", "--date", "2026-11-01", "--dir", dir)).out[0]!;
    expect(await readdir(join(dir, "people", "p-0000abcd", "entries"))).toEqual([`2026-11-01-${id}.md`]);
    expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);
  });

  it("refuses to run outside an instance", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await expect(run("list", "--dir", dir)).rejects.toThrow(/reflect init/);
  });
});
