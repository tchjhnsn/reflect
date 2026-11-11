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
    expect((await readdir(dir)).sort()).toEqual([".github", ".gitignore", "belonging", "community", "concepts", "documents", "household", "intents", "journal", "learn", "legacy", "life", "line", "memory.jsonl", "needs", "person.json", "polis", "reflect.json", "reflections", "telos", "work"]);
    expect((await readdir(join(dir, "household"))).sort()).toEqual(["members", "home", "money"].sort());
    expect((await readdir(join(dir, "legacy"))).sort()).toEqual(["README.md", "bucket-list.md", "letters", "wishes.md"]);
    expect(await readFile(join(dir, "household", "members", "README.md"), "utf8")).toContain("past the threshold");
    // A note the person adds anywhere in the vault's folders is a record written outside reflect.
    await writeFile(join(dir, "learn", "meditations.md"), "# Meditations\n");
    expect((await run("check", "--dir", dir)).out).toEqual(["written outside reflect: learn/meditations.md", "ok"]);
    // A Naive venture under work/ keeps its own provenance; Reflect's check leaves it alone.
    const { mkdir: mk } = await import("node:fs/promises");
    await mk(join(dir, "work", "bakery"));
    await writeFile(join(dir, "work", "bakery", "venture.json"), "{}");
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

  it("keeps a custody checklist for official documents, with connections that hold some", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    const before = (await run("documents", "--dir", dir)).out;
    expect(before.at(-1)).toBe("0 of 28 checked");
    expect(before.find((l) => l.includes("social-security-card"))).toMatch(/\[ \] social-security-card  keep the paper safe\. Paper only/);
    expect(before.find((l) => l.includes("transcripts"))).toMatch(/connect the service that holds it, such as Parchment or National Student Clearinghouse/);

    expect((await run("connect", "school", "--name", "Parchment", "--kind", "issuer", "--holds", "transcripts,diploma", "--dir", dir)).out).toEqual(["connected: Parchment holds diploma, transcripts"]);
    await expect(run("connect", "x", "--name", "X", "--holds", "passport-photo", "--dir", dir)).rejects.toThrow(/unknown document type passport-photo/);
    await run("documents", "social-security-card", "original", "--where", "fire safe at home", "--dir", dir);
    await run("documents", "transcripts", "connection", "--connection", "school", "--dir", dir);
    await expect(run("documents", "passport", "connection", "--connection", "school", "--dir", dir)).rejects.toThrow(/does not hold passport/);
    await expect(run("documents", "passport", "lost", "--dir", dir)).rejects.toThrow(/custody is one of/);
    await expect(run("documents", "pasport", "original", "--dir", dir)).rejects.toThrow(/no such document type/);
    await run("documents", "military-discharge", "not-applicable", "--dir", dir);
    // A later check-off adds a line; the earlier one stays.
    await run("documents", "social-security-card", "original", "--where", "safe deposit box", "--dir", dir);
    const card = await readFile(join(dir, "documents", "social-security-card.md"), "utf8");
    expect(card).toContain('where: "safe deposit box"');
    expect(card).toMatch(/- \d{4}-\d{2}-\d{2} original: fire safe at home\n- \d{4}-\d{2}-\d{2} original: safe deposit box\n/);

    const after = (await run("documents", "--dir", dir)).out;
    expect(after.at(-1)).toBe("3 of 28 checked");
    expect(after).toContain("  [x] transcripts  connection Parchment");
    expect(after.find((l) => l.includes("diploma"))).toContain("[ ] diploma (held by Parchment)");
    expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);

    // check finds a hand-edited card that names a connection which does not hold the document.
    await writeFile(join(dir, "documents", "passport.md"), "---\ntype: passport\ncustody: connection\nconnection: school\n---\n");
    const bad = await run("check", "--dir", dir);
    expect(bad.code).toBe(1);
    expect(bad.out).toContain("problem: documents/passport.md: names a connection that does not hold it");
  });

  it("keeps how each need is met, and passages between states, with state rebuilt from them", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    await run("need", "food", "supplied", "--provider", "market", "--dir", dir);
    await expect(run("need", "water", "stocked", "--dir", dir)).rejects.toThrow(/--runway/);
    await expect(run("need", "wifi", "supplied", "--dir", dir)).rejects.toThrow(/a need is one of/);
    await run("need", "food", "stocked", "--runway", "6m", "--note", "garden and freezer", "--dir", dir);
    const food = await readFile(join(dir, "needs", "food.md"), "utf8");
    expect(food).toContain("state: stocked\nrunway: 6m");
    expect(food).toMatch(/- \S+ supplied by market\n- \S+ stocked for 6m: garden and freezer\n/);

    await run("passage", "line", "seeking-a-mate", "--on", "2019-03", "--dir", dir);
    await run("passage", "line", "pair-bond", "--on", "2021-06-12", "--note", "married", "--dir", dir);
    await run("passage", "provide", "learning", "--on", "2016-08", "--dir", dir);
    // Kept late, but it happened earlier: state goes by when it happened.
    await run("passage", "line", "seeking-a-mate", "--on", "2018-01", "--dir", dir);
    await expect(run("passage", "line", "married", "--dir", dir)).rejects.toThrow(/in line a state is one of/);
    const state = JSON.parse(await readFile(join(dir, "life", "state.json"), "utf8"));
    expect(state).toEqual({ provide: { state: "learning", since: "2016-08" }, line: { state: "pair-bond", since: "2021-06-12" } });
    expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);
    await writeFile(join(dir, "life", "state.json"), JSON.stringify({ line: { state: "raising-children", since: "2025-01" } }));
    expect((await run("check", "--dir", dir)).out).toContain("problem: life/state.json does not match life/passages.jsonl; it is rebuilt from the passages, never edited");
  });

  it("offers legacy prompts and keeps what you write as entries marked to survive you", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    expect((await run("legacy", "--dir", dir)).out[0]!.split("\n")[1]).toMatch(/^Then keep it: reflect legacy [a-z-]+ < what-you-wrote.txt$/);
    const id = (await run("legacy", "remembered", "As someone who kept his word.", "--date", "2026-11-10", "--dir", dir)).out[0]!;
    const file = await readFile(join(dir, "journal", `2026-11-10-${id}.md`), "utf8");
    expect(file).toContain("prompt_id: legacy/remembered\nlegacy: true\nprompt: \"How do you want the world to remember you?\"");
    // It stays open: you can come back to it.
    await run("add", id, "And who made people laugh.", "--dir", dir);
    expect(await readFile(join(dir, "journal", `2026-11-10-${id}.md`), "utf8")).toContain("legacy: true");
    await expect(run("legacy", "obituary", "x", "--dir", dir)).rejects.toThrow(/no such prompt/);
    const docs = (await run("documents", "--dir", dir)).out;
    expect(docs).toContain("legacy");
    expect(docs.find((l) => l.includes("life-insurance"))).toContain("connect the service that holds it");
    expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);
  });

  it("keeps a release plan that can only give the legacy folder and entries marked to survive you", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    await writeFile(join(dir, "legacy", "letters", "sam.md"), "Dear Sam,\n");
    const entry = (await run("legacy", "hopes", "That you stay brave.", "--date", "2026-11-10", "--dir", dir)).out[0]!;
    await run("write", "A private worry.", "--date", "2026-11-10", "--dir", dir);
    const empty = (await run("release", "--dir", dir)).out;
    expect(empty).toContain("note: legacy/letters/sam.md is given to no one yet.");

    await run("release", "give", "legacy/letters/sam.md", "--to", "Sam", "--contact", "sam@example.test", "--dir", dir);
    await run("release", "give", "legacy-entries", "--to", "Sam", "--contact", "sam@example.test", "--dir", dir);
    await expect(run("release", "give", "journal", "--to", "Sam", "--contact", "x", "--dir", dir)).rejects.toThrow(/can never be released/);
    await expect(run("release", "give", "legacy/../journal", "--to", "Sam", "--contact", "x", "--dir", dir)).rejects.toThrow(/outside the vault/);
    await expect(run("release", "give", "legacy/letters/ana.md", "--to", "Ana", "--contact", "x", "--dir", dir)).rejects.toThrow(/does not exist/);
    await run("release", "trustee", "Ana", "--contact", "ana@example.test", "--dir", dir);
    const one = (await run("release", "--dir", dir)).out;
    expect(one).toContain("note: Nothing can be released yet: the plan asks 2 confirmations of 1 trustee(s); name another, or set fewer.");
    await run("release", "trustee", "Ben", "--contact", "ben@example.test", "--dir", dir);
    const r = await run("release", "terms", "--wait", "14", "--dir", dir);
    expect(r.code).toBe(0);
    expect(r.out).toContain("a release needs 2 trustee confirmation(s), then 14 day(s) of waiting in which you or any trustee can stop it");
    // Sam gets the letter and the one entry marked to survive; the private entry is never in it.
    expect(r.out).toContain(`to Sam: legacy/letters/sam.md, journal/2026-11-10-${entry}.md`);
    expect(r.out.filter((l) => l.startsWith("note:"))).toEqual([]);
    await expect(run("release", "terms", "--confirmations", "0", "--dir", dir)).rejects.toThrow(/confirmations is a whole number/);
    await run("check", "--adopt", "--dir", dir);
    expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);
    // A hand edit that gives away the journal fails check.
    const plan = JSON.parse(await readFile(join(dir, "legacy", "release.json"), "utf8"));
    plan.gifts.push({ what: "journal", to: "Sam", contact: "x" });
    await writeFile(join(dir, "legacy", "release.json"), JSON.stringify(plan));
    expect((await run("check", "--dir", dir)).out).toContain("problem: legacy/release.json: journal can never be released: only the legacy folder and entries marked to survive you can");
  });

  it("upgrades a vault made before the lifetime folders", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    // Make it look like a 2026-11-10 vault: ventures/ and library/ instead of the new folders.
    const { mkdir, rm: rmf } = await import("node:fs/promises");
    for (const d of ["belonging", "household", "community", "learn", "work"]) await rmf(join(dir, d), { recursive: true });
    for (const v of ["household", "work", "community", "friendship"]) {
      await mkdir(join(dir, "ventures", v), { recursive: true });
      await writeFile(join(dir, "ventures", v, "README.md"), `# ${v}\n`);
    }
    await mkdir(join(dir, "library"));
    await writeFile(join(dir, "library", "meditations.md"), "# Meditations\n");
    await writeFile(join(dir, "ventures", "friendship", "sam.md"), "# Sam\n");
    await run("check", "--adopt", "--dir", dir);
    const r = (await run("upgrade", "--dir", dir)).out;
    expect(r).toContain("moved: belonging/friends/sam.md");
    expect(r).toContain("moved: learn/meditations.md");
    expect(await readFile(join(dir, "household", "README.md"), "utf8")).toBe("# household\n");
    expect((await readdir(dir)).includes("ventures")).toBe(false);
    expect((await readdir(dir)).includes("library")).toBe(false);
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
