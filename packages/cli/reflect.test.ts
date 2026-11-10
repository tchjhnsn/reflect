import { mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
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
    await expect(run("init", "--name", "Again", "--dir", dir)).rejects.toThrow(/already/);
    const id = (await run("write", "Walked to the river.", "--date", "2026-11-10", "--dir", dir)).out[0]!;
    await run("add", id, "It was cold.", "--dir", dir);
    await run("close", id, "--dir", dir);
    await expect(run("add", id, "late", "--dir", dir)).rejects.toThrow(/closed/);
    expect((await run("list", "--dir", dir)).out[0]).toContain("closed");
    expect(await run("check", "--dir", dir)).toEqual({ code: 0, out: ["ok"] });

    const [person] = await readdir(join(dir, "people"));
    await writeFile(join(dir, "people", person!, "entries", "2026-11-09-hand.md"), "By hand.\n");
    const before = await run("check", "--dir", dir);
    expect(before.out).toContain(`written outside reflect: people/${person}/entries/2026-11-09-hand.md`);
    await run("check", "--adopt", "--dir", dir);
    expect(await run("check", "--dir", dir)).toEqual({ code: 0, out: ["ok"] });
  });

  it("fails on a grant that names entries", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    const [person] = await readdir(join(dir, "people"));
    const { mkdir } = await import("node:fs/promises");
    await mkdir(join(dir, "people", person!, "grants"));
    await writeFile(join(dir, "people", person!, "grants", "mentura.json"), JSON.stringify({ to: "mentura", paths: ["intents", "entries"] }));
    const r = await run("check", "--dir", dir);
    expect(r.code).toBe(1);
    expect(r.out.join("\n")).toMatch(/can never name entries/);
  });

  it("refuses to run outside an instance", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await expect(run("list", "--dir", dir)).rejects.toThrow(/reflect init/);
  });
});
