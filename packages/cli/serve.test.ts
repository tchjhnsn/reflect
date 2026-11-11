import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { main } from "./reflect.ts";
import { allowed, handle, serve } from "./serve.ts";

const dirs: string[] = [];
afterAll(async () => Promise.all(dirs.map((d) => rm(d, { recursive: true, force: true }))));

async function run(...args: string[]): Promise<string[]> {
  const out: string[] = [];
  await main(args, (s) => out.push(s));
  return out;
}

const get = (root: string, path: string) => handle(root, "GET", new URL(path, "http://127.0.0.1:4500"), new URLSearchParams());
const post = (root: string, path: string, form: Record<string, string>) => handle(root, "POST", new URL(path, "http://127.0.0.1:4500"), new URLSearchParams(form));

describe("reflect serve", () => {
  it("shows the soul vault and changes it only through the same core code", async () => {
    const base = await mkdtemp(join(tmpdir(), "reflect-serve-"));
    dirs.push(base);
    const life = join(base, "life");
    const soul = join(base, "soul");
    await run("init", "--name", "Tester", "--dir", life);
    expect((await get(life, "/")).body).toContain("No soul vault yet");
    const a = (await run("write", "Walked to the bakery at dawn <again>.", "--date", "2026-11-09", "--dir", life))[0]!;
    const b = (await run("write", "Baked bread with Sam.", "--date", "2026-11-10", "--dir", life))[0]!;
    await run("soul", "init", soul, "--dir", life);
    for (const [id, chosen] of [[a, ["will", "desire", "embodiment"]], [b, ["will", "desire"]]] as const) {
      await writeFile(join(soul, "readings", `${id}.draft.json`), JSON.stringify({ entry: id, read_at: "2026-11-11T00:00:00.000Z", guess_at: 0.5, weights: {}, chosen, self_terms: [] }));
      await run("keep", id, "--dir", life);
    }
    await mkdir(join(life, "work", "bakery"), { recursive: true });

    const home = await get(life, "/");
    expect(home.status).toBe(200);
    expect(home.headers["content-security-policy"]).toContain("default-src 'none'");
    expect(home.body).toContain("2 moments, 2 with a confirmed reading");
    expect(home.body).toMatch(/Will<\/span>.*?2\/2/s);

    const moments = (await get(life, "/moments")).body;
    expect(moments).toContain("Walked to the bakery at dawn &lt;again&gt;.");
    expect(moments).not.toContain("<again>");

    // Bind from the moment's page; a bad path comes back as a message, nothing written.
    const bad = await post(life, "/bind", { moment: a, note: "work/nowhere", back: `/moments/${a}` });
    expect(bad.status).toBe(303);
    expect(decodeURIComponent(bad.headers.location!)).toContain("Not kept: no note or folder work/nowhere");
    const ok = await post(life, "/bind", { moment: a, note: "work/bakery", role: "place", back: `/moments/${a}` });
    expect(decodeURIComponent(ok.headers.location!)).toBe(`/moments/${a}?msg=Bound to work/bakery.`);
    await post(life, "/bind", { moment: b, note: "work/bakery", back: `/moments/${b}` });
    expect((await get(life, `/moments/${a}`)).body).toContain('href="/notes?note=work%2Fbakery">work/bakery</a> <span class="meta">(place)</span>');
    expect(await readFile(join(soul, "bindings", `${a}.json`), "utf8")).toContain('"role": "place"');

    // The note's profile is counted from what was confirmed.
    const notes = (await get(life, "/notes")).body;
    expect(notes).toContain("work/bakery</a> <span class=\"meta\">2 moments</span>");
    const profile = (await get(life, "/notes?note=work/bakery")).body;
    expect(profile).toContain("2 moments bound");
    expect(profile).toMatch(/Embodiment<\/span>.*?1\/2/s);

    // Unbind keeps the line and stops it counting.
    await post(life, "/unbind", { moment: b, note: "work/bakery", back: `/moments/${b}` });
    expect((await get(life, "/notes?note=work/bakery")).body).toContain("1 moment bound");

    // Holds: the person's words.
    const h = await post(life, "/hold", { words: "Work is a gift to my neighbors.", as: "belief", held: "duty", name: "work as gift" });
    expect(decodeURIComponent(h.headers.location!)).toBe("/holds?msg=Kept holds/work-as-gift.md.");
    expect((await get(soul, "/holds")).body).toContain("held as duty (Conscience)");
    const refused = await post(life, "/hold", { words: "Bread.", as: "hunch" });
    expect(decodeURIComponent(refused.headers.location!)).toContain("Not kept: --as is belief or value");

    // Readings of you: a grounded one can be confirmed; one whose moments no longer show it says why.
    await writeFile(join(soul, "of-me", "tensions-wanting.md"), `---\ntitle: "Wanting and willing"\ntype: tension\nelements: ["will", "desire"]\nreading: ai\n---\n\n# Wanting and willing\n\nThe subject walked at dawn [[moments/2026-11-09-${a}]].\n`);
    await writeFile(join(soul, "of-me", "patterns-body.md"), `---\ntitle: "Body"\ntype: pattern\nelements: ["embodiment"]\nreading: ai\n---\n\n# Body\n\nBread [[moments/2026-11-10-${b}]].\n`);
    const ofMe = (await get(life, "/of-me")).body;
    expect(ofMe).toContain("Will against Desire");
    expect(ofMe).toContain(`href="/moment-file/2026-11-09-${a}">2026-11-09</a>`);
    expect(ofMe).toContain(`Does not answer to the element table: moments/2026-11-10-${b} has no confirmed reading showing embodiment`);
    expect((await get(life, `/moment-file/2026-11-09-${a}`)).headers.location).toBe(`/moments/${a}`);
    await post(life, "/confirm", { path: "of-me/tensions-wanting" });
    expect((await get(life, "/of-me")).body).toMatch(/Confirmed \d{4}-\d{2}-\d{2}/);

    // A redirect back never leaves the server.
    expect((await post(life, "/bind", { moment: a, note: "work/bakery", back: "//evil.example" })).headers.location).toMatch(/^\/\?msg=/);
    expect((await get(life, "/nowhere")).status).toBe(404);
  });

  it("answers only requests to its loopback host, and forms posted from its own pages", async () => {
    expect(allowed("GET", "127.0.0.1:4500", undefined, 4500)).toBe(true);
    expect(allowed("GET", "localhost:4500", undefined, 4500)).toBe(true);
    expect(allowed("GET", "evil.example:4500", undefined, 4500)).toBe(false);
    expect(allowed("GET", undefined, undefined, 4500)).toBe(false);
    expect(allowed("POST", "127.0.0.1:4500", "http://127.0.0.1:4500", 4500)).toBe(true);
    expect(allowed("POST", "127.0.0.1:4500", "https://evil.example", 4500)).toBe(false);

    const base = await mkdtemp(join(tmpdir(), "reflect-serve-"));
    dirs.push(base);
    await run("init", "--name", "Tester", "--dir", base);
    const port = 40000 + Math.floor(Math.random() * 20000);
    const server = await serve(base, port);
    try {
      const r = await fetch(`http://127.0.0.1:${port}/`);
      expect(r.status).toBe(200);
      expect(await r.text()).toContain("No soul vault yet");
      const cross = await fetch(`http://127.0.0.1:${port}/hold`, { method: "POST", headers: { origin: "https://evil.example", "content-type": "application/x-www-form-urlencoded" }, body: "words=x&as=belief" });
      expect(cross.status).toBe(403);
    } finally {
      server.close();
    }
  });
});
