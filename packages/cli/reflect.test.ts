import { chmod, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
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
    expect((await readdir(dir)).sort()).toEqual([".github", ".gitignore", "belonging", "community", "concepts", "documents", "household", "ideas", "intents", "journal", "learn", "legacy", "life", "line", "memory.jsonl", "needs", "person.json", "polis", "reflect.json", "reflections", "telos", "work"]);
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

  it("models what you know from the books you read and your own grasp of their subjects", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    expect(await readFile(join(dir, "learn", "books", "README.md"), "utf8")).toContain("without looking");
    const kept = await run("learn", "book", "Thinking in Systems", "--isbn", "978-1-60358-055-7", "--by", "Donella Meadows", "--subjects", "systems thinking,feedback loops", "--status", "read", "--on", "2025-06-01", "--dir", dir);
    expect(kept.out[1]).toContain("name without looking");
    const book = await readFile(join(dir, "learn", "books", "thinking-in-systems.md"), "utf8");
    expect(book).toContain('authors: "Donella Meadows"\nisbn: 9781603580557\nstatus: read\nfinished: 2025-06-01\nsubjects: [[learn/subjects/systems-thinking]], [[learn/subjects/feedback-loops]]');
    await expect(run("learn", "book", "Bad", "--isbn", "123", "--dir", dir)).rejects.toThrow(/10 or 13 digits/);

    // Read but not rated is read, not known.
    let map = (await run("learn", "--dir", dir)).out;
    expect(map).toContain("  thinking-in-systems  read 2025-06-01; not yet rated: systems-thinking, feedback-loops");
    expect(map).toContain("  nothing rated yet: reflect learn grasp <subject> <level>");

    // The recall check: name what you remember first, then see the book's subjects.
    const recall = (await run("learn", "recall", "Thinking in Systems", "It was about stocks and flows and feedback, and systems thinking in general.", "--dir", dir)).out;
    expect(recall).toContain("You named: systems-thinking");
    expect(recall).toContain("Its other subjects: feedback-loops");
    expect(await readFile(join(dir, "learn", "books", "thinking-in-systems.md"), "utf8")).toMatch(/- \S+ recall: It was about stocks and flows/);
    await expect(run("learn", "recall", "Nope", "x", "--dir", dir)).rejects.toThrow(/no book Nope/);

    await expect(run("learn", "grasp", "systems thinking", "knows", "--dir", dir)).rejects.toThrow(/grasp is one of/);
    await expect(run("learn", "grasp", "systems thinking", "teach", "--book", "Nope", "--dir", dir)).rejects.toThrow(/no book Nope/);
    await run("learn", "grasp", "systems thinking", "explain", "--book", "Thinking in Systems", "--dir", dir);
    await run("learn", "grasp", "feedback loops", "forgot", "--book", "Thinking in Systems", "--note", "only the bathtub", "--dir", dir);
    map = (await run("learn", "--dir", dir)).out;
    expect(map.find((l) => l.startsWith("  systems-thinking"))).toMatch(/could explain the key ideas \(\S+\), from thinking-in-systems$/);
    expect(map).not.toContain(expect.stringMatching(/^ {2}feedback-loops {2}don't/));
    expect(map).toContain("  thinking-in-systems  read 2025-06-01");

    // Grasp changes over time; the latest by date counts.
    await run("learn", "grasp", "feedback loops", "gist", "--dir", dir);
    map = (await run("learn", "--dir", dir)).out;
    expect(map.find((l) => l.startsWith("  feedback-loops"))).toMatch(/remember the gist/);

    await run("learn", "aspire", "Control theory", "--requires", "feedback loops,linear algebra", "--same-as", "wikidata:Q842015", "--dir", dir);
    await expect(run("learn", "need", "Accounting", "--dir", dir)).rejects.toThrow(/names what it is for/);
    await run("learn", "need", "Accounting", "--for", "intents/start-a-company", "--dir", dir);
    await run("connect", "fanout", "--name", "Fanout", "--kind", "platform", "--dir", dir);
    await expect(run("learn", "resource", "Some book", "--kind", "book", "--dir", dir)).rejects.toThrow(/reflect learn book/);
    await run("learn", "resource", "Control bootcamp", "--kind", "video", "--url", "https://www.youtube.com/@Eigensteve", "--subjects", "control theory", "--dir", dir);
    map = (await run("learn", "--dir", dir)).out;
    expect(map).toContain("  control-theory  first learn linear-algebra; from control-bootcamp (saved)");
    expect(map).toContain("  accounting  for intents/start-a-company; ready");
    expect(map).toContain("  linear-algebra  a first step toward control-theory");

    await run("check", "--adopt", "--dir", dir);
    expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);
    await writeFile(join(dir, "learn", "subjects", "linear-algebra.md"), "---\nsubject: linear-algebra\nrequires: [[learn/subjects/control-theory]]\n---\n\n- 2026-01-01 grasp teach [[learn/books/missing]]\n");
    const problems = (await run("check", "--dir", dir)).out.join("\n");
    expect(problems).toContain("prerequisites go round in a circle");
    expect(problems).toContain("names a book that is not in learn/books: missing");
  });

  it("looks a book up in Open Library and keeps only what the person says", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    const asked: string[] = [];
    const doc = { key: "/works/OL3737036W", title: "Thinking in systems", author_name: ["Donella H. Meadows", "Diana Wright"], first_publish_year: 2008, isbn: ["1603580557", "9781603580557"], subject: ["systems thinking", "Accessible book", "System analysis", "systems thinking", "Decision making"] };
    const fetcher = async (url: string) => {
      asked.push(url);
      const body = url.includes("/search.json") ? { docs: url.includes("isbn:0000000000") ? [] : [doc] } : { title: "Thinking in systems : a primer", table_of_contents: [{ title: "System structure and behavior", level: 0 }, { title: "The basics", level: 1 }] };
      return { ok: true, status: 200, json: async () => body };
    };
    const out: string[] = [];
    await main(["learn", "find", "thinking", "in", "systems", "--dir", dir], (x) => out.push(x), fetcher);
    expect(out[0]).toBe("Thinking in systems, Donella H. Meadows, Diana Wright (2008)  isbn 9781603580557");
    expect(asked[0]).toContain("q=thinking%20in%20systems");

    out.length = 0;
    await main(["learn", "book", "Thinking in Systems", "--isbn", "9781603580557", "--lookup", "--dir", dir], (x) => out.push(x), fetcher);
    expect(out).not.toContain(expect.stringMatching(/calls it/));
    expect(out).toContain("Open Library lists these subjects: systems thinking; System analysis; Decision making. Keep the ones that fit: --subjects a,b");
    expect(out).toContain("Contents:\n  System structure and behavior\n    The basics");
    const note = await readFile(join(dir, "learn", "books", "thinking-in-systems.md"), "utf8");
    expect(note).toContain('authors: "Donella H. Meadows; Diana Wright"\nisbn: 9781603580557\nopenlibrary: OL3737036W\nstatus: want');
    expect(note).not.toContain("subjects:");
    const saved = JSON.parse(await readFile(join(dir, "learn", "books", "thinking-in-systems.openlibrary.json"), "utf8"));
    expect(saved.source).toBe("https://openlibrary.org/works/OL3737036W");
    expect(saved.contents).toEqual(["System structure and behavior", "  The basics"]);
    await expect(main(["learn", "book", "Ghost", "--isbn", "0000000000", "--lookup", "--dir", dir], () => undefined, fetcher)).rejects.toThrow(/no book with isbn/);
    await expect(run("learn", "book", "Ghost", "--lookup", "--dir", dir)).rejects.toThrow(/needs --isbn/);
    expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);
  });

  it("drafts a transcript's courses, keeps only rows really in it, and keeps the ones the person confirms", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    const file = join(dir, "..", `transcript-${Date.now()}.txt`);
    await writeFile(file, "STATE UNIVERSITY\nFall 2016\nMATH 221  Linear Algebra   3.0  B-\nPHIL 101  Intro to Ethics  3.0  A\nTerm GPA 3.35\n");
    const chat = async () => ({ institution: "State University", courses: [
      { term: "Fall 2016", code: "MATH 221", title: "Linear Algebra", credits: 3, grade: "B-" },
      { term: "Fall 2016", code: "PHIL 101", title: "Intro to Ethics", credits: 3, grade: "A" },
      { term: "Fall 2016", code: "CS 50", title: "Computer Science", credits: 4, grade: "A" },
    ] });
    const out: string[] = [];
    await main(["learn", "transcript", file, "--dir", dir], (x) => out.push(x), undefined, chat);
    await rm(file);
    expect(out).toContain("1. Fall 2016  MATH 221 Linear Algebra, 3 credits, B-");
    expect(out).toContain("Left out, because they are not in the transcript as written: Computer Science");
    await run("learn", "transcript", "--keep", "1", "--dir", dir);
    const course = await readFile(join(dir, "learn", "courses", "fall-2016-math-221.md"), "utf8");
    expect(course).toContain('grade: "B-"\ninstitution: "State University"\nsubject: "[[learn/subjects/linear-algebra]]"');
    let map = (await run("learn", "--dir", dir)).out;
    expect(map).toContain("  Fall 2016  MATH 221 Linear Algebra  grade B-; not yet rated by you");
    await run("learn", "grasp", "Linear Algebra", "explain", "--note", "better than the grade says", "--dir", dir);
    map = (await run("learn", "--dir", dir)).out;
    expect(map).toContain("  Fall 2016  MATH 221 Linear Algebra  grade B-; you say: could explain the key ideas");
    await expect(run("learn", "transcript", "--keep", "9", "--dir", dir)).rejects.toThrow(/no row 9/);
    await run("check", "--adopt", "--dir", dir);
    expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);
  });

  it("keeps credentials, checks an Open Badges link, and reads a LinkedIn export only when asked to keep it", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    const fetcher = async (url: string) => ({ ok: true, status: 200, json: async () => (url.endsWith("/badge") ? { name: "AWS Certified Cloud Practitioner", issuer: { name: "Amazon Web Services" } } : { "@context": "https://w3id.org/openbadges/v2", type: "Assertion", badge: "https://badges.example/badge", issuedOn: "2024-05-02T00:00:00Z" }) });
    const out: string[] = [];
    await main(["learn", "credential", "Cloud Practitioner", "--url", "https://badges.example/a/1", "--check", "--subjects", "cloud computing", "--dir", dir], (x) => out.push(x), fetcher);
    expect(out).toContain("The issuer's badge says: AWS Certified Cloud Practitioner, from Amazon Web Services, 2024-05-02");
    const cred = await readFile(join(dir, "learn", "credentials", "amazon-web-services-cloud-practitioner.md"), "utf8");
    expect(cred).toContain('issuer: "Amazon Web Services"\nissued: 2024-05-02\nurl: https://badges.example/a/1\nsubjects: [[learn/subjects/cloud-computing]]');
    await expect(run("learn", "credential", "PMP", "--dir", dir)).rejects.toThrow(/--issuer/);

    const exp = join(dir, "..", `li-${Date.now()}`);
    await mkdir(exp);
    await writeFile(join(exp, "Endorsement_Received_Info.csv"), "Endorsement Date,Skill Name,Endorser First Name,Endorser Last Name,Endorser Public Url,Endorsement Status\n2024/03/01 10:00:00 UTC,Systems Thinking,Ana,Lee,https://www.linkedin.com/in/ana,ACCEPTED\n2024/04/01 10:00:00 UTC,Systems Thinking,Ben,Ode,,ACCEPTED\n2024/04/02 10:00:00 UTC,Golf,Cy,Ray,,REJECTED\n");
    await writeFile(join(exp, "Recommendations_Received.csv"), 'First Name,Last Name,Company,Job Title,Text,Creation Date,Status\nAna,Lee,Acme,CTO,"Torian sees the whole system, ""end to end"".\nA rare skill.",03/05/24,VISIBLE\n');
    await writeFile(join(exp, "Certifications.csv"), "Name,Url,Authority,Started On,Finished On,License Number\nPMP,,PMI,Mar 2021,,\n");
    await writeFile(join(exp, "Skills.csv"), "Name\nSystems Thinking\nNegotiation\n");
    const preview = (await run("learn", "linkedin", exp, "--dir", dir)).out;
    expect(preview).toContain("Endorsements: 2, on 1 skill(s)");
    expect(preview).toContain("  Systems Thinking: Ana Lee, Ben Ode");
    expect(preview).toContain("Recommendations: 1 (from Ana Lee)");
    await expect(readFile(join(dir, "learn", "subjects", "systems-thinking.md"), "utf8")).rejects.toThrow();
    expect((await run("learn", "linkedin", exp, "--keep", "all", "--dir", dir)).out[0]).toBe("kept: 2 new vouch(es), 1 recommendation(s), 1 certification(s), 2 skill(s) as subjects for you to rate");
    expect((await run("learn", "linkedin", exp, "--keep", "endorsements", "--dir", dir)).out[0]).toBe("kept: 0 new vouch(es)");
    await rm(exp, { recursive: true });
    expect(await readFile(join(dir, "learn", "subjects", "systems-thinking.md"), "utf8")).toContain("- 2024-03-01 vouched by Ana Lee <https://www.linkedin.com/in/ana> on linkedin\n- 2024-04-01 vouched by Ben Ode on linkedin");
    expect(await readFile(join(dir, "learn", "vouches", "ana-lee-2024-03-05.md"), "utf8")).toContain('Torian sees the whole system, "end to end".\nA rare skill.');
    expect(await readFile(join(dir, "learn", "credentials", "pmi-pmp.md"), "utf8")).toContain("issued: 2021-03");
    const map = (await run("learn", "--dir", dir)).out;
    expect(map).toContain("  systems-thinking  vouched for by 2");
    await run("learn", "grasp", "Systems Thinking", "teach", "--dir", dir);
    expect((await run("learn", "--dir", dir)).out.find((l) => l.startsWith("  systems-thinking"))).toMatch(/could teach it .*; vouched for by 2$/);
    await run("check", "--adopt", "--dir", dir);
    expect((await run("check", "--dir", dir)).out).toEqual(["ok"]);
  });

  it("keeps films, talks, conversations and posts as inputs, and ideas as outputs that link them", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    await run("learn", "book", "Politics", "--by", "Aristotle", "--status", "read", "--subjects", "political philosophy", "--dir", dir);
    await run("learn", "film", "Metropolis", "--year", "1927", "--status", "watched", "--subjects", "urban design", "--dir", dir);
    await expect(run("learn", "talk", "The Death and Life of Cities", "--dir", dir)).rejects.toThrow(/speaker/);
    await run("learn", "talk", "Strong Towns", "--speaker", "Chuck Marohn", "--on", "2025-04", "--dir", dir);
    await run("learn", "conversation", "whether a city can be corrupt", "--with", "Dad", "--on", "2026-10-01", "--dir", dir);
    await run("learn", "resource", "A thread on zoning", "--kind", "thread", "--url", "https://www.reddit.com/r/urbanplanning/comments/abc", "--dir", dir);
    expect(await readFile(join(dir, "learn/resources/a-thread-on-zoning.md"), "utf8")).toContain("source: reddit");

    await expect(run("idea", "Corrupt cities", "--dir", dir)).rejects.toThrow(/statement/);
    await run("idea", "Corrupt and correct cities", "A city is built in a correct form when it serves the common good, and a corrupt one when it serves its builders.", "--from", "Politics", "--how", "the test of Politics III", "--dir", dir);
    await run("idea", "Corrupt and correct cities", "--maybe", "Metropolis", "--maybe", "a sermon on Micah 6", "--dir", dir);
    await run("idea", "Corrupt and correct cities", "--related", "challenges:The Death and Life of Great American Cities", "--by", "Claude", "--note", "order that grows from below", "--dir", dir);
    await expect(run("idea", "Corrupt and correct cities", "--related", "agrees:x", "--by", "Claude", "--dir", dir)).rejects.toThrow(/stance/);
    await run("idea", "Corrupt and correct cities", "--from", "Metropolis", "--dir", dir);
    const note = await readFile(join(dir, "ideas/corrupt-and-correct-cities.md"), "utf8");
    expect(note).toContain("from: [[learn/books/politics]], [[learn/films/metropolis]]");
    expect(note).toContain('maybe from "a sermon on Micah 6"');
    expect(note).toContain('related challenges by Claude "The Death and Life of Great American Cities": order that grows from below');
    const list = (await run("idea", "--dir", dir)).out.join("\n");
    expect(list).toContain("from [[learn/films/metropolis]]");
    expect(list).not.toContain("maybe from [[learn/films/metropolis]]");
    expect(list).toContain("challenges: \"The Death and Life of Great American Cities\", suggested by Claude");
    await run("idea", "Corrupt and correct cities", "--drop", "a sermon on Micah 6", "--dir", dir);
    expect((await run("idea", "--dir", dir)).out.join("\n")).not.toContain("Micah");
    await expect(run("idea", "Other", "x", "--from", "learn/books/missing", "--dir", dir)).resolves.toBeTruthy();
    expect(await run("check", "--dir", dir)).toEqual({ code: 0, out: ["ok"] });
    const map = (await run("learn", "--dir", dir)).out.join("\n");
    expect(map).toContain("Films\n  metropolis  film, watched");
    expect(map).toContain("Talks\n  strong-towns  Chuck Marohn, 2025-04");
    expect(map).toContain("with Dad: whether a city can be corrupt");
  });

  it("keeps an AI-written wiki apart from the record: cited, quotes checked, readings until confirmed", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    const a = (await run("write", "Baked bread with Sam again. I keep building things alone and then asking why nobody joins.", "--date", "2026-11-09", "--dir", dir)).out[0]!;
    const b = (await run("write", "Sam said the bakery could open at five if we split the mornings.", "--date", "2026-11-10", "--dir", dir)).out[0]!;
    const ea = `journal/2026-11-09-${a}`;
    const eb = `journal/2026-11-10-${b}`;
    expect((await run("wiki", "--dir", dir)).out[0]).toMatch(/The wiki is off/);
    await expect(run("wiki", "prompt", "--dir", dir)).rejects.toThrow(/wiki is off/);
    expect((await run("wiki", "on", "--dir", dir)).out[0]).toMatch(/the wiki is on/);
    const prompt = (await run("wiki", "prompt", "--dir", dir)).out.join("\n");
    expect(prompt).toContain(`### ${ea} (2026-11-09)`);
    expect(prompt).toContain("I keep building things alone");
    const reply = {
      kind: "reflect.wiki",
      version: 1,
      absorbed: [ea, eb, "journal/not-a-real-entry"],
      articles: [
        { path: "people/sam", title: "Sam", type: "person", also: ["Samuel"], related: ["wiki/patterns/building-alone"], body: `Sam bakes with the subject and proposed splitting the mornings so the bakery could open at five [[${eb}]]. They baked bread together more than once [[${ea}]].\n\nSee [[wiki/patterns/building-alone]].` },
        { path: "patterns/building-alone", title: "Building alone", type: "pattern", body: `The subject described a cycle: "I keep building things alone" and then wondering why nobody joins [[${ea}]]. Sam's offer to split the mornings is the first counterexample in the entries [[${eb}]].` },
        { path: "places/corner", title: "The corner", type: "place", body: "The bakery would sit on the corner. Nobody cited this." },
        { path: "ideas/bakery", title: "Bakery", type: "idea", body: `The subject wrote "the bakery will make millions by spring" [[${eb}]].` },
        { path: "eras/2026", title: "2026", type: "era", body: `A year of bread \u2014 and of Sam [[${ea}]].` },
      ],
    };
    await writeFile(join(dir, "wiki-reply.md"), "```json\n" + JSON.stringify(reply) + "\n```");
    const draft = (await run("wiki", "absorb", join(dir, "wiki-reply.md"), "--dir", dir)).out;
    expect(draft[0]).toBe("2 article(s) to keep, 2 entries absorbed:");
    expect(draft.join("\n")).toContain("refused: places/corner: a passage cites no note in the record");
    expect(draft.join("\n")).toContain("refused: ideas/bakery: a quote is not in the notes its passage cites");
    expect(draft.join("\n")).toContain("refused: eras/2026: uses an em dash");
    const kept = (await run("wiki", "absorb", "--keep", "all", "--dir", dir)).out;
    expect(kept).toEqual(["kept: wiki/people/sam", "kept: wiki/patterns/building-alone (reading: ai)"]);
    const sam = await readFile(join(dir, "wiki/people/sam.md"), "utf8");
    expect(sam).toContain(`sources: ["${eb}", "${ea}"]`);
    expect(sam).not.toContain("reading:");
    expect(await readFile(join(dir, "wiki/patterns/building-alone.md"), "utf8")).toContain("reading: ai");
    const index = await readFile(join(dir, "wiki/_index.md"), "utf8");
    expect(index).toContain("- [[wiki/people/sam]] Sam (also: Samuel)");
    expect(index).toContain("- [[wiki/patterns/building-alone]] Building alone; the AI's reading, not confirmed; linked from 1");
    expect(JSON.parse(await readFile(join(dir, "wiki/_backlinks.json"), "utf8"))).toEqual({ "patterns/building-alone": ["people/sam"] });
    await expect(run("wiki", "confirm", "people/sam", "--dir", dir)).rejects.toThrow(/not a reading of you/);
    await run("wiki", "confirm", "patterns/building-alone", "--dir", dir);
    expect(await readFile(join(dir, "wiki/patterns/building-alone.md"), "utf8")).toMatch(/reading: confirmed \d{4}-\d{2}-\d{2}/);
    expect((await run("wiki", "--dir", dir)).out).toEqual(["2 article(s): 1 pattern, 1 person", "1 reading(s) of you, 1 confirmed", "2 entries absorbed, 0 waiting"]);
    expect((await run("wiki", "prompt", "--dir", dir)).out.join("\n")).toContain("(none: every entry is absorbed)");
    // A rewrite of a confirmed reading makes it the AI's reading again.
    const again = { kind: "reflect.wiki", version: 1, absorbed: [], articles: [{ ...reply.articles[1], body: `${reply.articles[1]!.body} The pattern held for a year [[${ea}]].` }] };
    await writeFile(join(dir, "wiki-reply.md"), JSON.stringify(again));
    await run("wiki", "absorb", join(dir, "wiki-reply.md"), "--dir", dir);
    expect((await run("wiki", "absorb", "--keep", "1", "--dir", dir)).out).toEqual(["kept: wiki/patterns/building-alone (reading: ai)"]);
    await run("index", "--dir", dir);
    expect(await readFile(join(dir, "index.md"), "utf8")).toContain("[[wiki/_index]]");
    expect(await run("check", "--dir", dir)).toEqual({ code: 0, out: ["ok"] });
  });

  it("moves the journal into a soul vault of its own, linked one way from the life vault", async () => {
    const base = await mkdtemp(join(tmpdir(), "reflect-two-"));
    const notes = await mkdtemp(join(tmpdir(), "reflect-notes-"));
    dirs.push(base, notes);
    const life = join(base, "life");
    const soul = join(base, "soul");
    await run("init", "--name", "Tester", "--dir", life);
    const first = (await run("write", "Walked to the bakery at dawn.", "--date", "2026-11-09", "--dir", life)).out[0]!;
    const kept = (await run("legacy", "hopes", "That you stay brave.", "--date", "2026-11-10", "--dir", life)).out[0]!;
    await writeFile(join(notes, "2024-03-01.md"), "An old page about the river.\n");
    await run("source", "add", "notes", notes, "--dir", life);
    await writeFile(join(life, "reply.md"), JSON.stringify({ kind: "reflect.placements", version: 1, source: "notes", placements: [{ path: "2024-03-01.md", kind: "entry", date: "2024-03-01" }] }));
    await run("source", "tag", "notes", join(life, "reply.md"), "--dir", life);
    await run("source", "tag", "notes", "--keep", "1", "--dir", life);
    expect((await run("soul", "--dir", life)).out[0]).toMatch(/No soul vault yet/);

    await expect(run("soul", "init", join(life, "inner"), "--dir", life)).rejects.toThrow(/folder of its own/);
    const init = (await run("soul", "init", soul, "--dir", life)).out;
    expect(init[0]).toBe(`started the soul vault at ${soul}; moved 3 file(s) of the journal into it`);
    await expect(run("soul", "init", join(base, "other"), "--dir", life)).rejects.toThrow(/already has a soul vault/);
    expect(await readdir(join(life, "journal"))).toEqual(["README.md"]);
    expect(await readFile(join(life, "journal", "README.md"), "utf8")).toContain("soul/moments");
    expect(JSON.parse(await readFile(join(life, "reflect.json"), "utf8")).soul).toBe("../soul");
    expect(JSON.parse(await readFile(join(soul, "reflect.json"), "utf8"))).toMatchObject({ layout: "soul", life: "../life" });
    expect(await readFile(join(soul, "moments", `2026-11-09-${first}.md`), "utf8")).toContain("Walked to the bakery");

    // The journal works from either folder, and lands in the soul vault; a pointer entry still reads its file.
    const fromLife = (await run("list", "--dir", life)).out.join("\n");
    expect(fromLife).toContain("Walked to the bakery");
    expect(fromLife).toContain("An old page about the river.");
    const next = (await run("write", "Bread again.", "--date", "2026-11-11", "--dir", life)).out[0]!;
    expect(await readFile(join(soul, "moments", `2026-11-11-${next}.md`), "utf8")).toContain("Bread again.");
    expect((await run("list", "--dir", soul)).out.join("\n")).toContain("Bread again.");
    // A spark is a moment; its noun is a concept, a thing in shared reality, so its note goes to the life vault.
    await run("spark", "river", "The river was high.", "--sentence", "--dir", soul);
    expect(await readFile(join(life, "concepts", "river.md"), "utf8")).toContain("river");
    await expect(readdir(join(soul, "concepts"))).rejects.toThrow();
    expect((await run("soul", "--dir", life)).out).toEqual([`soul vault: ${soul}`, "5 moment(s), 0 reading file(s), 0 binding file(s), 0 hold(s), 0 reading(s) of you"]);

    // The life vault's index does not list moments; its release hands on only the moment marked to survive.
    await run("index", "--dir", life);
    const index = await readFile(join(life, "index.md"), "utf8");
    expect(index).toContain("kept in my soul vault (5 moments), not listed here");
    expect(index).not.toContain("bakery at dawn");
    await run("release", "give", "legacy-entries", "--to", "Sam", "--contact", "sam@example.test", "--dir", life);
    expect((await run("release", "--dir", life)).out).toContain(`to Sam: soul:moments/2026-11-10-${kept}.md`);

    // The wiki reads moments and cites them by their path in the soul vault.
    await run("wiki", "on", "--dir", life);
    expect((await run("wiki", "prompt", "--dir", life)).out.join("\n")).toContain(`### moments/2026-11-09-${first} (2026-11-09)`);
    await writeFile(join(life, "w.json"), JSON.stringify({ kind: "reflect.wiki", version: 1, absorbed: [`moments/2026-11-09-${first}`], articles: [{ path: "places/bakery", title: "The bakery", type: "place", body: `The subject walked to the bakery at dawn [[moments/2026-11-09-${first}]].` }] }));
    expect((await run("wiki", "absorb", join(life, "w.json"), "--dir", life)).out[0]).toBe("1 article(s) to keep, 1 entry absorbed:");
    await run("wiki", "absorb", "--keep", "all", "--dir", life);
    expect(await run("check", "--dir", life)).toEqual({ code: 0, out: ["ok"] });
    expect(await run("check", "--dir", soul)).toEqual({ code: 0, out: ["ok"] });
  });

  it("writes index.md, the catalog your own AI starts from, without journal text", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    await run("learn", "book", "Politics", "--by", "Aristotle", "--status", "read", "--dir", dir);
    await run("learn", "film", "Metropolis", "--status", "watched", "--dir", dir);
    await run("learn", "aspire", "Urban design", "--dir", dir);
    await run("idea", "Corner bakery", "A bakery on my corner that opens at five.", "--from", "Politics", "--dir", dir);
    await run("write", "A private thought about Sam.", "--date", "2026-11-10", "--dir", dir);
    await mkdir(join(dir, "work", "acme"), { recursive: true });
    expect((await run("index", "--dir", dir)).out).toEqual(["wrote index.md"]);
    const index = await readFile(join(dir, "index.md"), "utf8");
    expect(index).toContain("- [[ideas/corner-bakery]] Corner bakery: A bakery on my corner that opens at five. (from [[learn/books/politics]])");
    expect(index).toContain("- [[learn/books/politics]] Politics, Aristotle: read");
    expect(index).toContain("- [[learn/films/metropolis]] Metropolis: watched");
    expect(index).toContain("- [[learn/subjects/urban-design]] Urban design: aspire");
    expect(index).toContain("- [[work/acme]]");
    expect(index).toMatch(/- \[\[journal\/2026-11-10-[^\]]+\]\] 2026-11-10/);
    expect(index).not.toContain("Sam");
    expect((await run("index", "--dir", dir)).out).toEqual(["index.md is up to date"]);
    expect(await run("check", "--dir", dir)).toEqual({ code: 0, out: ["ok"] });
  });

  it("sends an idea to Naive only when told to, and links the two both ways", async () => {
    const ws = await mkdtemp(join(tmpdir(), "reflect-ws-"));
    const bin = await mkdtemp(join(tmpdir(), "reflect-bin-"));
    dirs.push(ws, bin);
    const dir = join(ws, "vault");
    await run("init", "--name", "Tester", "--dir", dir);
    await run("learn", "book", "Politics", "--dir", dir);
    await run("idea", "Corner bakery", "A bakery on my corner that opens at five.", "--from", "Politics", "--how", "the household provides", "--dir", dir);
    await run("idea", "Corner bakery", "--maybe", "a friend's hunch", "--dir", dir);
    await expect(run("idea", "Corner bakery", "--send", "naive", "--dir", dir)).rejects.toThrow(/no Naive workspace/);
    await writeFile(join(ws, "naive-workspace.json"), "{}\n");
    // A stand-in for naive capture: keeps the text in raw/ and prints what Naive prints.
    const fake = join(bin, "naive");
    await writeFile(fake, `#!/usr/bin/env node\nconst fs = require("fs");\nfs.mkdirSync("raw", { recursive: true });\nfs.writeFileSync("raw/20261111-0530-ab12.md", process.argv[3]);\nconsole.log("kept raw/20261111-0530-ab12.md");\n`);
    await chmod(fake, 0o755);
    const before = process.env.NAIVE_BIN;
    process.env.NAIVE_BIN = fake;
    try {
      const preview = (await run("idea", "Corner bakery", "--send", "naive", "--dir", dir)).out.join("\n");
      expect(preview).toContain("A bakery on my corner that opens at five.");
      expect(preview).toContain("- [[learn/books/politics]]: the household provides");
      expect(preview).not.toContain("hunch");
      expect(preview).toContain("--send naive --yes");
      await expect(readdir(join(ws, "raw"))).rejects.toThrow();
      const sent = (await run("idea", "Corner bakery", "--send", "naive", "--yes", "--dir", dir)).out;
      expect(sent).toEqual([`sent: ${ws.split("/").pop()}:raw/20261111-0530-ab12.md`]);
      expect(await readFile(join(ws, "raw/20261111-0530-ab12.md"), "utf8")).toContain("Sent from Reflect: ideas/corner-bakery.md");
      expect(await readFile(join(dir, "ideas/corner-bakery.md"), "utf8")).toMatch(/- \d{4}-\d{2}-\d{2} sent to naive \S+:raw\/20261111-0530-ab12\.md\n$/);
      expect((await run("idea", "Corner bakery", "--send", "naive", "--dir", dir)).out[0]).toMatch(/^already sent/);
      await run("idea", "Corner bakery", "--from", "a walk downtown", "--dir", dir);
      expect(await readFile(join(dir, "ideas/corner-bakery.md"), "utf8")).toContain("sent to naive");
      expect(await run("check", "--dir", dir)).toEqual({ code: 0, out: ["ok"] });
    } finally {
      if (before === undefined) delete process.env.NAIVE_BIN;
      else process.env.NAIVE_BIN = before;
    }
  });

  it("reads a memory sweep from your AI and keeps only what you choose, inputs apart from ideas", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    dirs.push(dir);
    await run("init", "--name", "Tester", "--dir", dir);
    expect((await run("learn", "sweep", "--dir", dir)).out[0]).toContain("reflect.learn-source");
    const reply = {
      kind: "reflect.learn-source",
      version: 1,
      reported_by: { assistant: "Claude", basis: "saved memory", reported_at: "2026-11-11" },
      books: [{ title: "Politics", authors: ["Aristotle"], engaged: "referenced", subjects: ["political philosophy"], basis: "i-said", evidence: "you cite Book III often", confidence: "high" }],
      films: [{ title: "Metropolis", year: 1927, kind: "film", engaged: "finished", subjects: [], basis: "you-inferred", evidence: "you described its city", confidence: "medium" }],
      links: [{ title: "Zoning thread", url: "https://x.com/someone/status/1", kind: "post", subjects: [], basis: "i-said", evidence: "you shared it", confidence: "high" }],
      subjects: { shown: [{ name: "Political philosophy", how: "explained", evidence: "you explained Aristotle's six constitutions", confidence: "high" }], suggested: [{ name: "Urban design", because: "your interest in how cities are built", confidence: "medium" }] },
      ideas: [{ title: "Corrupt and correct cities", statement: "Aristotle's correct and corrupt forms, applied to urban development.", inputs: [{ title: "Politics", how: "the correct and corrupt test" }, { title: "a talk I half remember" }], basis: "you-inferred", evidence: "you framed it while studying", confidence: "medium" }],
    };
    await writeFile(join(dir, "reply.md"), `Here it is.\n\n\`\`\`json\n${JSON.stringify(reply)}\n\`\`\`\n\nLeast sure of the film.`);
    const draft = (await run("learn", "sweep", join(dir, "reply.md"), "--dir", dir)).out;
    expect(draft[0]).toBe("Claude (2026-11-11) found 6 item(s):");
    expect(draft[1]).toContain("1. book: Politics by Aristotle (referenced); you said, high");
    await expect(run("learn", "sweep", "--keep", "9", "--dir", dir)).rejects.toThrow(/no item 9/);
    await run("learn", "sweep", "--keep", "6,1,4,5", "--dir", dir);
    expect(await readFile(join(dir, "learn/books/politics.md"), "utf8")).toContain("status: referenced");
    const subject = await readFile(join(dir, "learn/subjects/political-philosophy.md"), "utf8");
    expect(subject).toContain("shown by Claude: explained: you explained Aristotle's six constitutions");
    expect(subject).not.toContain("grasp");
    const idea = await readFile(join(dir, "ideas/corrupt-and-correct-cities.md"), "utf8");
    expect(idea).toContain("maybe from [[learn/books/politics]]: the correct and corrupt test");
    expect(idea).toContain('maybe from "a talk I half remember"');
    await expect(readFile(join(dir, "learn/films/metropolis.md"), "utf8")).rejects.toThrow();
    const map = (await run("learn", "--dir", dir)).out.join("\n");
    expect(map).toContain("Shown, not yet rated by you\n  political-philosophy");
    expect(map).toContain("Suggested, not known\n  urban-design");
    expect(map).not.toMatch(/Known\n  political-philosophy/);
    expect(await run("check", "--dir", dir)).toEqual({ code: 0, out: ["ok"] });
  });

  it("points at a folder outside the vault, keeps the placements the person chose, and notices drift", async () => {
    const dir = await mkdtemp(join(tmpdir(), "reflect-cli-"));
    const notes = await mkdtemp(join(tmpdir(), "reflect-notes-"));
    dirs.push(dir, notes);
    await run("init", "--name", "Tester", "--dir", dir);
    await mkdir(join(notes, "Reading"), { recursive: true });
    await mkdir(join(notes, ".git"), { recursive: true });
    await mkdir(join(notes, "Private"), { recursive: true });
    await writeFile(join(notes, "Reading", "Politics.md"), "# Politics, Aristotle\n\nBook III: correct and corrupt constitutions.\n");
    await writeFile(join(notes, "cities.md"), "A city is correct when it serves the common good.\n");
    await writeFile(join(notes, "2024-03-01.md"), "Walked downtown today.\n");
    await writeFile(join(notes, "diploma.pdf"), "%PDF-1.4 fake");
    await writeFile(join(notes, "plan.md"), "A bakery on my corner that opens at five.\n");
    await writeFile(join(notes, "acme.md"), "Acme roadmap\n");
    await writeFile(join(notes, "pitch.md"), "something\n");
    await writeFile(join(notes, ".env"), "SHOULD_NOT_BE_SEEN=1\n");
    await writeFile(join(notes, "api_token.txt"), "nope\n");
    await writeFile(join(notes, "Private", "diary.md"), "private\n");
    await writeFile(join(notes, ".git", "config"), "x");

    await expect(run("source", "add", "Notes", notes, "--dir", dir)).rejects.toThrow(/lowercase/);
    await expect(run("source", "add", "inside", join(dir, "learn"), "--dir", dir)).rejects.toThrow(/inside the vault/);
    const added = (await run("source", "add", "notes", notes, "--exclude", "Private", "--dir", dir)).out;
    expect(added[0]).toContain("7 file(s) indexed");
    const prompt = (await run("source", "prompt", "notes", "--dir", dir)).out.join("\n");
    expect(prompt).toContain('"source": "notes"');
    expect(prompt).toContain("- Reading/Politics.md");
    expect(prompt).not.toMatch(/\.env|api_token|diary|\.git/);

    const reply = {
      kind: "reflect.placements",
      version: 1,
      source: "notes",
      placements: [
        { path: "Reading/Politics.md", kind: "book", name: "Politics", authors: ["Aristotle"], why: "notes on Book III", confidence: "high" },
        { path: "cities.md", kind: "idea", name: "Corrupt and correct cities", statement: "A city is correct when it serves the common good.", confidence: "medium" },
        { path: "2024-03-01.md", kind: "entry", date: "2024-03-01", confidence: "high" },
        { path: "diploma.pdf", kind: "document", name: "diploma", confidence: "medium" },
        { path: "plan.md", kind: "venture", name: "Corner Bakery", statement: "A bakery on my corner that opens at five.", confidence: "high" },
        { path: "acme.md", kind: "venture", name: "Acme", confidence: "high" },
        { path: "pitch.md", kind: "venture", name: "Side Thing", confidence: "low" },
        { path: ".env", kind: "none" },
        { path: "made-up.md", kind: "book", name: "Nothing" },
      ],
    };
    await writeFile(join(dir, "reply.md"), "```json\n" + JSON.stringify(reply) + "\n```");
    const draft = (await run("source", "tag", "notes", join(dir, "reply.md"), "--dir", dir)).out;
    expect(draft[0]).toBe("7 placement(s) for files in notes:");
    await mkdir(join(dir, "work", "acme"), { recursive: true });
    const keptOut = (await run("source", "tag", "notes", "--keep", "1,2,3,4,5,6,7", "--dir", dir)).out;
    // A venture with a folder under work/ is mapped to it, never written into; one without a folder but
    // with a statement is kept as an idea; one with neither is only mapped, and says so.
    expect(keptOut).toContain("kept: acme.md -> work/acme");
    expect(await readdir(join(dir, "work", "acme"))).toEqual([]);
    expect(await readFile(join(dir, "ideas/corner-bakery.md"), "utf8")).toMatch(/pointer notes:plan\.md #/);
    expect(keptOut.join("\n")).toContain("note: pitch.md: no venture folder work/side-thing");
    const book = await readFile(join(dir, "learn/books/politics.md"), "utf8");
    expect(book).toMatch(/pointer notes:Reading\/Politics\.md #[0-9a-f]{12}/);
    expect(await readFile(join(dir, "ideas/corrupt-and-correct-cities.md"), "utf8")).toMatch(/pointer notes:cities\.md #/);
    const map = await readFile(join(dir, "sources/notes.md"), "utf8");
    expect(map).toContain('placed "2024-03-01.md"');
    expect(map).toContain("as entry");
    expect(map).toContain("[[documents/diploma]]");
    // The entry is a journal entry that points at the file: listed with the file's text, never a copy of it.
    const entryFile = (await readdir(join(dir, "journal"))).find((n) => n.startsWith("2024-03-01-"))!;
    const entryText = await readFile(join(dir, "journal", entryFile), "utf8");
    expect(entryText).toContain("pointer: notes:2024-03-01.md\npointer_sha: ");
    expect(entryText).not.toContain("Walked downtown");
    expect(map).toContain(`[[journal/${entryFile.replace(/\.md$/, "")}]]`);
    expect((await run("list", "--dir", dir)).out.join("\n")).toContain("Walked downtown today.");
    await run("source", "tag", "notes", "--keep", "3", "--dir", dir);
    expect((await readdir(join(dir, "journal"))).filter((n) => n.startsWith("2024-03-01-"))).toHaveLength(1);
    expect(await readFile(join(notes, "Reading", "Politics.md"), "utf8")).toContain("Book III");
    expect(await run("check", "--dir", dir)).toEqual({ code: 0, out: ["ok"] });
    expect((await run("source", "--dir", dir)).out[1]).toBe("  7 file(s): 7 placed, 0 open");

    await writeFile(join(notes, "Reading", "Politics.md"), "# Politics\n\nRewritten.\n");
    await rm(join(notes, "cities.md"));
    await writeFile(join(notes, "2024-03-01.md"), "Walked downtown today, then home by the river.\n");
    expect((await run("list", "--dir", dir)).out.join("\n")).toContain("home by the river");
    const status = (await run("source", "notes", "--dir", dir)).out.join("\n");
    expect(status).toContain("changed since placed: Reading/Politics.md");
    expect(status).toContain("no longer in the folder: cities.md");
    const check = await run("check", "--dir", dir);
    expect(check.code).not.toBe(0);
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
