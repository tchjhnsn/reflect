#!/usr/bin/env node
// reflect: the command line over one person's instance in a folder. Node 22.18 or later runs it as is.
import { readFileSync } from "node:fs";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseBundle } from "../core/src/bundle.ts";
import { keepDraft, makeDraft, type Draft } from "../core/src/draft.ts";
import { parseArgs } from "node:util";
import { filesJournal } from "../core/src/files-journal.ts";
import { adopt, checkInstance, initInstance, layoutOf, readConfig, upgradeVault } from "../core/src/instance.ts";
import { addGift, addTrustee, LEGACY_ENTRIES, planProblems, previewRelease, readPlan, setTerms } from "../core/src/release.ts";
import { BOOK_STATUS, FILM_STATUS, GRASP, keepBook, keepConversation, keepFilm, keepRecall, keepTalk, latestGrasp, slugOf, parseSubject, keepResource, keepSubject, learnMap, RESOURCE_STATUS, type BookStatus, type FilmStatus, type Grasp, type ResourceStatus, type SubjectRow } from "../core/src/learning.ts";
import { isStance, keepIdea, listIdeas, resolveTarget, STANCES, targetText, type Stance } from "../core/src/ideas.ts";
import { keepSweep, readSweep, sweepLine, SWEEP_PROMPT, type Sweep } from "../core/src/sweep.ts";
import { writeIndex } from "../core/src/catalog.ts";
import { journalHome } from "../core/src/home.ts";
import { initSoul, soulStatus } from "../core/src/soul.ts";
import { confirmReading, keepWikiDraft, readWikiDraft, readWikiReply, wikiOn, wikiPrompt, wikiStatus, type WikiDraft } from "../core/src/wiki.ts";
import { planHandoff, runNaiveCapture, sendIdea } from "../core/src/naive-handoff.ts";
import { addSource, getSource, indexSource, keepPlacements, placementLine, placementPrompt, readDraft, readPlacements, saveDraft, sourceStatus } from "../core/src/sources.ts";
import { keepNeed, keepPassage, LEGACY_PROMPTS, legacyPrompt } from "../core/src/lifetime.ts";
import { lookupIsbn, readCatalogEntry, saveCatalogEntry, searchBooks, type Fetch } from "../core/src/openlibrary.ts";
import { draftTranscript, keepCourse, listCourses, transcriptText, type Chat, type TranscriptDraft } from "../core/src/transcript.ts";
import { chatJson } from "../core/src/reader.ts";
import { checkBadge, keepCredential, keepRecommendation, keepVouches } from "../core/src/credentials.ts";
import { readLinkedInExport } from "../core/src/linkedin.ts";
import { recordPaths } from "../core/src/layout.ts";
import { CUSTODY, DOCUMENT_GROUPS, filesDocuments, isCustody, type ChecklistRow, type Connection } from "../core/src/documents.ts";
import type { IsoDate } from "../core/src/person.ts";
import { readSalience } from "../core/src/reader.ts";
import { cleanNoun, isSparkForm, pickNoun, sparkPrompt, SPARK_FORMS, type SparkForm } from "../core/src/spark.ts";

const USAGE = `reflect <command> [--dir <instance>]

  init --name <name>              start a vault in the folder: the Ventures of a Lifetime (life, needs,
                                  learn, work, belonging, line, household, community, polis, legacy),
                                  with journal, telos, intents and documents
  upgrade                         move a vault made before 2026-11-11 onto those folders
  need <need> <how> [--provider <who>] [--runway 30d] [--note <text>]
                                  keep how a need is met: self-provided, stocked, supplied, cared-for
  passage <region> <state> [--on YYYY-MM[-DD]] [--note <text>]
                                  keep a move in your life: provide, belonging or line
  legacy [prompt] [text]          a prompt for what you want to survive you; with text, keep it as a
                                  journal entry marked to survive you
  release                         your release plan: who may confirm your death, and what of your
                                  legacy goes to whom (only legacy/ and entries marked to survive you)
  release trustee <name> --contact <how to reach them>
  release give <legacy/... | legacy-entries> --to <name> --contact <how to reach them>
  release terms [--confirmations N] [--wait <days>]
  learn                           what you know, what you showed and what was suggested, your books,
                                  films, talks and conversations, what you aspire to and need to learn,
                                  and what you could learn next
  learn find <title or author>     look a book up in Open Library: title, authors, year, ISBN
  learn book <title> [--isbn N [--lookup]] [--by <author;author>] [--status want|reading|read|abandoned|referenced]
                [--subjects a,b] [--on YYYY-MM[-DD]]
                                  --lookup fills authors from Open Library and shows its subjects and
                                  contents; you keep the subjects that fit with --subjects
  learn recall <book> [text]      name a book's key subjects without looking (text from the argument or
                                  stdin); then see which of its subjects you named, and its contents
  learn transcript <file.txt|file.pdf>
                                  a model drafts the courses on your transcript (only rows really in it);
                                  learn/courses/transcript.draft.json holds the draft until you keep rows
  learn transcript --keep <1,2,5 | all>
                                  keep those rows as courses; rate each like any subject
  learn credential <name> --issuer <who> [--issued <date>] [--expires <date>] [--url <link>]
                [--subjects a,b] [--check]
                                  a certificate, license or badge; --check reads an Open Badges link
  learn linkedin <export folder or zip> [--keep all | endorsements,recommendations,certifications,skills]
                                  what your LinkedIn data export says others vouch for; shows it first
  learn grasp <subject> <teach|explain|gist|forgot> [--book <title>] [--note <text>] [--on ...]
                                  your own grasp of a subject: could teach it, could explain the key
                                  ideas, remember the gist, or don't remember it
  learn aspire <subject> [--requires a,b] [--part-of x] [--same-as wikidata:Q123 | lcsh:sh123]
  learn need <subject> --for <intent, venture, need or role> [--requires a,b] [--same-as ...]
  learn resource <name> [--kind video|channel|article|post|thread|newsletter|course|podcast|paper|lab|site]
                [--url <link>] [--subjects a,b] [--by <who>] [--platform <id>] [--status saved|started|finished|dropped]
                                  a link you keep: an X, Substack or Reddit link gets its source from the link
  learn film <title> [--kind film|documentary|series|short] [--year N] [--by <director>]
                [--status want|watching|watched|abandoned|referenced] [--subjects a,b]
  learn talk <title> --speaker <who> [--on <date>] [--where <place>] [--url <link>] [--subjects a,b]
                                  someone you saw speak: a lecture, a sermon, a keynote, a video of one
  learn conversation <what it was about> --with <name;name> [--on <date>] [--subjects a,b]
                                  a conversation that fed your thinking, in your words
  learn sweep                     the prompt to paste into your AI: it lists from its memory of you what
                                  you read, watched and heard, subjects you showed or it suggests, and
                                  your own ideas with their inputs
  learn sweep <reply file>        read your AI's reply into learn/sweep.draft.json and list its items
  learn sweep --keep <1,2,5 | all>
                                  keep those items; an idea's inputs your AI only inferred are kept as
                                  maybe from, and nothing sets your grasp
  source                          your other folders, pointed at instead of moved in: each with how many
                                  files are placed, still open, changed or missing since
  source add <name> <folder> [--exclude <path,*.ext>]
                                  name a folder outside the vault as a source; nothing in it is moved,
                                  edited or copied
  source index <name>             fingerprint its files (hidden folders, dependencies and anything that
                                  looks like a secret are left out)
  source prompt <name> [--limit 200]
                                  the prompt for your own AI: it reads the files and says where each
                                  belongs (a book, film, talk, link, conversation, subject, idea, entry,
                                  venture, document, or nowhere)
  source tag <name> <reply file>  read your AI's reply into a draft and list each placement
  source tag <name> --keep <1,2,5 | all>
                                  keep those: each becomes a pointer on its note and a line in
                                  sources/<name>.md; the file stays where it is. An entry becomes a
                                  journal entry that reads its text from the file, so you can reflect
                                  on it without copying it in. A venture maps to its folder under
                                  work/ (never written into), else becomes an idea if it has a statement
  wiki                            the wiki: written by your own AI from your notes, kept apart from them
  wiki on                         turn it on (wiki/README.md says what it is)
  wiki prompt [--limit 15]        the prompt for your AI: your index, the wiki so far, and entries it has
                                  not absorbed yet
  wiki absorb <reply file>        check your AI's articles: every passage must cite your notes, and a
                                  quote must be in what it cites; lists what passed and what was refused
  wiki absorb --keep <1,2 | all>  keep those articles, log the entries absorbed, rebuild wiki/_index.md
  wiki confirm <path>             a pattern, tension, identity or philosophy is the AI's reading until you
                                  confirm it; a rewrite makes it a reading again
  soul                            your soul vault: moments, readings, bindings, holds, readings of you
  soul init <folder>              start a soul vault in that folder and move the journal into it (entries
                                  become moments, reflections become readings); the two vaults name each
                                  other, and Reflect reads and writes the journal there from either one
  index                           rewrite index.md, the catalog of your notes where your own AI starts
                                  reading: one linked line per note, journal entries by date only
  idea                            your ideas, each with its inputs and recommendations
  idea <title> [statement] [--from <input>]... [--maybe <input>]... [--how <text>]
                [--related <stance>:<thing> --by <who> [--note <text>]]... [--drop <thing>]...
                                  keep an idea in your words (outputs, apart from learn/). An input is a
                                  note (learn/books/x), a title you kept, or your own words; stance is
                                  supports, extends, parallels, challenges or opposes
  idea <title> --send naive [--workspace <dir>] [--yes]
                                  send the idea to Naive with naive capture: shows what goes first, and
                                  sends only with --yes; the idea gets a line naming where it landed
  write [text] [--date YYYY-MM-DD] write an entry (text from the argument or stdin); prints its id
  spark [--minutes 1|5|10|15 | --sentence]
                                  a noun to write about, and how long
  spark <noun> [text] [--minutes N | --sentence] [--date YYYY-MM-DD]
                                  keep what you wrote on that noun (text from the argument or stdin);
                                  the noun becomes a concept note in concepts/
  documents                       the custody checklist for official documents, with what we recommend
  documents <type> <custody> [--where <text>] [--connection <id>]
                                  check one off; custody is original, digital-copy, connection,
                                  missing or not-applicable
  connect <id> --name <name> --kind vault|issuer|portal|platform [--holds <type,type>]
                                  record a service that holds documents for you, or a learning
                                  platform such as Khan Academy, Udemy or Fanout (no credentials)
  add <entry> [text]              add to an entry that is still open
  close <entry>                   close an entry
  list                            the entries, newest first
  read [entry] [--pending]        ask the reader for its guesses; writes reflections/<entry>.draft.json
  keep [entry] [--all]            keep a draft as you left it (edit chosen, basis, keep first)
  check [--adopt]                 check the instance; --adopt records hand-written changes in memory
  export [--out <file>] [--moving-to <where>]
                                  the whole journal as a bundle (stdout or a file); --moving-to then
                                  records that the instance moved and refuses writes here
  import <bundle>                 import a bundle into this instance's empty journal

The reader needs AI_GATEWAY_API_KEY or TYPESAFEAI_API_KEY.`;

const today = (): IsoDate => new Date().toISOString().slice(0, 10) as IsoDate;

function textFrom(arg: string | undefined): string {
  const t = (arg ?? readFileSync(0, "utf8")).replace(/\r\n/g, "\n").trim();
  if (!t) throw new Error("nothing to write");
  return t;
}

const ADVICE = { "keep-paper-safe": "keep the paper safe", "store-digital": "store it digitally", connect: "connect the service that holds it" } as const;

function checklistLine(r: ChecklistRow, names: Map<string, string>): string {
  const held = r.heldBy.length ? ` (held by ${r.heldBy.map((h) => names.get(h) ?? h).join(", ")})` : "";
  if (r.custody) {
    const at = r.custody === "connection" ? ` ${names.get(r.connection ?? "") ?? r.connection}` : r.where ? `: ${r.where}` : "";
    return `[x] ${r.type.id}  ${r.custody}${at}`;
  }
  const services = r.type.recommend === "connect" && r.type.services ? `, such as ${r.type.services.join(" or ")}` : "";
  return `[ ] ${r.type.id}${held}  ${ADVICE[r.type.recommend]}${services}. ${r.type.norm}`;
}

function sparkForm(minutes: string | undefined, sentence: boolean): SparkForm {
  if (sentence && minutes) throw new Error("a spark is --minutes or --sentence, not both");
  if (sentence) return "sentence";
  const form = `${minutes ?? "5"}m`;
  if (!isSparkForm(form)) throw new Error(`--minutes is one of ${SPARK_FORMS.filter((f) => f !== "sentence").map((f) => f.replace("m", "")).join(", ")}`);
  return form;
}

export async function main(argv: string[], out: (s: string) => void = console.log, fetcher: Fetch = (url) => fetch(url, { headers: { "user-agent": "reflect-cli" } }), chat: Chat = (system, t) => chatJson(system, t, process.env.AI_GATEWAY_API_KEY ?? null)): Promise<number> {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      dir: { type: "string", default: "." },
      name: { type: "string" },
      date: { type: "string" },
      adopt: { type: "boolean", default: false },
      pending: { type: "boolean", default: false },
      all: { type: "boolean", default: false },
      minutes: { type: "string" },
      sentence: { type: "boolean", default: false },
      where: { type: "string" },
      connection: { type: "string" },
      kind: { type: "string" },
      holds: { type: "string" },
      provider: { type: "string" },
      runway: { type: "string" },
      note: { type: "string" },
      on: { type: "string" },
      to: { type: "string" },
      contact: { type: "string" },
      confirmations: { type: "string" },
      wait: { type: "string" },
      for: { type: "string" },
      requires: { type: "string" },
      "part-of": { type: "string" },
      "same-as": { type: "string" },
      by: { type: "string" },
      url: { type: "string" },
      subjects: { type: "string" },
      isbn: { type: "string" },
      lookup: { type: "boolean", default: false },
      keep: { type: "string" },
      issuer: { type: "string" },
      issued: { type: "string" },
      expires: { type: "string" },
      check: { type: "boolean", default: false },
      book: { type: "string" },
      platform: { type: "string" },
      status: { type: "string" },
      out: { type: "string" },
      "moving-to": { type: "string" },
      speaker: { type: "string" },
      year: { type: "string" },
      with: { type: "string" },
      how: { type: "string" },
      from: { type: "string", multiple: true },
      maybe: { type: "string", multiple: true },
      related: { type: "string", multiple: true },
      drop: { type: "string", multiple: true },
      exclude: { type: "string" },
      limit: { type: "string" },
      send: { type: "string" },
      workspace: { type: "string" },
      yes: { type: "boolean", default: false },
      help: { type: "boolean", short: "h" },
    },
  });
  const [cmd, ...rest] = positionals;
  const root = values.dir!;
  if (!cmd || values.help) {
    out(USAGE);
    return cmd ? 0 : 2;
  }
  if (cmd === "init") {
    if (!values.name) throw new Error("init needs --name");
    const c = await initInstance(root, values.name);
    out(`started ${c.person} in ${root}`);
    return 0;
  }
  if (cmd === "check") {
    const r = await checkInstance(root);
    for (const p of r.problems) out(`problem: ${p}`);
    if (values.adopt && r.problems.length === 0 && r.outside.length) {
      await adopt(root, r.outside);
      out(`adopted ${r.outside.length} record(s) written outside reflect`);
    } else for (const p of r.outside) out(`written outside reflect: ${p}`);
    if (r.problems.length === 0) out("ok");
    return r.problems.length ? 1 : 0;
  }
  if (cmd === "spark" && !rest[0]) {
    const noun = pickNoun();
    const form = sparkForm(values.minutes, values.sentence);
    out(`${noun}\n${sparkPrompt(noun, form)}\nThen keep it: reflect spark "${noun}"${form === "sentence" ? " --sentence" : ` --minutes ${form.replace("m", "")}`} < what-you-wrote.txt`);
    return 0;
  }
  const config = await readConfig(root);
  if (cmd === "soul") {
    const [sub, arg] = rest;
    if (sub === "init") {
      if (!arg) throw new Error("soul init <folder for the soul vault>");
      const r = await initSoul(root, arg, new Date());
      out(`started the soul vault at ${r.soul}; moved ${r.moved.length} file(s) of the journal into it`);
      out("Reflect now reads and writes your journal there, from either folder.");
      return 0;
    }
    if (sub) throw new Error("soul takes init, or nothing to show the soul vault");
    const home = await journalHome(root);
    if (!home.soul) {
      out("No soul vault yet. Your journal and readings can live in a vault of their own, apart from this one: reflect soul init <folder>");
      return 0;
    }
    const st = await soulStatus(home.root);
    out(`soul vault: ${st.root}`);
    out(`${st.moments} moment(s), ${st.readings} reading file(s), ${st.bindings} binding file(s), ${st.holds} hold(s), ${st.ofMe} reading(s) of you`);
    return 0;
  }
  const paths = recordPaths(layoutOf(config), config.person);
  const home = await journalHome(root);
  const journal = filesJournal(home.root, config.person, { layout: home.layout, lifeRoot: home.lifeRoot });
  if (cmd === "list") {
    for (const e of await journal.listEntries()) out(`${e.id}  ${e.localDate}  ${e.closedAt ? "closed" : "open  "}  ${(e.parts[0]?.text ?? "").split("\n")[0]!.slice(0, 60)}`);
    return 0;
  }
  if (cmd === "export") {
    const text = `${JSON.stringify(await journal.exportBundle(), null, 2)}\n`;
    if (values.out) await writeFile(values.out, text);
    else out(text.trimEnd());
    if (values["moving-to"]) {
      await writeFile(join(root, "reflect.json"), `${JSON.stringify({ ...config, moved_to: values["moving-to"] }, null, 2)}\n`);
      await adopt(root, ["reflect.json"]);
    }
    return 0;
  }
  const docs = filesDocuments(root, paths);
  if (cmd === "documents" && !rest[0]) {
    const rows = await docs.checklist();
    const names = new Map((await docs.connections()).map((k) => [k.id, k.name]));
    for (const g of DOCUMENT_GROUPS) {
      out(g);
      for (const r of rows.filter((x) => x.type.group === g)) out(`  ${checklistLine(r, names)}`);
    }
    const open = rows.filter((r) => r.custody === null).length;
    out(`${rows.length - open} of ${rows.length} checked`);
    return 0;
  }
  if (cmd === "legacy" && !rest[0]) {
    const p = LEGACY_PROMPTS[Math.floor(Math.random() * LEGACY_PROMPTS.length)]!;
    out(`${p.text}\nThen keep it: reflect legacy ${p.id} < what-you-wrote.txt`);
    out(`Other prompts: ${LEGACY_PROMPTS.filter((x) => x.id !== p.id).map((x) => x.id).join(", ")}`);
    return 0;
  }
  if (config.moved_to) throw new Error(`this instance moved to ${config.moved_to}; write there`);
  if (cmd === "import" && rest[0]) {
    await journal.importBundle(parseBundle(await readFile(rest[0], "utf8")));
    out(`imported ${rest[0]}`);
    return 0;
  }
  const date = values.date ?? today();
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error("--date is YYYY-MM-DD");
  if (cmd === "write") {
    out(await journal.createEntry(textFrom(rest[0]), date as IsoDate, null));
    return 0;
  }
  if (cmd === "spark") {
    const form = sparkForm(values.minutes, values.sentence);
    const text = textFrom(rest[1]);
    out(await journal.writeSpark(cleanNoun(rest[0]!), form, text, date as IsoDate));
    return 0;
  }
  const reflDir = join(home.root, home.paths.reflections);
  const draftPath = (entry: string) => join(reflDir, `${entry}.draft.json`);
  const exists = (p: string) => readFile(p).then(() => true, () => false);
  if (cmd === "read") {
    let ids: string[];
    if (values.pending) {
      ids = [];
      for (const e of await journal.listEntries(10_000)) {
        const refl = await readFile(join(reflDir, `${e.id}.json`), "utf8").then((t) => JSON.parse(t) as { labelings?: unknown[] }, () => null);
        if (!(refl?.labelings?.length) && !(await exists(draftPath(e.id)))) ids.push(e.id);
      }
    } else if (rest[0]) ids = [rest[0]];
    else throw new Error("read needs an entry or --pending");
    for (const id of ids) {
      const text = await journal.entryText(id);
      if (text === null) throw new Error(`no such entry: ${id}`);
      const s = await readSalience(text);
      if (!s) throw new Error("the reader is not set up or did not answer; set AI_GATEWAY_API_KEY or TYPESAFEAI_API_KEY");
      await writeFile(draftPath(id), `${JSON.stringify(makeDraft(id, s, new Date().toISOString()), null, 2)}\n`);
      out(`draft: ${join(home.paths.reflections, `${id}.draft.json`)}`);
    }
    return 0;
  }
  if (cmd === "keep") {
    let ids: string[];
    if (values.all) {
      const { readdir } = await import("node:fs/promises");
      ids = (await readdir(reflDir).catch(() => [] as string[])).filter((n) => n.endsWith(".draft.json")).map((n) => n.replace(/\.draft\.json$/, ""));
    } else if (rest[0]) ids = [rest[0]];
    else throw new Error("keep needs an entry or --all");
    for (const id of ids) {
      const draft = JSON.parse(await readFile(draftPath(id), "utf8")) as Draft;
      if (draft.entry !== id) throw new Error(`${id}.draft.json is for another entry`);
      const { labels, terms } = keepDraft(draft);
      await journal.keepLabels(id, labels, terms);
      await rm(draftPath(id));
      out(`kept: ${id} (${draft.chosen.join(", ") || "no elements"})`);
    }
    return 0;
  }
  if (cmd === "upgrade") {
    const r = await upgradeVault(root);
    for (const m of r.moved) out(`moved: ${m}`);
    for (const k of r.kept) out(`left in place, because the new path is taken: ${k}`);
    out(`added ${r.written.length} folder guide(s) and template(s)`);
    return 0;
  }
  if (cmd === "learn") {
    if (layoutOf(config) !== "vault") throw new Error("learn needs a vault; this instance uses the older people/<id> layout");
    const [sub, name, level] = rest;
    const now = new Date();
    const list = (v: string | undefined, sep = ",") => (v ?? "").split(sep).map((x) => x.trim()).filter(Boolean);
    const where = { requires: list(values.requires), part_of: list(values["part-of"]), ...(values["same-as"] ? { same_as: values["same-as"] } : {}) };
    if (sub === "find") {
      if (!name) throw new Error("learn find <title or author>");
      const found = await searchBooks(rest.slice(1).join(" "), fetcher);
      for (const e of found) out(`${e.title}, ${e.authors.join(", ") || "author unknown"}${e.year ? ` (${e.year})` : ""}${e.isbn ? `  isbn ${e.isbn}` : ""}`);
      if (!found.length) out("Open Library has nothing by that name");
      else out('Keep one: reflect learn book "<title>" --isbn <isbn> --lookup');
      return 0;
    }
    if (sub === "book") {
      if (!name) throw new Error("learn book <title> [--isbn N] [--status read]");
      if (values.status && !(BOOK_STATUS as readonly string[]).includes(values.status)) throw new Error(`--status is one of ${BOOK_STATUS.join(", ")}`);
      if (values.lookup && !values.isbn) throw new Error("--lookup needs --isbn; find it with: reflect learn find <title>");
      const entry = values.lookup ? await lookupIsbn(values.isbn!, fetcher) : null;
      if (values.lookup && !entry) throw new Error(`Open Library has no book with isbn ${values.isbn}`);
      const authors = list(values.by, ";");
      const b = await keepBook(root, name, { ...(values.isbn ? { isbn: values.isbn } : {}), ...(entry ? { openlibrary: entry.work } : {}), authors: authors.length ? authors : (entry?.authors ?? []), subjects: list(values.subjects), ...(values.status ? { status: values.status as BookStatus } : {}), ...(values.on ? { on: values.on } : {}) }, now);
      out(`kept: learn/books/${b.id}.md ${b.status}`);
      if (entry) {
        await saveCatalogEntry(root, b.id, entry, now);
        if (entry.title && entry.title.toLowerCase() !== b.title.toLowerCase()) out(`Open Library calls it "${entry.title}"`);
        if (entry.subjects.length) out(`Open Library lists these subjects: ${entry.subjects.join("; ")}. Keep the ones that fit: --subjects a,b`);
        if (entry.contents.length) out(`Contents:\n${entry.contents.map((c) => `  ${c}`).join("\n")}`);
      }
      if (b.status === "read" && b.subjects.length) out(`Which of its subjects can you name without looking? Then rate each: reflect learn grasp <subject> <${GRASP.map((g) => g.id).join("|")}> --book "${b.title}"`);
      return 0;
    }
    if (sub === "recall") {
      if (!name) throw new Error("learn recall <book> [what you remember]");
      if (rest[2] === undefined && process.stdin.isTTY) {
        out(`Without looking, name the key subjects of ${name}: what it is about, its main ideas, what you would tell someone. Then: reflect learn recall "${name}" "<what you named>"`);
        return 0;
      }
      const r = await keepRecall(root, name, textFrom(rest[2]), now, (await readCatalogEntry(root, name))?.contents ?? []);
      out(`kept your recall on learn/books/${r.book.id}.md`);
      if (r.named.length) out(`You named: ${r.named.join(", ")}`);
      if (r.unnamed.length) out(`Its other subjects: ${r.unnamed.join(", ")}`);
      if (!r.book.subjects.length) out("The book has no subjects yet; add the ones you named: reflect learn book <title> --subjects a,b");
      if (r.contents.length) out(`Contents:\n${r.contents.map((c) => `  ${c}`).join("\n")}`);
      out(`Now rate each: reflect learn grasp <subject> <${GRASP.map((g) => g.id).join("|")}> --book "${r.book.title}"`);
      return 0;
    }
    if (sub === "transcript") {
      const draftRel = join("learn", "courses", "transcript.draft.json");
      if (values.keep) {
        const draft = JSON.parse(await readFile(join(root, draftRel), "utf8").catch(() => { throw new Error("no transcript draft; run: reflect learn transcript <file>"); })) as TranscriptDraft;
        const picks = values.keep === "all" ? draft.courses.map((_, i) => i + 1) : list(values.keep).map(Number);
        for (const n of picks) {
          const c = draft.courses[n - 1];
          if (!c) throw new Error(`the draft has no row ${n}`);
          out(`kept: ${await keepCourse(root, c, draft.institution, now)}`);
        }
        out(`Rate each: reflect learn grasp "<course title>" <${GRASP.map((g) => g.id).join("|")}> --note "<what you remember>"`);
        return 0;
      }
      if (!name) throw new Error("learn transcript <file.txt|file.pdf>, or --keep <rows>");
      const bytes = new Uint8Array(await readFile(name));
      let text: string;
      if (/\.pdf$/i.test(name)) {
        const mod = "unpdf";
        const unpdf = (await import(mod).catch(() => null)) as { extractText(d: Uint8Array, o: { mergePages: true }): Promise<{ text: string }> } | null;
        if (!unpdf) throw new Error("reading a PDF needs the unpdf package; or save the transcript as text and pass that");
        text = (await unpdf.extractText(bytes, { mergePages: true })).text;
      } else text = transcriptText(bytes, name);
      const draft = await draftTranscript(text, chat);
      if (!draft) throw new Error("no model is set up: set AI_GATEWAY_API_KEY or OLLAMA_API_KEY");
      await mkdir(join(root, "learn", "courses"), { recursive: true });
      await writeFile(join(root, draftRel), `${JSON.stringify(draft, null, 2)}\n`);
      if (draft.institution) out(draft.institution);
      draft.courses.forEach((c, i) => out(`${i + 1}. ${c.term}  ${c.code ? `${c.code} ` : ""}${c.title}${c.credits !== undefined ? `, ${c.credits} credits` : ""}${c.grade ? `, ${c.grade}` : ""}`));
      if (draft.dropped.length) out(`Left out, because they are not in the transcript as written: ${draft.dropped.join("; ")}`);
      out("Check each row against your transcript, then keep the right ones: reflect learn transcript --keep 1,2,3 (or all)");
      return 0;
    }
    if (sub === "credential") {
      if (!name) throw new Error("learn credential <name> --issuer <who>");
      const checked = values.check && values.url ? await checkBadge(values.url, fetcher) : null;
      if (values.check && !values.url) throw new Error("--check needs --url, the badge's link");
      const c = await keepCredential(root, name, { issuer: values.issuer ?? checked?.issuer ?? "", ...(values.issued ?? checked?.issued ? { issued: values.issued ?? checked!.issued } : {}), ...(values.expires ? { expires: values.expires } : {}), ...(values.url ? { url: values.url } : {}), subjects: list(values.subjects), ...(checked ? { checked: { ...checked, at: now.toISOString() } } : {}) }, now);
      out(`kept: learn/credentials/${c.id}.md`);
      if (values.check) out(checked ? `The issuer's badge says: ${checked.name}, from ${checked.issuer}${checked.issued ? `, ${checked.issued}` : ""}` : "That link is not an Open Badges badge; the credential is kept as you stated it");
      return 0;
    }
    if (sub === "linkedin") {
      if (!name) throw new Error("learn linkedin <export folder or zip>");
      const files = new Map<string, string>();
      if (/\.zip$/i.test(name)) {
        const mod = "fflate";
        const fflate = (await import(mod).catch(() => null)) as { unzipSync(d: Uint8Array): Record<string, Uint8Array> } | null;
        if (!fflate) throw new Error("reading a zip needs the fflate package; or unzip it and pass the folder");
        for (const [k, v] of Object.entries(fflate.unzipSync(new Uint8Array(await readFile(name))))) if (k.toLowerCase().endsWith(".csv")) files.set(k, new TextDecoder().decode(v));
      } else {
        const { readdir } = await import("node:fs/promises");
        for (const f of await readdir(name)) if (f.toLowerCase().endsWith(".csv")) files.set(f, await readFile(join(name, f), "utf8"));
      }
      const x = readLinkedInExport(files);
      const bySkill = new Map<string, typeof x.endorsements>();
      for (const e of x.endorsements) bySkill.set(e.skill, [...(bySkill.get(e.skill) ?? []), e]);
      if (!values.keep) {
        out(`Endorsements: ${x.endorsements.length}, on ${bySkill.size} skill(s)`);
        for (const [skill, es] of bySkill) out(`  ${skill}: ${es.map((e) => e.by).join(", ")}`);
        out(`Recommendations: ${x.recommendations.length}${x.recommendations.length ? ` (from ${x.recommendations.map((r) => r.by).join(", ")})` : ""}`);
        out(`Certifications: ${x.certifications.length}${x.certifications.length ? ` (${x.certifications.map((c) => c.name).join(", ")})` : ""}`);
        out(`Skills you listed: ${x.skills.length}`);
        out("Keep what is right: reflect learn linkedin <export> --keep all (or endorsements,recommendations,certifications,skills)");
        return 0;
      }
      const kinds = values.keep === "all" ? ["endorsements", "recommendations", "certifications", "skills"] : list(values.keep);
      for (const k of kinds) if (!["endorsements", "recommendations", "certifications", "skills"].includes(k)) throw new Error(`--keep takes all or endorsements, recommendations, certifications, skills; not ${k}`);
      let n = 0;
      if (kinds.includes("endorsements")) for (const [skill, es] of bySkill) n += await keepVouches(root, skill, es.map((e) => ({ on: e.on, by: e.by, ...(e.url ? { url: e.url } : {}), source: "linkedin" })), now);
      if (kinds.includes("recommendations")) for (const r of x.recommendations) await keepRecommendation(root, { ...r, source: "linkedin" }, now);
      if (kinds.includes("certifications")) for (const c of x.certifications) await keepCredential(root, c.name, { issuer: c.issuer, ...(c.issued ? { issued: c.issued } : {}), ...(c.expires ? { expires: c.expires } : {}), ...(c.url ? { url: c.url } : {}) }, now);
      if (kinds.includes("skills")) for (const sk of x.skills) await keepSubject(root, sk, {}, now);
      out(`kept: ${n} new vouch(es)${kinds.includes("recommendations") ? `, ${x.recommendations.length} recommendation(s)` : ""}${kinds.includes("certifications") ? `, ${x.certifications.length} certification(s)` : ""}${kinds.includes("skills") ? `, ${x.skills.length} skill(s) as subjects for you to rate` : ""}`);
      return 0;
    }
    if (sub === "grasp") {
      if (!name || !level) throw new Error(`learn grasp <subject> <${GRASP.map((g) => g.id).join("|")}>`);
      const s = await keepSubject(root, name, { ...where, grasp: { grasp: level as Grasp, ...(values.book ? { book: values.book } : {}), ...(values.note ? { note: values.note } : {}), ...(values.on ? { on: values.on } : {}) } }, now);
      out(`kept: learn/subjects/${s.id}.md ${GRASP.find((g) => g.id === level)!.text}`);
      return 0;
    }
    if (sub === "aspire" || sub === "need") {
      if (!name) throw new Error(`learn ${sub} <subject>`);
      const s = await keepSubject(root, name, { ...where, want: sub, ...(values.for ? { for: values.for } : {}) }, now);
      out(`kept: learn/subjects/${s.id}.md ${sub}${s.for ? ` for ${s.for}` : ""}`);
      return 0;
    }
    if (sub === "film") {
      if (!name) throw new Error("learn film <title> [--status watched] [--subjects a,b]");
      if (values.status && !(FILM_STATUS as readonly string[]).includes(values.status)) throw new Error(`--status is one of ${FILM_STATUS.join(", ")}`);
      const f = await keepFilm(root, name, { ...(values.kind ? { kind: values.kind } : {}), ...(values.year ? { year: Number(values.year) } : {}), ...(values.by ? { by: values.by } : {}), ...(values.status ? { status: values.status as FilmStatus } : {}), subjects: list(values.subjects), ...(values.on ? { on: values.on } : {}) }, now);
      out(`kept: learn/films/${f.id}.md ${f.status}`);
      return 0;
    }
    if (sub === "talk") {
      if (!name) throw new Error("learn talk <title> --speaker <who>");
      const t = await keepTalk(root, name, { ...(values.speaker ? { speaker: values.speaker } : {}), ...(values.on ? { on: values.on } : {}), ...(values.where ? { where: values.where } : {}), ...(values.url ? { url: values.url } : {}), subjects: list(values.subjects) }, now);
      out(`kept: learn/talks/${t.id}.md ${t.speaker}`);
      return 0;
    }
    if (sub === "conversation") {
      if (!name || !values.with) throw new Error("learn conversation <what it was about> --with <name;name>");
      const c = await keepConversation(root, { with: values.with.split(";"), ...(values.on ? { on: values.on } : {}), about: name, subjects: list(values.subjects) }, now);
      out(`kept: learn/conversations/${c.id}.md`);
      return 0;
    }
    if (sub === "sweep") {
      const draftRel = join("learn", "sweep.draft.json");
      if (values.keep) {
        const s = JSON.parse(await readFile(join(root, draftRel), "utf8").catch(() => { throw new Error("no sweep draft; run: reflect learn sweep <your AI's reply>"); })) as Sweep;
        const nums = values.keep === "all" ? s.items.map((_, i) => i + 1) : values.keep.split(",").map((x) => Number(x.trim()));
        if (nums.some((n) => !Number.isInteger(n) || n < 1)) throw new Error("--keep takes all or item numbers like 1,2,5");
        for (const k of await keepSweep(root, s, nums, now)) out(`kept: ${k}`);
        out(`Rate what you know: reflect learn grasp <subject> <${GRASP.map((g) => g.id).join("|")}>`);
        return 0;
      }
      if (!name) {
        out(SWEEP_PROMPT);
        return 0;
      }
      const s = readSweep(await readFile(name, "utf8"));
      await mkdir(join(root, "learn"), { recursive: true });
      await writeFile(join(root, draftRel), `${JSON.stringify(s, null, 2)}\n`);
      out(`${s.assistant}${s.reportedAt ? ` (${s.reportedAt})` : ""} found ${s.items.length} item(s):`);
      s.items.forEach((i, n) => out(`  ${n + 1}. ${sweepLine(i)}`));
      for (const oq of s.openQuestions) out(`  open: ${oq}`);
      out("Check each against what you know, then keep the right ones: reflect learn sweep --keep 1,2,3 (or all)");
      return 0;
    }
    if (sub === "resource") {
      if (!name) throw new Error("learn resource <name> --kind <kind> [--url <link>] [--subjects a,b]");
      if (values.status && !(RESOURCE_STATUS as readonly string[]).includes(values.status)) throw new Error(`--status is one of ${RESOURCE_STATUS.join(", ")}`);
      const r = await keepResource(root, name, { ...(values.kind ? { kind: values.kind } : {}), ...(values.url ? { url: values.url } : {}), ...(values.by ? { by: values.by } : {}), ...(values.platform ? { platform: values.platform } : {}), subjects: list(values.subjects), ...(values.status ? { status: values.status as ResourceStatus } : {}) }, now);
      out(`kept: learn/resources/${r.id}.md ${r.status}`);
      return 0;
    }
    if (sub) throw new Error("learn takes find, book, recall, transcript, credential, linkedin, grasp, aspire, need, resource, film, talk, conversation or sweep");
    const m = await learnMap(root);
    const text = (g: Grasp) => GRASP.find((x) => x.id === g)!.text;
    const from = (r: SubjectRow) => [...r.books.map((b) => b.id), ...r.films.map((f) => f.id), ...r.talks.map((t) => t.id), ...r.resources.map((x) => `${x.id} (${x.status})`)];
    const ready = (r: SubjectRow) => (r.blockedBy.length ? `first learn ${r.blockedBy.join(", ")}` : "ready");
    const via = (r: SubjectRow) => (from(r).length ? `; from ${from(r).join(", ")}` : "");
    out("Known");
    const vouched = (r: SubjectRow) => (r.vouches ? `; vouched for by ${r.vouches}` : "");
    for (const r of m.known) out(`  ${r.subject.id}  ${text(r.grasp!.grasp)} (${r.grasp!.on})${r.grasp!.book ? `, from ${r.grasp!.book}` : ""}${vouched(r)}`);
    if (!m.known.length) out("  nothing rated yet: reflect learn grasp <subject> <level>");
    const courses = await listCourses(root);
    if (courses.length) {
      out("Courses");
      const graspOf = new Map([...m.known, ...m.aspire, ...m.need, ...m.possible].map((r) => [r.subject.id, r.grasp] as const));
      for (const c of courses) {
        let g = graspOf.get(c.subject);
        if (!g) {
          const t = await readFile(join(root, "learn", "subjects", `${c.subject}.md`), "utf8").catch(() => "");
          g = latestGrasp(parseSubject(c.subject, t).grasps);
        }
        out(`  ${c.term}  ${c.code ? `${c.code} ` : ""}${c.title}${c.grade ? `  grade ${c.grade}` : ""}; ${g ? `you say: ${text(g.grasp)}` : "not yet rated by you"}`);
      }
    }
    out("Books");
    for (const { book: b, unrated } of m.books) out(`  ${b.id}  ${b.status}${b.finished ? ` ${b.finished}` : ""}${unrated.length ? `; not yet rated: ${unrated.join(", ")}` : ""}`);
    if (!m.books.length) out("  none yet: reflect learn book <title> --status read");
    if (m.shown.length) {
      out("Shown, not yet rated by you");
      for (const r of m.shown) out(`  ${r.subject.id}  shown ${r.subject.shown} time(s)`);
    }
    if (m.vouched.length) {
      out("Vouched for by others, not yet rated by you");
      for (const r of m.vouched) out(`  ${r.subject.id}  vouched for by ${r.vouches}`);
    }
    if (m.suggested.length) {
      out("Suggested, not known");
      for (const r of m.suggested) out(`  ${r.subject.id}${via(r)}`);
    }
    if (m.films.length) {
      out("Films");
      for (const f of m.films) out(`  ${f.id}  ${f.kind}, ${f.status}${f.watched ? ` ${f.watched}` : ""}`);
    }
    if (m.talks.length) {
      out("Talks");
      for (const t of m.talks) out(`  ${t.id}  ${t.speaker}${t.on ? `, ${t.on}` : ""}`);
    }
    if (m.conversations.length) {
      out("Conversations");
      for (const c of m.conversations) out(`  ${c.on}  with ${c.with.join(", ")}: ${c.about}`);
    }
    out("Aspire");
    for (const r of m.aspire) out(`  ${r.subject.id}  ${ready(r)}${via(r)}`);
    if (!m.aspire.length) out("  nothing yet: reflect learn aspire <subject>");
    out("Need");
    for (const r of m.need) out(`  ${r.subject.id}  for ${r.subject.for}; ${ready(r)}${via(r)}`);
    if (!m.need.length) out("  nothing yet: reflect learn need <subject> --for <what needs it>");
    out("Possible next");
    for (const r of m.possible) out(`  ${r.subject.id}  ${r.subject.requires.length ? "you know what it requires" : `a first step toward ${[...m.aspire, ...m.need].filter((w) => w.subject.requires.includes(r.subject.id)).map((w) => w.subject.id).join(", ") || "what needs it"}`}`);
    if (!m.possible.length) out("  nothing your map names yet; the rest is the whole internet");
    return 0;
  }
  if (cmd === "source") {
    if (layoutOf(config) !== "vault") throw new Error("source needs a vault; this instance uses the older people/<id> layout");
    const [sub, name, arg] = rest;
    const now = new Date();
    if (sub === "add") {
      if (!name || !arg) throw new Error("source add <name> <folder>");
      const s = await addSource(root, name, arg, { exclude: (values.exclude ?? "").split(",") }, now);
      const { files, skipped } = await indexSource(root, s.name);
      out(`kept: ${s.name} -> ${s.path}; ${files.length} file(s) indexed, ${skipped} left out`);
      out(`Next: reflect source prompt ${s.name}, and give it to your AI`);
      return 0;
    }
    if (sub === "index") {
      if (!name) throw new Error("source index <name>");
      const { files, skipped } = await indexSource(root, name);
      out(`${name}: ${files.length} file(s) indexed, ${skipped} left out`);
      return 0;
    }
    if (sub === "prompt") {
      if (!name) throw new Error("source prompt <name>");
      const p = await placementPrompt(root, name, values.limit ? Number(values.limit) : 200);
      out(p.text.trimEnd());
      if (p.remaining) out(`(${p.remaining} more after these; keep this batch, then ask again)`);
      return 0;
    }
    if (sub === "tag") {
      if (!name) throw new Error("source tag <name> <reply file>, or --keep <numbers>");
      if (values.keep) {
        const draft = await readDraft(root, name);
        const nums = values.keep === "all" ? draft.map((_, i) => i + 1) : values.keep.split(",").map((x) => Number(x.trim()));
        if (nums.some((n) => !Number.isInteger(n) || n < 1)) throw new Error("--keep takes all or numbers like 1,2,5");
        for (const k of await keepPlacements(root, name, nums, config.person, now, layoutOf(config))) out(k.startsWith("note: ") ? k : `kept: ${k}`);
        return 0;
      }
      if (!arg) throw new Error("source tag <name> <reply file>");
      await getSource(root, name);
      const ps = await readPlacements(root, name, await readFile(arg, "utf8"));
      await saveDraft(root, name, ps);
      out(`${ps.length} placement(s) for files in ${name}:`);
      ps.forEach((p, i) => out(`  ${i + 1}. ${placementLine(p)}`));
      out(`Check each, then keep the right ones: reflect source tag ${name} --keep 1,2,3 (or all)`);
      return 0;
    }
    const all = await sourceStatus(root);
    const shown = sub ? all.filter((s) => s.source.name === sub) : all;
    if (sub && !shown.length) throw new Error(`no source named ${sub}`);
    if (!all.length) out("No sources yet: reflect source add <name> <folder>");
    for (const s of shown) {
      out(`${s.source.name}  ${s.source.path}${s.reachable ? "" : "  (not reachable here)"}`);
      if (s.reachable) out(`  ${s.files} file(s): ${s.placed} placed, ${s.open} open${s.changed.length ? `, ${s.changed.length} changed` : ""}${s.missing.length ? `, ${s.missing.length} missing` : ""}`);
      if (sub) {
        for (const p of s.changed) out(`  changed since placed: ${p}`);
        for (const p of s.missing) out(`  no longer in the folder: ${p}`);
      }
    }
    return 0;
  }
  if (cmd === "wiki") {
    if (layoutOf(config) !== "vault") throw new Error("wiki needs a vault; this instance uses the older people/<id> layout");
    const [sub, arg] = rest;
    const now = new Date();
    const show = (d: WikiDraft) => {
      out(`${d.articles.length} article(s) to keep, ${d.absorbed.length} entr${d.absorbed.length === 1 ? "y" : "ies"} absorbed:`);
      d.articles.forEach((a, n) => out(`  ${n + 1}. wiki/${a.path} (${a.type}) ${a.title}`));
      for (const r of d.rejected) out(`  refused: ${r.path}: ${r.why}`);
      if (d.articles.length) out("Read them, then keep the right ones: reflect wiki absorb --keep 1,2 (or all)");
    };
    if (sub === "on") {
      out((await wikiOn(root, now)) ? "the wiki is on: wiki/README.md says what it is. Next: reflect wiki prompt" : "the wiki is already on");
      return 0;
    }
    if (sub === "prompt") {
      const p = await wikiPrompt(root, layoutOf(config), config.person, values.limit ? Number(values.limit) : 15);
      out(p.text);
      out("");
      out(`(${p.entries.length} entr${p.entries.length === 1 ? "y" : "ies"} in this prompt, ${p.remaining} more after. Save your AI's reply to a file, then: reflect wiki absorb <file>)`);
      return 0;
    }
    if (sub === "absorb") {
      if (values.keep) {
        const nums = values.keep === "all" ? "all" : values.keep.split(",").map((x) => Number(x.trim()));
        if (nums !== "all" && nums.some((n) => !Number.isInteger(n) || n < 1)) throw new Error("--keep takes numbers like 1,2,5 or all");
        for (const k of await keepWikiDraft(root, nums, now)) out(`kept: ${k}`);
        return 0;
      }
      if (!arg) {
        show(await readWikiDraft(root));
        return 0;
      }
      show(await readWikiReply(root, layoutOf(config), config.person, await readFile(arg, "utf8")));
      return 0;
    }
    if (sub === "confirm") {
      if (!arg) throw new Error("wiki confirm <article path, like patterns/building-alone>");
      await confirmReading(root, arg, now);
      out(`confirmed: wiki/${arg.replace(/^wiki\//, "").replace(/\.md$/, "")} is yours now, as written`);
      return 0;
    }
    if (sub) throw new Error("wiki takes on, prompt, absorb or confirm");
    const s = await wikiStatus(root, layoutOf(config), config.person);
    if (!s.on) {
      out("The wiki is off. It is written by your own AI from your notes and journal, and kept apart from them. Turn it on: reflect wiki on");
      return 0;
    }
    out(`${s.articles} article(s)${Object.keys(s.byType).length ? `: ${Object.entries(s.byType).map(([t, n]) => `${n} ${t}`).join(", ")}` : ""}`);
    out(`${s.readings} reading(s) of you, ${s.confirmed} confirmed`);
    out(`${s.absorbed} entr${s.absorbed === 1 ? "y" : "ies"} absorbed, ${s.pending} waiting`);
    return 0;
  }
  if (cmd === "index") {
    if (layoutOf(config) !== "vault") throw new Error("index needs a vault; this instance uses the older people/<id> layout");
    out((await writeIndex(root, layoutOf(config), config.person, new Date())) ? "wrote index.md" : "index.md is up to date");
    return 0;
  }
  if (cmd === "idea") {
    if (layoutOf(config) !== "vault") throw new Error("idea needs a vault; this instance uses the older people/<id> layout");
    const [title, statement] = rest;
    if (!title) {
      const ideas = await listIdeas(root);
      if (!ideas.length) out("No ideas yet: reflect idea <title> <the idea in your words> --from <what fed it>");
      for (const i of ideas) {
        out(`${i.id}  ${i.statement ?? ""}`);
        for (const x of i.from) out(`  from ${targetText(x)}${x.how ? `: ${x.how}` : ""}`);
        for (const x of i.maybe) out(`  maybe from ${targetText(x)}${x.how ? `: ${x.how}` : ""}`);
        for (const r of i.related) out(`  ${r.stance}: ${targetText(r)}, suggested by ${r.by}${r.note ? `: ${r.note}` : ""}`);
      }
      return 0;
    }
    if (values.send !== undefined) {
      if (values.send !== "naive") throw new Error("--send takes naive: reflect idea <title> --send naive");
      const plan = await planHandoff(root, slugOf(title), values.workspace ?? null);
      for (const x of plan.sent) out(`already sent ${x.on} to ${x.workspace}:${x.raw}`);
      if (!values.yes) {
        out(`This goes to the Naive workspace at ${plan.workspace}, through naive capture:`);
        out("");
        for (const l of plan.text.split("\n")) out(`  ${l}`);
        out("");
        out(`Send it: reflect idea ${JSON.stringify(title)} --send naive${values.workspace ? ` --workspace ${JSON.stringify(values.workspace)}` : ""} --yes`);
        return 0;
      }
      out(`sent: ${await sendIdea(root, plan, runNaiveCapture, new Date())}`);
      return 0;
    }
    const inputs = async (xs: string[] | undefined) => Promise.all((xs ?? []).map(async (x) => ({ target: await resolveTarget(root, x), ...(values.how ? { how: values.how } : {}) })));
    const related = await Promise.all((values.related ?? []).map(async (r) => {
      const at = r.indexOf(":");
      const stance = at > 0 ? r.slice(0, at).trim() : "";
      if (!isStance(stance)) throw new Error(`--related is <stance>:<thing>; stance is one of ${STANCES.map((x) => x.id).join(", ")}`);
      if (!values.by) throw new Error("--related needs --by: who suggested it (you, your AI, a friend)");
      return { target: await resolveTarget(root, r.slice(at + 1)), stance: stance as Stance, by: values.by, ...(values.note ? { note: values.note } : {}) };
    }));
    const drop = await Promise.all((values.drop ?? []).map((x) => resolveTarget(root, x)));
    const i = await keepIdea(root, title, { ...(statement ? { statement } : {}), from: await inputs(values.from), maybe: await inputs(values.maybe), related, drop, ...(values.on ? { on: values.on } : {}) }, new Date());
    out(`kept: ideas/${i.id}.md; ${i.from.length} from, ${i.maybe.length} maybe from, ${i.related.length} recommendation(s)`);
    return 0;
  }
  if ((cmd === "need" || cmd === "passage") && layoutOf(config) !== "vault") throw new Error(`${cmd} needs a vault; this instance uses the older people/<id> layout`);
  if (cmd === "need" && rest[0]) {
    out(`kept: ${await keepNeed(root, rest[0], rest[1] ?? "", { provider: values.provider, runway: values.runway, note: values.note }, new Date())}`);
    return 0;
  }
  if (cmd === "passage" && rest[0]) {
    const p = await keepPassage(root, rest[0], rest[1] ?? "", values.on ?? today(), values.note, new Date());
    out(`kept: ${p.region} ${p.to} since ${p.on}`);
    return 0;
  }
  if (cmd === "legacy") {
    const p = legacyPrompt(rest[0]!);
    if (!p) throw new Error(`no such prompt; one of ${LEGACY_PROMPTS.map((x) => x.id).join(", ")}`);
    out(await journal.writeWithHead(textFrom(rest[1]), date as IsoDate, `legacy/${p.id}`, ["legacy: true", `prompt: ${JSON.stringify(p.text)}`]));
    return 0;
  }
  if (cmd === "release") {
    if (layoutOf(config) !== "vault") throw new Error("release needs a vault; this instance uses the older people/<id> layout");
    const [sub, what] = rest;
    const now = new Date();
    if (sub === "trustee") {
      if (!what || !values.contact) throw new Error("release trustee <name> --contact <how to reach them>");
      await addTrustee(root, what, values.contact, now);
    } else if (sub === "give") {
      if (!what || !values.to || !values.contact) throw new Error(`release give <legacy/... | ${LEGACY_ENTRIES}> --to <name> --contact <how to reach them>`);
      await addGift(root, what, values.to, values.contact, now);
    } else if (sub === "terms") {
      const terms: { confirmations?: number; wait_days?: number } = {};
      if (values.confirmations !== undefined) terms.confirmations = Number(values.confirmations);
      if (values.wait !== undefined) terms.wait_days = Number(values.wait);
      await setTerms(root, terms, now);
    } else if (sub) throw new Error("release takes trustee, give or terms");
    const plan = await readPlan(root);
    out(`trustees: ${plan.trustees.map((t) => t.name).join(", ") || "none yet"}`);
    out(`a release needs ${plan.confirmations} trustee confirmation(s), then ${plan.wait_days} day(s) of waiting in which you or any trustee can stop it`);
    for (const [to, files] of await previewRelease(root, paths, plan, home)) out(`to ${to}: ${files.join(", ") || "nothing yet"}`);
    const { problems, notes } = await planProblems(root, plan);
    for (const p of problems) out(`problem: ${p}`);
    for (const n of notes) out(`note: ${n}`);
    return problems.length ? 1 : 0;
  }
  if (cmd === "documents") {
    const [type, custody] = rest;
    if (!custody || !isCustody(custody)) throw new Error(`custody is one of ${CUSTODY.join(", ")}`);
    const c = await docs.check(type!, custody, { where: values.where, connection: values.connection });
    out(`checked: ${c.type} ${c.custody}`);
    return 0;
  }
  if (cmd === "connect" && rest[0]) {
    if (!values.name) throw new Error("connect needs --name");
    const kind = values.kind ?? "vault";
    if (!(kind === "vault" || kind === "issuer" || kind === "portal" || kind === "platform")) throw new Error("--kind is vault, issuer, portal or platform");
    const holds = (values.holds ?? "").split(",").map((h) => h.trim()).filter(Boolean);
    const c = await docs.connect(rest[0], values.name, kind as Connection["kind"], holds);
    out(`connected: ${c.name} holds ${c.holds.join(", ") || "nothing yet"}`);
    return 0;
  }
  if (cmd === "add" && rest[0]) {
    await journal.addPart(rest[0], textFrom(rest[1]));
    return 0;
  }
  if (cmd === "close" && rest[0]) {
    await journal.closeEntry(rest[0]);
    return 0;
  }
  out(USAGE);
  return 2;
}

if (import.meta.main) {
  main(process.argv.slice(2)).then(
    (code) => process.exit(code),
    (e: unknown) => {
      console.error(`reflect: ${(e as Error).message}`);
      process.exit(1);
    },
  );
}
