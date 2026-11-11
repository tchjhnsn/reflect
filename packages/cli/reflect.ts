#!/usr/bin/env node
// reflect: the command line over one person's instance in a folder. Node 22.18 or later runs it as is.
import { readFileSync } from "node:fs";
import { readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseBundle } from "../core/src/bundle.ts";
import { keepDraft, makeDraft, type Draft } from "../core/src/draft.ts";
import { parseArgs } from "node:util";
import { filesJournal } from "../core/src/files-journal.ts";
import { adopt, checkInstance, initInstance, layoutOf, readConfig, upgradeVault } from "../core/src/instance.ts";
import { addGift, addTrustee, LEGACY_ENTRIES, planProblems, previewRelease, readPlan, setTerms } from "../core/src/release.ts";
import { EVIDENCE_KINDS, keepResource, keepTopic, learnMap, RESOURCE_STATUS, type MapRow, type ResourceStatus } from "../core/src/learning.ts";
import { keepNeed, keepPassage, LEGACY_PROMPTS, legacyPrompt } from "../core/src/lifetime.ts";
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
  learn                           what you have learned, aspire to learn and need to learn, and what
                                  you could learn next
  learn aspire <topic> [--requires a,b] [--part-of x] [--same-as wikidata:Q123]
  learn need <topic> --for <intent, venture, need or role> [--requires a,b] [--same-as ...]
  learn evidence <topic> <said|studied|practiced|made|assessed|certified|vouched|taught>
                [--link <url or record>] [--by <who>] [--note <text>] [--on YYYY-MM[-DD]]
  learn resource <name> [--kind video|channel|article|book|course|podcast|paper|lab|site]
                [--url <link | isbn:N | doi:X>] [--topics a,b] [--by <author>] [--platform <id>]
                [--status saved|started|finished|dropped]
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

export async function main(argv: string[], out: (s: string) => void = console.log): Promise<number> {
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
      link: { type: "string" },
      by: { type: "string" },
      url: { type: "string" },
      topics: { type: "string" },
      platform: { type: "string" },
      status: { type: "string" },
      out: { type: "string" },
      "moving-to": { type: "string" },
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
  const paths = recordPaths(layoutOf(config), config.person);
  const journal = filesJournal(root, config.person, { layout: layoutOf(config) });
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
  const reflDir = join(root, paths.reflections);
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
      out(`draft: ${join(paths.reflections, `${id}.draft.json`)}`);
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
    const [sub, name, kind] = rest;
    const now = new Date();
    const list = (v: string | undefined) => (v ?? "").split(",").map((x) => x.trim()).filter(Boolean);
    const where = { requires: list(values.requires), part_of: list(values["part-of"]), ...(values["same-as"] ? { same_as: values["same-as"] } : {}) };
    if (sub === "aspire" || sub === "need") {
      if (!name) throw new Error(`learn ${sub} <topic>`);
      const t = await keepTopic(root, name, { ...where, want: sub, ...(values.for ? { for: values.for } : {}) }, now);
      out(`kept: learn/topics/${t.id}.md ${sub}${t.for ? ` for ${t.for}` : ""}`);
      return 0;
    }
    if (sub === "evidence") {
      if (!name || !kind) throw new Error(`learn evidence <topic> <${EVIDENCE_KINDS.map((k) => k.id).join("|")}>`);
      const t = await keepTopic(root, name, where, now, { kind: kind as MapRow["evidence"][number]["kind"], ...(values.link ? { link: values.link } : {}), ...(values.by ? { by: values.by } : {}), ...(values.note ? { note: values.note } : {}), ...(values.on ? { on: values.on } : {}) });
      out(`kept: learn/topics/${t.id}.md evidence ${kind}`);
      return 0;
    }
    if (sub === "resource") {
      if (!name) throw new Error("learn resource <name> --kind <kind> [--url <link>] [--topics a,b]");
      if (values.status && !(RESOURCE_STATUS as readonly string[]).includes(values.status)) throw new Error(`--status is one of ${RESOURCE_STATUS.join(", ")}`);
      const r = await keepResource(root, name, { ...(values.kind ? { kind: values.kind } : {}), ...(values.url ? { url: values.url } : {}), ...(values.by ? { by: values.by } : {}), ...(values.platform ? { platform: values.platform } : {}), topics: list(values.topics), ...(values.status ? { status: values.status as ResourceStatus } : {}) }, now);
      out(`kept: learn/resources/${r.id}.md ${r.status}`);
      return 0;
    }
    if (sub) throw new Error("learn takes aspire, need, evidence or resource");
    const m = await learnMap(root);
    const res = (r: MapRow) => (r.resources.length ? `; resources: ${r.resources.map((x) => `${x.id} (${x.status})`).join(", ")}` : "");
    const ready = (r: MapRow) => (r.blockedBy.length ? `first learn ${r.blockedBy.join(", ")}` : "ready");
    out("Learned");
    for (const r of m.learned) out(`  ${r.topic.id}  ${r.standing}: ${[...new Set(r.evidence.map((e) => e.kind))].join(", ")}`);
    if (!m.learned.length) out("  nothing yet; add evidence: reflect learn evidence <topic> <kind>");
    out("Aspire");
    for (const r of m.aspire) out(`  ${r.topic.id}  ${ready(r)}${res(r)}`);
    if (!m.aspire.length) out("  nothing yet: reflect learn aspire <topic>");
    out("Need");
    for (const r of m.need) out(`  ${r.topic.id}  for ${r.topic.for}; ${ready(r)}${res(r)}`);
    if (!m.need.length) out("  nothing yet: reflect learn need <topic> --for <what needs it>");
    out("Possible next");
    for (const r of m.possible) out(`  ${r.topic.id}  ${r.topic.requires.length ? "you have learned what it requires" : `a first step toward ${m.aspire.concat(m.need).filter((w) => w.topic.requires.includes(r.topic.id)).map((w) => w.topic.id).join(", ") || "what needs it"}`}`);
    if (!m.possible.length) out("  nothing your map names yet; the rest is the whole internet");
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
    for (const [to, files] of await previewRelease(root, paths, plan)) out(`to ${to}: ${files.join(", ") || "nothing yet"}`);
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
