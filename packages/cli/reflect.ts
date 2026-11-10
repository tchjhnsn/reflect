#!/usr/bin/env node
// reflect: the command line over one person's instance in a folder. Node 22.18 or later runs it as is.
import { readFileSync } from "node:fs";
import { readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseBundle } from "../core/src/bundle.ts";
import { keepDraft, makeDraft, type Draft } from "../core/src/draft.ts";
import { parseArgs } from "node:util";
import { filesJournal } from "../core/src/files-journal.ts";
import { adopt, checkInstance, initInstance, layoutOf, readConfig } from "../core/src/instance.ts";
import { recordPaths } from "../core/src/layout.ts";
import type { IsoDate } from "../core/src/person.ts";
import { readSalience } from "../core/src/reader.ts";
import { cleanNoun, isSparkForm, pickNoun, sparkPrompt, SPARK_FORMS, type SparkForm } from "../core/src/spark.ts";

const USAGE = `reflect <command> [--dir <instance>]

  init --name <name>              start a vault in the folder: journal, library, ventures of a
                                  lifetime (household, work, community, friendship), telos, intents
  write [text] [--date YYYY-MM-DD] write an entry (text from the argument or stdin); prints its id
  spark [--minutes 1|5|10|15 | --sentence]
                                  a noun to write about, and how long
  spark <noun> [text] [--minutes N | --sentence] [--date YYYY-MM-DD]
                                  keep what you wrote on that noun (text from the argument or stdin);
                                  the noun becomes a concept note in concepts/
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
