#!/usr/bin/env node
// reflect: the command line over one person's instance in a folder. Node 22.18 or later runs it as is.
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { filesJournal } from "../core/src/files-journal.ts";
import { adopt, checkInstance, initInstance, readConfig } from "../core/src/instance.ts";
import type { IsoDate } from "../core/src/person.ts";

const USAGE = `reflect <command> [--dir <instance>]

  init --name <name>              start an instance in the folder
  write [text] [--date YYYY-MM-DD] write an entry (text from the argument or stdin); prints its id
  add <entry> [text]              add to an entry that is still open
  close <entry>                   close an entry
  list                            the entries, newest first
  check [--adopt]                 check the instance; --adopt records hand-written changes in memory`;

const today = (): IsoDate => new Date().toISOString().slice(0, 10) as IsoDate;

function textFrom(arg: string | undefined): string {
  const t = (arg ?? readFileSync(0, "utf8")).replace(/\r\n/g, "\n").trim();
  if (!t) throw new Error("nothing to write");
  return t;
}

export async function main(argv: string[], out: (s: string) => void = console.log): Promise<number> {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: { dir: { type: "string", default: "." }, name: { type: "string" }, date: { type: "string" }, adopt: { type: "boolean", default: false }, help: { type: "boolean", short: "h" } },
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
  const config = await readConfig(root);
  const journal = filesJournal(root, config.person);
  if (cmd === "list") {
    for (const e of await journal.listEntries()) out(`${e.id}  ${e.localDate}  ${e.closedAt ? "closed" : "open  "}  ${(e.parts[0]?.text ?? "").split("\n")[0]!.slice(0, 60)}`);
    return 0;
  }
  if (config.moved_to) throw new Error(`this instance moved to ${config.moved_to}; write there`);
  if (cmd === "write") {
    const date = values.date ?? today();
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error("--date is YYYY-MM-DD");
    out(await journal.createEntry(textFrom(rest[0]), date as IsoDate, null));
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
