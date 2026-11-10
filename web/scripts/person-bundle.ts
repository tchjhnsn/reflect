// Move one person's journal between our Postgres and a bundle file (see packages/core/src/bundle.ts).
//   pnpm bundle export --user <id> --out <file>
//   pnpm bundle import <file> --user <id>      (into an empty journal only)
// Needs DATABASE_URL and REFLECT_DATA_KEY for the branch it runs against.
import { readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { parseBundle } from "../../packages/core/src/bundle.ts";
import { db } from "../src/db/pool";
import { exportJournal, importJournal } from "../src/server/journal-store";

const { values, positionals } = parseArgs({ allowPositionals: true, options: { user: { type: "string" }, out: { type: "string" } } });
const [cmd, file] = positionals;
if (!values.user || !(cmd === "export" ? values.out : cmd === "import" && file)) {
  console.error("usage: bundle export --user <id> --out <file> | bundle import <file> --user <id>");
  process.exit(2);
}
try {
  if (cmd === "export") {
    await writeFile(values.out!, `${JSON.stringify(await exportJournal(values.user), null, 2)}\n`);
    console.log(`exported ${values.user} to ${values.out}`);
  } else {
    await importJournal(values.user, parseBundle(await readFile(file!, "utf8")));
    console.log(`imported ${file} into ${values.user}`);
  }
} finally {
  await db().end();
}
