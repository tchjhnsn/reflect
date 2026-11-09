import { readFileSync, writeFileSync } from "node:fs";

const source = new URL("../../thrivesight_core/nts/structure.json", import.meta.url);
const copy = new URL("../src/core/structure.json", import.meta.url);
const text = readFileSync(source, "utf8");

if (process.argv.includes("--check")) {
  if (readFileSync(copy, "utf8") !== text) {
    console.error("web/src/core/structure.json differs from the Lean export. Run `npm run sync`.");
    process.exit(1);
  }
} else {
  writeFileSync(copy, text);
}
