// Writes the two catalogs Reflect uses to know, for any book in a list, whether it can be read or heard in
// Reflect: Project Gutenberg's English texts (from its daily catalog file) and LibriVox's English recordings
// (from its API, 500 at a time). Each line is "id|title|author", lower case. Run it again to refresh them:
//   npx tsx scripts/free-catalogs.ts
import { writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

const UA = { headers: { "user-agent": "reflect-app" } };
const clean = (s: string) => s.toLowerCase().replace(/[|\n\r]+/g, " ").replace(/\s+/g, " ").trim();

/** One CSV row's fields, quotes and doubled quotes honoured. */
function* rows(csv: string): Generator<string[]> {
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < csv.length; i++) {
    const c = csv[i]!;
    if (quoted) {
      if (c === '"' && csv[i + 1] === '"') { cell += '"'; i++; }
      else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(cell); cell = ""; }
    else if (c === "\n") { row.push(cell); yield row; row = []; cell = ""; }
    else if (c !== "\r") cell += c;
  }
  if (cell || row.length) { row.push(cell); yield row; }
}

const gz = Buffer.from(await (await fetch("https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz", UA)).arrayBuffer());
const pg: string[] = [];
let header = true;
for (const r of rows(gunzipSync(gz).toString("utf8"))) {
  if (header) { header = false; continue; }
  const [id, type, , title, language, authors] = r;
  if (type !== "Text" || !language?.split(/;\s*/).includes("en") || !id || !title) continue;
  // "Austen, Jane, 1775-1817; ..." keeps each author's names without the dates.
  const by = (authors ?? "").split(";").map((a) => a.replace(/\d{1,4}\??\s*(BCE|BC|CE|AD)?-?\d{0,4}\??\s*(BCE|BC|CE|AD)?/g, "").replace(/\[[^\]]*\]/g, "").replace(/[,()]/g, " ")).join(" ");
  pg.push(`${id}|${clean(title).split(/\s*[;:]\s/)[0]!.slice(0, 90)}|${clean(by).slice(0, 60)}`);
}
writeFileSync(new URL("../src/data/gutenberg.txt", import.meta.url), pg.join("\n") + "\n");
console.log(`gutenberg: ${pg.length} English texts`);

const lv: string[] = [];
for (let offset = 0; ; offset += 500) {
  const r = await fetch(`https://librivox.org/api/feed/audiobooks?format=json&limit=500&offset=${offset}&fields=%7Bid,title,authors,language%7D`, UA);
  const body = (await r.json().catch(() => ({}))) as { books?: { id?: string; title?: string; language?: string; authors?: { first_name?: string; last_name?: string }[] }[] };
  if (!body.books?.length) break;
  for (const b of body.books) if (b.language === "English" && b.id && b.title) lv.push(`${b.id}|${clean(b.title)}|${clean((b.authors ?? []).map((a) => `${a.first_name ?? ""} ${a.last_name ?? ""}`).join(" "))}`);
}
writeFileSync(new URL("../src/data/librivox.txt", import.meta.url), lv.join("\n") + "\n");
console.log(`librivox: ${lv.length} English recordings`);
