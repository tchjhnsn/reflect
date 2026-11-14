// Writes the catalogs Work browses (src/data): O*NET's occupations (jobs that exist), Wikidata's historical
// professions (jobs that have existed), Wikidata's fictional professions and companies (work imagined in fiction),
// and Y Combinator's company directory (companies, with the newest batch forming now). Run it again to refresh:
//   npx tsx scripts/work-catalogs.ts
// O*NET 29.0 Database by the U.S. Department of Labor, Employment and Training Administration, CC BY 4.0.
// Wikidata is CC0. The Y Combinator directory is read from yc-oss.github.io, which republishes YC's public pages.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const UA = { headers: { "user-agent": "reflect-app/1.0 (https://reflect-wheat-three.vercel.app)" } };
const out = (name: string, data: unknown) => writeFileSync(join(import.meta.dirname, "../src/data", name), JSON.stringify(data));

// O*NET: occupations with their description, core tasks, new tasks, interests, job zone, titles and related jobs.
const dir = mkdtempSync(join(tmpdir(), "onet-"));
writeFileSync(join(dir, "db.zip"), Buffer.from(await (await fetch("https://www.onetcenter.org/dl_files/database/db_29_0_text.zip", UA)).arrayBuffer()));
execFileSync("unzip", ["-q", "-o", join(dir, "db.zip"), "-d", dir]);
const tsv = (f: string) => readFileSync(join(dir, "db_29_0_text", f), "utf8").trim().split(/\r?\n/).slice(1).map((l) => l.split("\t").map((c) => c.trim()));
const occ = new Map<string, { code: string; title: string; about: string; zone?: number; tasks: string[]; emerging: string[]; interests: [string, number][]; titles: string[]; related: string[] }>();
for (const [code, title, about] of tsv("Occupation Data.txt")) occ.set(code!, { code: code!, title: title!, about: about!, tasks: [], emerging: [], interests: [], titles: [], related: [] });
for (const [code, zone] of tsv("Job Zones.txt")) { const o = occ.get(code!); if (o) o.zone = Number(zone); }
// Core tasks first; a job O*NET has not split into core and supplemental (software developers, among others) keeps its others.
const RANK: Record<string, number> = { Core: 0, "n/a": 1, Supplemental: 2 };
const tasks = new Map<string, [number, string][]>();
for (const [code, , task, type] of tsv("Task Statements.txt")) tasks.set(code!, [...(tasks.get(code!) ?? []), [RANK[type!] ?? 3, task!]]);
for (const [code, list] of tasks) { const o = occ.get(code); if (o) o.tasks = list.sort((a, b) => a[0] - b[0]).slice(0, 8).map(([, t]) => t); }
for (const [code, task, cat] of tsv("Emerging Tasks.txt")) { const o = occ.get(code!); if (o && cat === "New" && o.emerging.length < 6) o.emerging.push(task!); }
for (const [code, , name, scale, value] of tsv("Interests.txt")) { const o = occ.get(code!); if (o && scale === "OI" && ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"].includes(name!)) o.interests.push([name!, Number(value)]); }
for (const [code, title, shown] of tsv("Sample of Reported Titles.txt")) { const o = occ.get(code!); if (o && shown === "Y" && o.titles.length < 8) o.titles.push(title!); }
for (const [code, rel, tier] of tsv("Related Occupations.txt")) { const o = occ.get(code!); if (o && tier?.startsWith("Primary") && o.related.length < 8) o.related.push(rel!); }
out("onet.json", [...occ.values()].filter((o) => o.tasks.length).map((o) => ({ ...o, interests: o.interests.sort((a, b) => b[1] - a[1]).slice(0, 3).map(([n]) => n) })));

// Wikidata: historical professions and fictional professions and companies, each with its picture and, for fiction, the work it is from.
async function sparql(q: string): Promise<Record<string, { value: string }>[]> {
  const r = await fetch(`https://query.wikidata.org/sparql?format=json&query=${encodeURIComponent(q)}`, UA);
  return ((await r.json()) as { results: { bindings: Record<string, { value: string }>[] } }).results.bindings;
}
const NOT_WORK = new Set(["comfort woman", "slave", "concubine", "eunuch", "hetaera", "sex slave"]);
async function kind(cls: string, max: number) {
  const rows = await sparql(`SELECT ?o ?oLabel ?d ?links ?img ?fromLabel WHERE { ?o wdt:P31 wd:${cls} . ?o wikibase:sitelinks ?links . OPTIONAL { ?o wdt:P18 ?img } OPTIONAL { ?o wdt:P1441 ?from } OPTIONAL { ?o schema:description ?d FILTER(LANG(?d)="en") } ?o rdfs:label ?oLabel FILTER(LANG(?oLabel)="en") SERVICE wikibase:label { bd:serviceParam wikibase:language "en". ?from rdfs:label ?fromLabel } } ORDER BY DESC(?links)`);
  const seen = new Map<string, { id: string; name: string; about: string; image: string | null; from: string | null; links: number }>();
  for (const r of rows) {
    const id = r.o!.value.split("/").pop()!;
    const name = r.oLabel!.value;
    if (NOT_WORK.has(name.toLowerCase()) || /^Q\d+$/.test(name)) continue;
    const from = r.fromLabel?.value && !/^Q\d+$/.test(r.fromLabel.value) ? r.fromLabel.value : null;
    const had = seen.get(id);
    if (had) { if (!had.from && from) had.from = from; continue; }
    seen.set(id, { id, name, about: r.d?.value ?? "", image: r.img?.value.replace(/^http:/, "https:") ?? null, from, links: Number(r.links!.value) });
  }
  return [...seen.values()].slice(0, max);
}
out("wikidata-work.json", {
  past: (await kind("Q16335296", 600)).map((x) => ({ ...x, kind: "past" })),
  fiction: [...(await kind("Q17305127", 300)).map((x) => ({ ...x, kind: "fiction-job" })), ...(await kind("Q5446565", 500)).map((x) => ({ ...x, kind: "fiction-company" }))],
});

// Y Combinator: every company in the directory, with what it does, its batch, its status and its industries.
type Yc = { slug: string; name: string; one_liner: string | null; long_description: string | null; batch: string; status: string; industries: string[]; small_logo_thumb_url: string | null; website: string | null; team_size: number | null; all_locations: string | null; launched_at: number | null };
/** A description cut to about 600 characters at the end of a sentence, never mid-word. */
const clip = (t: string) => { const s = t.replace(/\s+/g, " ").trim(); if (s.length <= 600) return s; const cut = s.slice(0, 600); const end = cut.lastIndexOf(". "); return end > 200 ? cut.slice(0, end + 1) : `${cut.slice(0, cut.lastIndexOf(" "))}…`; };
const yc = (await (await fetch("https://yc-oss.github.io/api/companies/all.json", UA)).json()) as Yc[];
out("yc.json", yc.filter((c) => c.slug && c.name).map((c) => ({
  slug: c.slug, name: c.name, line: c.one_liner ?? "", about: clip(c.long_description ?? ""), batch: c.batch, status: c.status, industries: c.industries ?? [],
  logo: c.small_logo_thumb_url?.startsWith("https://") && !c.small_logo_thumb_url.includes("missing") ? c.small_logo_thumb_url : null,
  site: c.website ?? null, team: c.team_size ?? null, where: c.all_locations ?? null, launched: c.launched_at ?? null,
})));
console.log("wrote onet.json, wikidata-work.json, yc.json");
