import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { filingsFrom, type Company, type Filing, type Job, type WikiWork } from "@reflect/core/work-browse";

// The catalogs Work browses, read from src/data (written by scripts/work-catalogs.ts): O*NET's occupations,
// Wikidata's historical and fictional work, and Y Combinator's directory. Companies forming this week are read
// from SEC EDGAR's full-text search, kept six hours.

const cache = new Map<string, unknown>();
function load<T>(name: string, empty: T): T {
  if (cache.has(name)) return cache.get(name) as T;
  const file = [join(process.cwd(), "src/data", name), join(process.cwd(), "web/src/data", name)].find((f) => existsSync(f));
  const data = file ? (JSON.parse(readFileSync(file, "utf8")) as T) : empty;
  cache.set(name, data);
  return data;
}

export const jobs = (): Job[] => load<Job[]>("onet.json", []);
export const jobByCode = (code: string): Job | null => jobs().find((j) => j.code === code) ?? null;
const wiki = () => load<{ past: WikiWork[]; fiction: WikiWork[] }>("wikidata-work.json", { past: [], fiction: [] });
export const pastWork = (): WikiWork[] => wiki().past;
export const fictionWork = (): WikiWork[] => wiki().fiction;
export const wikiById = (id: string): WikiWork | null => [...wiki().past, ...wiki().fiction].find((w) => w.id === id) ?? null;
export const companies = (): Company[] => load<Company[]>("yc.json", []);
export const companyBySlug = (slug: string): Company | null => companies().find((c) => c.slug === slug) ?? null;

/** Companies that filed with the SEC in the last week: to raise money (Form D) or to go public (S-1). Funds are left out. */
export async function filingsThisWeek(form: "D" | "S-1", now = new Date()): Promise<Filing[]> {
  const end = now.toISOString().slice(0, 10);
  const start = new Date(now.getTime() - 7 * 86400_000).toISOString().slice(0, 10);
  try {
    const r = await fetch(`https://efts.sec.gov/LATEST/search-index?forms=${form}&dateRange=custom&startdt=${start}&enddt=${end}`, {
      // The SEC asks automated clients to name themselves in the user agent.
      headers: { "user-agent": "Reflect reflect-app@reflect-wheat-three.vercel.app" },
      next: { revalidate: 21600 },
    });
    if (!r.ok) return [];
    const j = (await r.json()) as { hits?: { hits?: Parameters<typeof filingsFrom>[0] } };
    return filingsFrom(j.hits?.hits ?? [], form);
  } catch {
    return [];
  }
}

/** A credit line for each catalog, shown where its entries are. */
export const CREDIT = {
  onet: { name: "O*NET 29.0 Database, U.S. Department of Labor, Employment and Training Administration (CC BY 4.0)", href: "https://www.onetcenter.org/database.html" },
  wikidata: { name: "Wikidata (CC0)", href: "https://www.wikidata.org" },
  yc: { name: "Y Combinator's company directory", href: "https://www.ycombinator.com/companies" },
  sec: { name: "SEC EDGAR", href: "https://efts.sec.gov/LATEST/search-index" },
} as const;
