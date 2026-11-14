/**
 * Browse beyond books: what to find for each medium, from free catalogs, and whether it can be done inside Reflect.
 * Courses come from Open Yale Courses' catalog and MIT Learn's catalog of OpenCourseWare (courses with lecture videos
 * only, since those play here); art from the Art Institute of Chicago's public-domain highlights; podcasts from
 * Apple's podcast search. Fetching is passed in, so it runs anywhere.
 */
import { courseSite, plain } from "./courses";
import { embedOf } from "./media";
import { searchMedium, type Found } from "./lookup";
import { mediumOf, type Medium, type Way } from "./ways";

export type BrowseMedium = "books" | Medium;
/** The media Browse offers, in its menu's order (articles are found by their link, not browsed). */
export const BROWSE_MEDIA: readonly { id: BrowseMedium; label: string }[] = [
  { id: "books", label: "Books" }, { id: "course", label: "Courses" }, { id: "lecture", label: "Lectures" }, { id: "debate", label: "Debates" },
  { id: "podcast", label: "Podcasts" }, { id: "theatre", label: "Plays" }, { id: "film", label: "Films" }, { id: "series", label: "Series" },
  { id: "music", label: "Music" }, { id: "art", label: "Art" }, { id: "game", label: "Games" },
];
export const isBrowseMedium = (m: unknown): m is BrowseMedium => typeof m === "string" && BROWSE_MEDIA.some((x) => x.id === m);

/** The way a medium belongs to, or rest for music, art and games until the person says what they taught. */
export const browseWay = (m: BrowseMedium): Way | "leisure" => (m === "books" ? "teaching" : mediumOf(m).lesson ? "leisure" : mediumOf(m).way);

/**
 * Whether Reflect can play or show it inside the app: a course from Open Yale or MIT OpenCourseWare, a podcast from
 * its feed, a recording on YouTube, Vimeo or the Internet Archive, a work in the Art Institute of Chicago, a play
 * whose text is free (said by the caller).
 */
export function playsHere(f: { medium: Medium; url?: string | null | undefined; image?: string | null | undefined }): boolean {
  if (f.medium === "course") return courseSite(f.url) !== null;
  if (f.medium === "podcast") return Boolean(f.url && /podcasts\.apple\.com\/.+\/id\d+/.test(f.url));
  if (f.medium === "art") return Boolean(f.url?.includes("artic.edu/artworks/") && f.image);
  const e = embedOf(f.url);
  return e !== null && (e.kind === "youtube" || /vimeo|archive/i.test(e.site));
}

const obj = (v: unknown) => (v && typeof v === "object" ? (v as Record<string, unknown>) : {});
const arr = (v: unknown) => (Array.isArray(v) ? v : []);
const str = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : undefined);
/** Fetching as text, null when it fails. */
export type Fetch = (url: string) => Promise<string | null>;
const json = async (f: Fetch, url: string) => { const t = await f(url); try { return t ? JSON.parse(t) as unknown : null; } catch { return null; } };

/** Open Yale's course catalog page read into courses. */
export function oycCatalog(html: string): Found[] {
  const out: Found[] = [];
  for (const row of html.match(/<tr[\s\S]*?<\/tr>/g) ?? []) {
    const t = /views-field-title-1[^>]*>\s*<a href="(\/[^"]+)"[^>]*>([\s\S]*?)<\/a>/.exec(row);
    if (!t) continue;
    const by = /field-course-professor-last-name[^>]*>([\s\S]*?)<\/td>/.exec(row)?.[1];
    const code = /field-course-number[^>]*>\s*<a[^>]*>([\s\S]*?)<\/a>/.exec(row)?.[1];
    const title = plain(t[2]!).replace(/\s*\((\d{4})\)$/, "");
    out.push({ medium: "course", title: code ? `${title} (${plain(code)})` : title, ...(by && plain(by) ? { by: plain(by) } : {}), url: `https://oyc.yale.edu${t[1]}`, source: "Open Yale Courses" });
  }
  return out;
}

/** MIT Learn's search of OpenCourseWare courses with lecture videos, most viewed first when there is no search. */
export async function mitCourses(q: string, f: Fetch, limit = 24): Promise<Found[]> {
  const d = obj(await json(f, `https://api.learn.mit.edu/api/v1/learning_resources_search/?platform=ocw&resource_type=course&course_feature=Lecture%20Videos&limit=${limit}${q ? `&q=${encodeURIComponent(q)}` : "&sortby=-views"}`));
  return arr(d.results).map(obj).filter((r) => str(r.title) && courseSite(str(r.url))).map((r) => {
    const people = arr(obj(arr(r.runs)[0]).instructors).map((i) => str(obj(i).full_name)?.replace(/^(Prof\.|Dr\.)\s+/, "")).filter(Boolean).slice(0, 3);
    const image = str(obj(r.image).url);
    return { medium: "course" as const, title: str(r.title)!, ...(people.length ? { by: people.join(", ") } : {}), url: str(r.url)!, ...(image?.startsWith("https://") ? { image } : {}), source: "MIT OpenCourseWare" };
  });
}

/** Courses to take in Reflect: Open Yale's whole catalog (filtered by the search) and MIT's courses with videos. */
export async function findCourses(q: string, f: Fetch, oycHtml: string | null): Promise<Found[]> {
  const words = q.toLowerCase().split(/\s+/).filter(Boolean);
  const yale = (oycHtml ? oycCatalog(oycHtml) : []).filter((c) => words.every((w) => `${c.title} ${c.by ?? ""}`.toLowerCase().includes(w)));
  const mit = await mitCourses(q, f).catch(() => []);
  // With no search, MIT's most viewed (with pictures) and Yale's alternate, so neither catalog's order buries the other.
  if (q) return [...yale, ...mit];
  const out: Found[] = [];
  for (let i = 0; i < Math.max(yale.length, mit.length); i++) { if (mit[i]) out.push(mit[i]!); if (yale[i]) out.push(yale[i]!); }
  return out;
}

/** The Art Institute of Chicago's best-known works in the public domain, or a search of them. */
export async function findArt(q: string, f: Fetch, limit = 24): Promise<Found[]> {
  const filter = "query%5Bbool%5D%5Bmust%5D%5B%5D%5Bterm%5D%5Bis_public_domain%5D=true" + (q ? "" : "&query%5Bbool%5D%5Bmust%5D%5B%5D%5Bterm%5D%5Bis_boosted%5D=true");
  const d = obj(await json(f, `https://api.artic.edu/api/v1/artworks/search?${filter}${q ? `&q=${encodeURIComponent(q)}` : ""}&limit=${limit}&fields=id,title,artist_title,date_display,image_id`));
  return arr(d.data).map(obj).filter((a) => str(a.title) && str(a.image_id)).map((a) => ({
    medium: "art" as const, title: str(a.title)!, ...(str(a.artist_title) ? { by: str(a.artist_title)! } : {}), url: `https://www.artic.edu/artworks/${Number(a.id)}`,
    image: `https://www.artic.edu/iiif/2/${str(a.image_id)}/full/400,/0/default.jpg`, ...(str(a.date_display) ? { year: str(a.date_display)! } : {}), source: "Art Institute of Chicago",
  }));
}

/** What to find for a medium (books are browsed apart): its own catalog, or the free catalog search. */
export async function findMedium(m: Medium, q: string, f: Fetch, extra: { oycHtml?: string | null } = {}): Promise<{ found: Found[]; error?: string }> {
  try {
    if (m === "course") return { found: await findCourses(q, f, extra.oycHtml ?? null) };
    if (m === "art") return { found: await findArt(q, f) };
    if (!q) return { found: [] };
    return await searchMedium(m, q, async (u) => { const t = await f(u); return { ok: t !== null, status: t === null ? 502 : 200, json: async () => JSON.parse(t ?? "null") as unknown }; });
  } catch {
    return { found: [], error: "The catalog did not answer. Try again in a moment." };
  }
}

/** The most watched feature films in the Internet Archive's public-domain collection, which play in Reflect. */
export async function freeFilmsToWatch(f: Fetch, limit = 24): Promise<Found[]> {
  const d = obj(await json(f, `https://archive.org/advancedsearch.php?output=json&rows=${limit}&sort%5B%5D=downloads+desc&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=year&fl%5B%5D=creator&q=${encodeURIComponent("collection:(feature_films) AND mediatype:(movies)")}`));
  return arr(obj(d.response).docs).map(obj).filter((x) => str(x.identifier) && str(x.title)).map((x) => {
    const id = str(x.identifier)!;
    const creator = Array.isArray(x.creator) ? str(x.creator[0]) : str(x.creator);
    const year = /\d{4}/.exec(String(x.year ?? ""))?.[0];
    return { medium: "film" as const, title: str(x.title)!, ...(creator ? { by: creator } : {}), url: `https://archive.org/details/${id}`, image: `https://archive.org/services/img/${id}`, ...(year ? { year } : {}), source: "Internet Archive" };
  });
}
