/**
 * Reading the files other services export into learn items, for the person to pick from before anything is kept:
 * Letterboxd's export (watched.csv or diary.csv, films with the day watched), a podcast subscription list (OPML,
 * which every podcast app exports), YouTube's watch history from Google Takeout (watch-history.json), Spotify's
 * listening history (StreamingHistory*.json or the extended Streaming_History_Audio_*.json: music by most
 * listened, podcasts by show), a Steam library (the games list copied from a profile, one per line) and any plain
 * list, one per line. Nothing is fetched and nothing is kept here; each parser gives candidates only.
 */
import type { ItemStatus, Medium } from "./ways";

export type Candidate = { medium: Medium; title: string; by?: string; url?: string; status: ItemStatus; on?: string; note?: string };
export const IMPORT_KINDS = [
  { id: "letterboxd", label: "Letterboxd (watched.csv or diary.csv)", accept: ".csv" },
  { id: "opml", label: "Podcast subscriptions (OPML)", accept: ".opml,.xml" },
  { id: "youtube", label: "YouTube watch history (Google Takeout watch-history.json)", accept: ".json" },
  { id: "spotify", label: "Spotify listening history (StreamingHistory .json)", accept: ".json" },
  { id: "steam", label: "Steam games, one per line", accept: ".txt" },
  { id: "lines", label: "A list, one per line", accept: ".txt" },
] as const;
export type ImportKind = (typeof IMPORT_KINDS)[number]["id"];
export const MAX_CANDIDATES = 500;

const str = (v: unknown): string | undefined => (typeof v === "string" && v.trim() ? v.trim() : undefined);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" ? (v as Record<string, unknown>) : {});
const day = (v: string | undefined) => (v && /^\d{4}-\d{2}-\d{2}/.test(v) ? v.slice(0, 10) : undefined);
const https = (u: string | undefined) => (u && /^https?:\/\/\S+$/i.test(u) ? u : undefined);

/** RFC 4180 CSV rows: quoted fields may hold commas, doubled quotes and line breaks. */
export function csvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i]!;
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((x) => x !== "")) rows.push(row);
      row = [];
    } else field += c;
  }
  row.push(field);
  if (row.some((x) => x !== "")) rows.push(row);
  return rows;
}

/** Letterboxd: Date, Name, Year, Letterboxd URI (watched.csv), or with Watched Date (diary.csv). */
export function fromLetterboxd(text: string): Candidate[] {
  const [head, ...rows] = csvRows(text.replace(/^﻿/, ""));
  if (!head) return [];
  const col = (n: string) => head.findIndex((h) => h.trim().toLowerCase() === n);
  const name = col("name"), year = col("year"), uri = col("letterboxd uri"), watched = col("watched date"), date = col("date");
  if (name < 0) throw new Error("This is not a Letterboxd export: it has no Name column.");
  const seen = new Set<string>();
  const out: Candidate[] = [];
  for (const r of rows) {
    const title = str(r[name]);
    if (!title) continue;
    const y = str(r[year]);
    const key = `${title}|${y ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const on = day(str(r[watched]) ?? str(r[date]));
    out.push({ medium: "film", title, ...(y ? { note: y } : {}), ...(https(str(r[uri])) ? { url: https(str(r[uri]))! } : {}), status: "done", ...(on ? { on } : {}) });
  }
  return out;
}

const unxml = (s: string) => s.replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n))).replace(/&amp;/g, "&");

/** OPML: each outline with an xmlUrl is a podcast; its text or title is the show's name. */
export function fromOpml(text: string): Candidate[] {
  const out: Candidate[] = [];
  for (const m of text.matchAll(/<outline\b([^>]*)>/gi)) {
    const attrs = new Map([...m[1]!.matchAll(/([\w:]+)\s*=\s*"([^"]*)"/g)].map((a) => [a[1]!.toLowerCase(), unxml(a[2]!)]));
    if (!attrs.get("xmlurl")) continue;
    const title = str(attrs.get("text")) ?? str(attrs.get("title"));
    const site = https(str(attrs.get("htmlurl")));
    if (title) out.push({ medium: "podcast", title, ...(site ? { url: site } : {}), status: "doing" });
  }
  if (!out.length && !/<opml/i.test(text)) throw new Error("This is not an OPML file.");
  return out;
}

/** Google Takeout's watch-history.json: "Watched <title>", the channel under subtitles, newest first. Each video once. */
export function fromYoutube(text: string): Candidate[] {
  const rows = JSON.parse(text) as unknown;
  if (!Array.isArray(rows)) throw new Error("This is not YouTube's watch-history.json.");
  const seen = new Set<string>();
  const out: Candidate[] = [];
  for (const r of rows.map(obj)) {
    const url = https(str(r.titleUrl));
    const title = str(r.title)?.replace(/^Watched\s+/, "");
    if (!url || !title || seen.has(url) || title === url) continue;
    seen.add(url);
    const channel = str(obj((r.subtitles as unknown[] | undefined)?.[0]).name);
    const on = day(str(r.time));
    out.push({ medium: "lecture", title, ...(channel ? { by: channel } : {}), url: url.replace("music.youtube.com", "www.youtube.com"), status: "done", ...(on ? { on } : {}) });
  }
  return out;
}

/** Spotify: music grouped by track and podcasts by show, most listened first, with minutes in the note. */
export function fromSpotify(text: string): Candidate[] {
  const rows = JSON.parse(text) as unknown;
  if (!Array.isArray(rows)) throw new Error("This is not a Spotify streaming history file.");
  const tally = new Map<string, Candidate & { ms: number }>();
  for (const r of rows.map(obj)) {
    const ms = Number(r.msPlayed ?? r.ms_played ?? 0);
    const show = str(r.episode_show_name) ?? (str(r.episodeName) ? str(r.showName) : undefined);
    const track = str(r.trackName) ?? str(r.master_metadata_track_name);
    const artist = str(r.artistName) ?? str(r.master_metadata_album_artist_name);
    const on = day(str(r.endTime) ?? str(r.ts));
    const c: Candidate | null = show ? { medium: "podcast", title: show, status: "doing" } : track ? { medium: "music", title: track, ...(artist ? { by: artist } : {}), status: "done" } : null;
    if (!c) continue;
    const key = `${c.medium}|${c.title}|${c.by ?? ""}`;
    const t = tally.get(key) ?? { ...c, ms: 0 };
    t.ms += Number.isFinite(ms) ? ms : 0;
    if (on && (!t.on || on > t.on)) t.on = on;
    tally.set(key, t);
  }
  return [...tally.values()].sort((a, b) => b.ms - a.ms).map(({ ms, ...c }) => ({ ...c, note: `${Math.round(ms / 60000)} min` }));
}

/** A list, one per line: "Title" or "Title | By". Lines starting with # are skipped. */
export function fromLines(text: string, medium: Medium, status: ItemStatus = "done"): Candidate[] {
  const seen = new Set<string>();
  return text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !l.startsWith("#")).map((l) => {
    const [title, ...rest] = l.split("|");
    return { medium, title: title!.trim().slice(0, 300), ...(rest.join("|").trim() ? { by: rest.join("|").trim().slice(0, 200) } : {}), status };
  }).filter((c) => c.title && !seen.has(c.title.toLowerCase()) && seen.add(c.title.toLowerCase()));
}

/** Reads a file of one kind into candidates, at most MAX_CANDIDATES. */
export function readImport(kind: ImportKind, text: string, medium: Medium = "lecture"): Candidate[] {
  const all = kind === "letterboxd" ? fromLetterboxd(text) : kind === "opml" ? fromOpml(text) : kind === "youtube" ? fromYoutube(text) : kind === "spotify" ? fromSpotify(text) : kind === "steam" ? fromLines(text, "game") : fromLines(text, medium);
  return all.slice(0, MAX_CANDIDATES);
}

/** Leaves out what the person already keeps: the same medium and title, ignoring case and punctuation. */
export function notKept(cands: readonly Candidate[], kept: readonly { medium: string; title: string }[]): Candidate[] {
  const n = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  const have = new Set(kept.map((k) => `${k.medium}|${n(k.title)}`));
  return cands.filter((c) => !have.has(`${c.medium}|${n(c.title)}`));
}
