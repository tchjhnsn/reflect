/**
 * Hearing a podcast inside Reflect. A show is found by its Apple Podcasts link (Apple's lookup gives its feed) or by
 * its feed itself; the feed lists each episode with its audio file, which Reflect plays in its own player, and a
 * transcript when the show publishes one (the Podcasting 2.0 transcript tag). Nothing is copied. Fetching is passed
 * in, so it runs anywhere.
 */
import { passagesFromCaptions, plain, type Passage } from "./courses";

export type Episode = { key: string; title: string; date: string | null; seconds: number | null; audio: string; about: string | null; transcript: { url: string; type: string } | null };
export type Show = { title: string; feed: string; episodes: Episode[] };
export type Fetch = (url: string) => Promise<string | null>;

/** A short id for an episode, stable across reads of the feed (FNV-1a of its guid, else its audio address). */
export function episodeKey(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  let g = 0xcbf29ce4;
  for (let i = s.length - 1; i >= 0; i--) { g ^= s.charCodeAt(i); g = Math.imul(g, 0x01000193) >>> 0; }
  return h.toString(36) + g.toString(36);
}

const tag = (xml: string, name: string) => {
  const m = new RegExp(`<${name}(?:\\s[^>]*)?>([\\s\\S]*?)</${name}>`, "i").exec(xml);
  if (!m) return null;
  const v = m[1]!.replace(/^\s*<!\[CDATA\[([\s\S]*?)\]\]>\s*$/, "$1");
  return plain(v) || null;
};
const attr = (el: string, name: string) => new RegExp(`\\s${name}\\s*=\\s*"([^"]*)"`, "i").exec(el)?.[1]?.replace(/&amp;/g, "&") ?? null;
const https = (u: string | null) => (u && /^https?:\/\/\S+$/i.test(u) ? u : null);

/** How long an episode is: "3600", "1:00:00" or "59:59". */
export function duration(v: string | null): number | null {
  if (!v) return null;
  if (/^\d+$/.test(v.trim())) return Number(v.trim());
  const m = /^(?:(\d+):)?(\d{1,2}):(\d{2})$/.exec(v.trim());
  return m ? Number(m[1] ?? 0) * 3600 + Number(m[2]) * 60 + Number(m[3]) : null;
}

/** A show's feed read into its episodes, newest first, up to 300. */
export function readFeed(xml: string, feed: string): Show {
  const head = xml.split(/<item[\s>]/i)[0] ?? "";
  const episodes: Episode[] = [];
  for (const m of xml.matchAll(/<item[\s>]([\s\S]*?)<\/item>/gi)) {
    const it = m[1]!;
    const enc = /<enclosure\b[^>]*>/i.exec(it)?.[0];
    const audio = https(enc ? attr(enc, "url") : null);
    const title = tag(it, "title");
    if (!audio || !title) continue;
    const tr = [...it.matchAll(/<podcast:transcript\b[^>]*>/gi)].map((t) => ({ url: https(attr(t[0], "url")), type: attr(t[0], "type") ?? "" })).filter((t) => t.url && /vtt|srt|text\/plain|html/i.test(t.type));
    const best = tr.find((t) => /vtt/i.test(t.type)) ?? tr.find((t) => /srt/i.test(t.type)) ?? tr[0];
    const date = tag(it, "pubDate");
    const d = date ? new Date(date) : null;
    episodes.push({
      key: episodeKey(tag(it, "guid") ?? audio), title, date: d && !Number.isNaN(d.getTime()) ? d.toISOString().slice(0, 10) : null,
      seconds: duration(tag(it, "itunes:duration")), audio: audio.replace(/^http:/, "https:"),
      about: (tag(it, "itunes:summary") ?? tag(it, "description"))?.slice(0, 2000) ?? null,
      transcript: best ? { url: best.url!, type: best.type } : null,
    });
    if (episodes.length >= 300) break;
  }
  episodes.sort((a, b) => (b.date ?? "").localeCompare(a.date ?? ""));
  return { title: tag(head, "title") ?? "", feed, episodes };
}

/** The feed of a show from its Apple Podcasts link, or the link itself when it is a feed. */
export async function feedOf(raw: string, f: Fetch): Promise<string | null> {
  let u: URL;
  try { u = new URL(raw); } catch { return null; }
  if (u.hostname === "podcasts.apple.com") {
    const id = /\/id(\d+)/.exec(u.pathname)?.[1];
    if (!id) return null;
    const j = await f(`https://itunes.apple.com/lookup?id=${id}&entity=podcast`);
    try { return https((JSON.parse(j ?? "{}") as { results?: { feedUrl?: string }[] }).results?.[0]?.feedUrl ?? null); } catch { return null; }
  }
  return u.protocol === "https:" || u.protocol === "http:" ? raw : null;
}

/** A show and its episodes, from its link. */
export async function showOf(raw: string | null, f: Fetch): Promise<Show | null> {
  if (!raw) return null;
  const feed = await feedOf(raw, f);
  const xml = feed ? await f(feed) : null;
  if (!xml || !/<rss|<channel/i.test(xml)) return null;
  return readFeed(xml, feed!);
}

/** An episode's published transcript as passages. */
export async function episodeTranscript(e: Episode, f: Fetch): Promise<Passage[]> {
  if (!e.transcript) return [];
  const t = await f(e.transcript.url);
  if (!t) return [];
  if (/vtt|srt/i.test(e.transcript.type)) return passagesFromCaptions(t);
  const text = /html/i.test(e.transcript.type) ? t.replace(/<\/p>/gi, "\n").replace(/<[^>]+>/g, "") : t;
  return text.split(/\n\s*\n|\n/).map((x) => x.trim()).filter(Boolean).slice(0, 2000).map((x) => ({ at: null, heading: false, who: null, text: x }));
}
