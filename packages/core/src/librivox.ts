// LibriVox (librivox.org): public-domain books read aloud by volunteers, free, with an open API and no key.
// The recordings are kept by the Internet Archive. Open Library lists a work's LibriVox ids; this reads one
// recording and its chapters for the book page's player. Nothing is saved.
import type { Fetch } from "./openlibrary.ts";

export type Chapter = { n: number; title: string; url: string; seconds: number; reader?: string };
export type Audiobook = { id: string; title: string; url: string; seconds: number; chapters: Chapter[] };

const seconds = (t: string | undefined) => (t ?? "").split(":").map(Number).reduce((a, n) => (Number.isFinite(n) ? a * 60 + n : a), 0);
const https = (u: string | undefined) => (u && /^https?:\/\/(www\.)?archive\.org\//.test(u) ? u.replace(/^http:/, "https:") : null);

/** The first English recording among a work's LibriVox ids, with its chapters, or null when there is none. */
export async function librivoxBook(ids: readonly string[], fetcher: Fetch): Promise<Audiobook | null> {
  for (const id of ids.filter((x) => /^\d+$/.test(x)).slice(0, 4)) {
    const r = await fetcher(`https://librivox.org/api/feed/audiobooks?id=${id}&format=json&extended=1`);
    if (!r.ok) continue;
    const b = ((await r.json()) as { books?: { id?: string; title?: string; language?: string; totaltime?: string; url_librivox?: string; sections?: { section_number?: string; title?: string; listen_url?: string; playtime?: string; readers?: { display_name?: string }[] }[] }[] }).books?.[0];
    if (!b || b.language !== "English") continue;
    const chapters = (b.sections ?? []).flatMap((s, i) => {
      const url = https(s.listen_url);
      if (!url) return [];
      const reader = s.readers?.[0]?.display_name?.trim();
      return [{ n: Number(s.section_number) || i + 1, title: (s.title ?? `Part ${i + 1}`).trim().slice(0, 200), url, seconds: Number(s.playtime) || 0, ...(reader ? { reader } : {}) }];
    });
    if (!chapters.length) continue;
    return { id, title: (b.title ?? "").trim(), url: b.url_librivox?.startsWith("https://librivox.org/") ? b.url_librivox : `https://librivox.org/`, seconds: seconds(b.totaltime), chapters };
  }
  return null;
}
