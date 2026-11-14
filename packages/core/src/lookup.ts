/**
 * Finding a learn item instead of typing it: from a pasted link, or by searching a free catalog for its medium.
 * Links are read through each site's own oEmbed or lookup endpoint (YouTube, Vimeo, Spotify, SoundCloud, Apple
 * Podcasts, Steam, the Internet Archive, the Art Institute of Chicago), never by fetching an arbitrary page. Searches go to:
 * films and theatre in Wikidata (and films free to watch in the Internet Archive), series in TVmaze, games in the Steam store, podcasts in Apple's podcast search,
 * music in MusicBrainz (covers from the Cover Art Archive) and art in the Art Institute of Chicago's open
 * collection. Lectures, courses, debates and discussions are found by their link. A catalog that fails gives
 * nothing back with its reason; nothing is guessed.
 */
import { isMedium, type Medium } from "./ways";

export type Fetch = (url: string) => Promise<{ ok: boolean; status: number; json(): Promise<unknown> }>;
export type Found = { medium: Medium; title: string; by?: string; url?: string; image?: string; year?: string; source: string };
export type Searched = { found: Found[]; error?: string };

const str = (v: unknown): string | undefined => (typeof v === "string" && v.trim() ? v.trim() : undefined);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" ? (v as Record<string, unknown>) : {});
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const enc = encodeURIComponent;
const https = (u: string | undefined) => (u && /^https:\/\/\S+$/i.test(u) ? u : undefined);

async function getJson(f: Fetch, url: string): Promise<unknown> {
  const r = await f(url);
  if (!r.ok) throw new Error(`${new URL(url).hostname} answered ${r.status}`);
  return r.json();
}

/** What kind of link this is: its medium and how to read it, or null for a site Reflect does not read. */
export function linkKind(raw: string): { medium: Medium; site: string; read: string } | null {
  let u: URL;
  try {
    u = new URL(raw.trim());
  } catch {
    return null;
  }
  if (u.protocol !== "https:" && u.protocol !== "http:") return null;
  const h = u.hostname.replace(/^(www|m|music)\./, "");
  const link = enc(u.toString());
  if (h === "youtube.com" || h === "youtu.be") return { medium: "lecture", site: "YouTube", read: `https://www.youtube.com/oembed?format=json&url=${link}` };
  if (h === "vimeo.com") return { medium: "lecture", site: "Vimeo", read: `https://vimeo.com/api/oembed.json?url=${link}` };
  if (h === "open.spotify.com") return { medium: /^\/(show|episode)\//.test(u.pathname) ? "podcast" : "music", site: "Spotify", read: `https://open.spotify.com/oembed?url=${link}` };
  if (h === "soundcloud.com") return { medium: "music", site: "SoundCloud", read: `https://soundcloud.com/oembed?format=json&url=${link}` };
  const apple = /\/id(\d+)/.exec(u.pathname);
  if (h === "podcasts.apple.com" && apple) return { medium: "podcast", site: "Apple Podcasts", read: `https://itunes.apple.com/lookup?id=${apple[1]}` };
  const steam = /^\/app\/(\d+)/.exec(u.pathname);
  if (h === "store.steampowered.com" && steam) return { medium: "game", site: "Steam", read: `https://store.steampowered.com/api/appdetails?filters=basic&appids=${steam[1]}` };
  const ia = /^\/(?:details|embed)\/([\w.-]+)/.exec(u.pathname);
  if (h === "archive.org" && ia) return { medium: "film", site: "Internet Archive", read: `https://archive.org/metadata/${ia[1]}` };
  const aic = /^\/artworks\/(\d+)/.exec(u.pathname);
  if (h === "artic.edu" && aic) return { medium: "art", site: "Art Institute of Chicago", read: `https://api.artic.edu/api/v1/artworks/${aic[1]}?fields=id,title,artist_title,date_display,image_id` };
  return null;
}

const aicImage = (id: unknown) => (str(id) ? `https://www.artic.edu/iiif/2/${str(id)}/full/400,/0/default.jpg` : undefined);

/** Reads a link into an item: its medium, title, maker and picture, from the site's own endpoint. */
export async function lookLink(raw: string, f: Fetch): Promise<{ found: Found | null; error?: string }> {
  const k = linkKind(raw);
  if (!k) return { found: null, error: "Reflect reads links from YouTube, Vimeo, Spotify, SoundCloud, Apple Podcasts, Steam, the Internet Archive and the Art Institute of Chicago. Fill in the rest by hand." };
  const url = raw.trim();
  try {
    const d = obj(await getJson(f, k.read));
    if (k.site === "Apple Podcasts") {
      const r = obj(arr(d.results)[0]);
      const title = str(r.collectionName) ?? str(r.trackName);
      return { found: title ? { medium: "podcast", title, ...by(str(r.artistName)), url, ...img(https(str(r.artworkUrl600))), source: k.site } : null };
    }
    if (k.site === "Steam") {
      const app = obj(Object.values(d)[0]);
      const data = obj(app.data);
      const title = str(data.name);
      return { found: title ? { medium: "game", title, ...by(arr(data.developers).map(str).filter(Boolean).join(", ") || undefined), url, ...img(https(str(data.header_image))), source: k.site } : null };
    }
    if (k.site === "Internet Archive") {
      const m = obj(d.metadata);
      const id = str(m.identifier);
      const first = (v: unknown) => str(Array.isArray(v) ? v[0] : v);
      const title = first(m.title);
      return { found: title && id ? { medium: "film", title, ...by(first(m.creator) ?? first(m.director)), url: `https://archive.org/details/${id}`, image: `https://archive.org/services/img/${id}`, ...yr(first(m.year) ?? first(m.date)), source: k.site } : null };
    }
    if (k.site === "Art Institute of Chicago") {
      const a = obj(d.data);
      const title = str(a.title);
      return { found: title ? { medium: "art", title, ...by(str(a.artist_title)), url, ...img(aicImage(a.image_id)), ...yr(str(a.date_display)), source: k.site } : null };
    }
    const title = str(d.title);
    if (!title) return { found: null, error: `${k.site} gave no title for that link.` };
    // Spotify's oEmbed carries no maker; a SoundCloud title is "Track by Artist".
    const sc = k.site === "SoundCloud" ? /^(.*) by (.+)$/.exec(title) : null;
    return { found: { medium: k.medium, title: sc ? sc[1]! : title, ...by(sc ? sc[2] : str(d.author_name)), url, ...img(https(str(d.thumbnail_url))), source: k.site } };
  } catch (e) {
    return { found: null, error: `${k.site} could not be read: ${(e as Error).message}.` };
  }
}

const by = (v: string | undefined) => (v ? { by: v } : {});
const img = (v: string | undefined) => (v ? { image: v } : {});
const yr = (v: string | undefined) => {
  const m = v ? /\b(\d{4})\b/.exec(v) : null;
  return m ? { year: m[1]! } : {};
};

/** The Wikidata class a medium is searched under. */
const WIKIDATA_CLASS: Partial<Record<Medium, string>> = { film: "Q11424", theatre: "Q25379" };
export const SEARCHABLE: readonly Medium[] = ["film", "series", "theatre", "podcast", "music", "art", "game"];

/** "1957 film directed by Sidney Lumet" gives Sidney Lumet; "play by Arthur Miller" gives Arthur Miller. */
export function byFromDescription(d: string | undefined): string | undefined {
  const m = d ? /\b(?:directed |written |created )?by ([^,;(]+?)(?: and [^,;(]+)?\s*(?:$|[,;(])/i.exec(d) : null;
  return m ? m[1]!.trim() : undefined;
}

/** Films in the Internet Archive's feature films collection, free to watch in Reflect. */
export async function freeFilms(term: string, f: Fetch): Promise<Found[]> {
  const q = `title:(${term.replace(/[():"]/g, " ")}) AND collection:(feature_films) AND mediatype:(movies)`;
  const d = obj(await getJson(f, `https://archive.org/advancedsearch.php?output=json&rows=8&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=year&fl%5B%5D=creator&q=${enc(q)}`));
  return arr(obj(d.response).docs).map(obj).filter((x) => str(x.identifier) && str(x.title)).map((x) => {
    const id = str(x.identifier)!;
    const creator = Array.isArray(x.creator) ? str(x.creator[0]) : str(x.creator);
    return { medium: "film" as const, title: str(x.title)!, ...by(creator), url: `https://archive.org/details/${id}`, image: `https://archive.org/services/img/${id}`, ...yr(str(String(x.year ?? ""))), source: "Internet Archive" };
  });
}

/** Searches the catalog for a medium. Lectures, courses, debates and discussions are found by link instead. */
export async function searchMedium(medium: string, q: string, f: Fetch): Promise<Searched> {
  const term = q.trim().slice(0, 200);
  if (!term || !isMedium(medium)) return { found: [] };
  if (!SEARCHABLE.includes(medium)) return { found: [], error: "Paste its link (YouTube, Vimeo, Spotify or Apple Podcasts) to fill it in." };
  try {
    if (medium === "series") {
      const rows = arr(await getJson(f, `https://api.tvmaze.com/search/shows?q=${enc(term)}`));
      return { found: rows.slice(0, 10).map((r) => obj(obj(r).show)).filter((s) => str(s.name)).map((s) => ({ medium, title: str(s.name)!, ...by(str(obj(s.network).name) ?? str(obj(s.webChannel).name)), ...(https(str(s.url)) ? { url: https(str(s.url))! } : {}), ...img(https(str(obj(s.image).medium))), ...yr(str(s.premiered)), source: "TVmaze" })) };
    }
    if (medium === "game") {
      const d = obj(await getJson(f, `https://store.steampowered.com/api/storesearch/?cc=us&l=en&term=${enc(term)}`));
      return { found: arr(d.items).slice(0, 10).map(obj).filter((g) => str(g.name)).map((g) => ({ medium, title: str(g.name)!, url: `https://store.steampowered.com/app/${Number(g.id)}`, ...img(https(str(g.tiny_image))), source: "Steam" })) };
    }
    if (medium === "podcast") {
      const d = obj(await getJson(f, `https://itunes.apple.com/search?media=podcast&limit=10&term=${enc(term)}`));
      return { found: arr(d.results).map(obj).filter((p) => str(p.collectionName)).map((p) => ({ medium, title: str(p.collectionName)!, ...by(str(p.artistName)), ...(https(str(p.collectionViewUrl)) ? { url: https(str(p.collectionViewUrl))!.split("?")[0]! } : {}), ...img(https(str(p.artworkUrl600))), source: "Apple Podcasts" })) };
    }
    if (medium === "music") {
      const d = obj(await getJson(f, `https://musicbrainz.org/ws/2/release-group/?fmt=json&limit=10&query=${enc(term)}`));
      return {
        found: arr(d["release-groups"]).map(obj).filter((r) => str(r.title) && str(r.id)).map((r) => ({
          medium, title: str(r.title)!, ...by(arr(r["artist-credit"]).map((a) => str(obj(a).name)).filter(Boolean).join(", ") || undefined),
          url: `https://musicbrainz.org/release-group/${str(r.id)}`, image: `https://coverartarchive.org/release-group/${str(r.id)}/front-250`, ...yr(str(r["first-release-date"])), source: "MusicBrainz",
        })),
      };
    }
    if (medium === "art") {
      const d = obj(await getJson(f, `https://api.artic.edu/api/v1/artworks/search?limit=10&fields=id,title,artist_title,date_display,image_id&q=${enc(term)}`));
      return { found: arr(d.data).map(obj).filter((a) => str(a.title)).map((a) => ({ medium, title: str(a.title)!, ...by(str(a.artist_title)), url: `https://www.artic.edu/artworks/${Number(a.id)}`, ...img(aicImage(a.image_id)), ...yr(str(a.date_display)), source: "Art Institute of Chicago" })) };
    }
    const cls = WIKIDATA_CLASS[medium]!;
    const free = medium === "film" ? await freeFilms(term, f).catch(() => []) : [];
    const d = obj(await getJson(f, `https://www.wikidata.org/w/api.php?action=query&format=json&generator=search&gsrlimit=10&gsrsearch=${enc(`${term} haswbstatement:P31=${cls}`)}&prop=pageterms%7Cpageimages&wbptterms=label%7Cdescription&piprop=thumbnail&pithumbsize=300`));
    const pages = Object.values(obj(obj(d.query).pages)).map(obj).sort((a, b) => Number(a.index ?? 0) - Number(b.index ?? 0));
    return {
      found: pages.map((p): Found | null => {
        const terms = obj(p.terms);
        const label = str(arr(terms.label)[0]);
        const desc = str(arr(terms.description)[0]);
        return label ? { medium, title: label, ...by(byFromDescription(desc)), url: `https://www.wikidata.org/wiki/${str(p.title)}`, ...img(https(str(obj(p.thumbnail).source))), ...yr(desc), source: "Wikidata" } : null;
      }).filter((x): x is Found => x !== null).concat(free),
    };
  } catch (e) {
    if (medium === "film") {
      const free = await freeFilms(term, f).catch(() => []);
      if (free.length) return { found: free, error: `Wikidata could not be searched (${(e as Error).message}); these are free to watch at the Internet Archive.` };
    }
    return { found: [], error: `The catalog could not be searched: ${(e as Error).message}.` };
  }
}
