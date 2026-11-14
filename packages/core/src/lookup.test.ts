import { describe, expect, it } from "vitest";
import { byFromDescription, linkKind, lookLink, searchMedium, type Fetch } from "./lookup";

const answer = (body: unknown, ok = true): Fetch => async () => ({ ok, status: ok ? 200 : 503, json: async () => body });

describe("links", () => {
  it("knows each site and its medium, and refuses the rest", () => {
    expect(linkKind("https://www.youtube.com/watch?v=kBdfcR-8hEY")).toMatchObject({ medium: "lecture", site: "YouTube" });
    expect(linkKind("https://youtu.be/kBdfcR-8hEY")?.read).toContain("youtube.com/oembed");
    expect(linkKind("https://open.spotify.com/show/abc")?.medium).toBe("podcast");
    expect(linkKind("https://open.spotify.com/album/abc")?.medium).toBe("music");
    expect(linkKind("https://podcasts.apple.com/us/podcast/philosophize-this/id659155419")?.read).toBe("https://itunes.apple.com/lookup?id=659155419");
    expect(linkKind("https://store.steampowered.com/app/289070/Civ")?.medium).toBe("game");
    expect(linkKind("https://www.artic.edu/artworks/111628/nighthawks")?.medium).toBe("art");
    expect(linkKind("https://example.com/x")).toBeNull();
    expect(linkKind("javascript:alert(1)")).toBeNull();
  });
  it("reads a YouTube link through oEmbed", async () => {
    const r = await lookLink("https://www.youtube.com/watch?v=kBdfcR-8hEY", answer({ title: "Justice: Episode 01", author_name: "Harvard University", thumbnail_url: "https://i.ytimg.com/vi/x/hqdefault.jpg" }));
    expect(r.found).toEqual({ medium: "lecture", title: "Justice: Episode 01", by: "Harvard University", url: "https://www.youtube.com/watch?v=kBdfcR-8hEY", image: "https://i.ytimg.com/vi/x/hqdefault.jpg", source: "YouTube" });
  });
  it("splits a SoundCloud title, reads Steam and says why a link failed", async () => {
    expect((await lookLink("https://soundcloud.com/forss/flickermood", answer({ title: "Flickermood by Forss" }))).found).toMatchObject({ title: "Flickermood", by: "Forss" });
    expect((await lookLink("https://store.steampowered.com/app/289070", answer({ 289070: { data: { name: "Civilization VI", developers: ["Firaxis"], header_image: "https://x/h.jpg" } } }))).found).toMatchObject({ medium: "game", title: "Civilization VI", by: "Firaxis" });
    expect((await lookLink("https://vimeo.com/1", answer({}, false))).error).toContain("Vimeo could not be read");
    expect((await lookLink("https://example.com", answer({}))).error).toContain("Fill in the rest by hand");
  });
});

describe("search", () => {
  it("finds series in TVmaze and podcasts in Apple's search", async () => {
    const s = await searchMedium("series", "the wire", answer([{ show: { name: "The Wire", url: "https://www.tvmaze.com/shows/179", premiered: "2002-06-02", network: { name: "HBO" }, image: { medium: "https://static.tvmaze.com/x.jpg" } } }]));
    expect(s.found[0]).toMatchObject({ medium: "series", title: "The Wire", by: "HBO", year: "2002", source: "TVmaze" });
    const p = await searchMedium("podcast", "philosophize", answer({ results: [{ collectionName: "Philosophize This!", artistName: "Stephen West", collectionViewUrl: "https://podcasts.apple.com/us/podcast/x/id659155419?uo=4" }] }));
    expect(p.found[0]).toMatchObject({ title: "Philosophize This!", by: "Stephen West", url: "https://podcasts.apple.com/us/podcast/x/id659155419" });
  });
  it("finds films in Wikidata with the director from the description", async () => {
    const s = await searchMedium("film", "twelve angry men", answer({ query: { pages: { 1: { index: 1, title: "Q2345", terms: { label: ["12 Angry Men"], description: ["1957 film directed by Sidney Lumet"] } } } } }));
    expect(s.found[0]).toMatchObject({ title: "12 Angry Men", by: "Sidney Lumet", year: "1957", url: "https://www.wikidata.org/wiki/Q2345" });
    expect(byFromDescription("play by Arthur Miller")).toBe("Arthur Miller");
    expect(byFromDescription("1999 film")).toBeUndefined();
  });
  it("sends lectures to their link, and reports a failed catalog", async () => {
    expect((await searchMedium("lecture", "justice", answer({}))).error).toContain("Paste its link");
    expect((await searchMedium("music", "kind of blue", answer({}, false)))).toMatchObject({ found: [], error: expect.stringContaining("musicbrainz.org answered 503") });
  });
});

describe("Internet Archive", () => {
  it("reads a film link and finds free films beside Wikidata", async () => {
    const r = await lookLink("https://archive.org/details/nosferatu_1922", answer({ metadata: { identifier: "nosferatu_1922", title: "Nosferatu", creator: "F. W. Murnau", year: "1922" } }));
    expect(r.found).toMatchObject({ medium: "film", title: "Nosferatu", by: "F. W. Murnau", year: "1922", url: "https://archive.org/details/nosferatu_1922", image: "https://archive.org/services/img/nosferatu_1922" });
    const calls: string[] = [];
    const f: Fetch = async (u) => { calls.push(u); return { ok: !u.includes("wikidata"), status: u.includes("wikidata") ? 429 : 200, json: async () => ({ response: { docs: [{ identifier: "nosferatu_1922", title: "Nosferatu", year: 1922 }] } }) }; };
    const s = await searchMedium("film", "nosferatu", f);
    expect(s.found.map((x) => x.source)).toEqual(["Internet Archive"]);
    expect(s.error).toContain("free to watch");
  });
});
