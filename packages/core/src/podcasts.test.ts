import { describe, expect, it } from "vitest";
import { duration, episodeKey, episodeTranscript, feedOf, readFeed, showOf } from "./podcasts";

const FEED = `<?xml version="1.0"?><rss><channel><title>Philosophize This!</title>
<item><title><![CDATA[Episode #2 ... Plato]]></title><guid>ep-2</guid><pubDate>Tue, 27 Oct 2026 10:00:00 GMT</pubDate><itunes:duration>36:04</itunes:duration>
<enclosure url="https://cdn.example/ep2.mp3?a=1&amp;b=2" type="audio/mpeg"/><description>Plato &amp; the Forms</description>
<podcast:transcript url="https://cdn.example/ep2.vtt" type="text/vtt"/></item>
<item><title>Episode #1 ... Socrates</title><guid>ep-1</guid><pubDate>Mon, 20 Oct 2026 10:00:00 GMT</pubDate><itunes:duration>3600</itunes:duration><enclosure url="http://cdn.example/ep1.mp3" type="audio/mpeg"/></item>
<item><title>No audio</title></item>
</channel></rss>`;

describe("podcasts heard in Reflect", () => {
  it("reads a feed into episodes, newest first, with audio and a transcript when published", () => {
    const s = readFeed(FEED, "https://feed.example/rss");
    expect(s.title).toBe("Philosophize This!");
    expect(s.episodes.map((e) => [e.title, e.date, e.seconds])).toEqual([["Episode #2 ... Plato", "2026-10-27", 2164], ["Episode #1 ... Socrates", "2026-10-20", 3600]]);
    expect(s.episodes[0]).toMatchObject({ audio: "https://cdn.example/ep2.mp3?a=1&b=2", about: "Plato & the Forms", transcript: { url: "https://cdn.example/ep2.vtt", type: "text/vtt" }, key: episodeKey("ep-2") });
    expect(s.episodes[1]!.audio).toBe("https://cdn.example/ep1.mp3");
    expect(episodeKey("ep-2")).not.toBe(episodeKey("ep-1"));
    expect(duration("1:00:00")).toBe(3600);
    expect(duration("nonsense")).toBeNull();
  });

  it("finds a show's feed from its Apple Podcasts link and reads its transcript", async () => {
    const pages: Record<string, string> = {
      "https://itunes.apple.com/lookup?id=659155419&entity=podcast": JSON.stringify({ results: [{ feedUrl: "https://feed.example/rss" }] }),
      "https://feed.example/rss": FEED,
      "https://cdn.example/ep2.vtt": "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nWelcome.\n",
    };
    const f = async (u: string) => pages[u] ?? null;
    expect(await feedOf("https://podcasts.apple.com/us/podcast/philosophize-this/id659155419", f)).toBe("https://feed.example/rss");
    const s = await showOf("https://podcasts.apple.com/us/podcast/philosophize-this/id659155419", f);
    expect(s?.episodes).toHaveLength(2);
    expect(await episodeTranscript(s!.episodes[0]!, f)).toEqual([{ at: 1, heading: false, who: null, text: "Welcome." }]);
    expect(await showOf("https://example.com/not-a-feed", f)).toBeNull();
    expect(await feedOf("javascript:alert(1)", f)).toBeNull();
  });
});
