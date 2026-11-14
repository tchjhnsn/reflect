import { describe, expect, it } from "vitest";
import { csvRows, fromLetterboxd, fromLines, fromOpml, fromSpotify, fromYoutube, notKept, readImport } from "./item-imports";

describe("imports", () => {
  it("reads CSV with quotes, commas and line breaks", () => {
    expect(csvRows('a,"b, c","d ""e"""\n"x\ny",2\n')).toEqual([["a", "b, c", 'd "e"'], ["x\ny", "2"]]);
  });
  it("reads Letterboxd watched and diary files, once per film", () => {
    const w = fromLetterboxd("Date,Name,Year,Letterboxd URI\n2024-01-02,12 Angry Men,1957,https://boxd.it/x\n2024-02-02,12 Angry Men,1957,https://boxd.it/x\n");
    expect(w).toEqual([{ medium: "film", title: "12 Angry Men", note: "1957", url: "https://boxd.it/x", status: "done", on: "2024-01-02" }]);
    expect(fromLetterboxd("Date,Name,Year,Letterboxd URI,Rating,Rewatch,Tags,Watched Date\n2024-03-01,Ran,1985,https://boxd.it/r,5,,,2024-02-28\n")[0]!.on).toBe("2024-02-28");
    expect(() => fromLetterboxd("a,b\n1,2")).toThrow("not a Letterboxd export");
  });
  it("reads OPML subscriptions", () => {
    expect(fromOpml('<opml><body><outline text="Philosophize This!" type="rss" xmlUrl="https://f/x" htmlUrl="https://p.com"/><outline text="Group"/></body></opml>')).toEqual([{ medium: "podcast", title: "Philosophize This!", url: "https://p.com", status: "doing" }]);
  });
  it("reads YouTube history once per video, with the channel", () => {
    const y = fromYoutube(JSON.stringify([{ title: "Watched Justice Ep 1", titleUrl: "https://www.youtube.com/watch?v=a", subtitles: [{ name: "Harvard University" }], time: "2024-05-01T10:00:00Z" }, { title: "Watched Justice Ep 1", titleUrl: "https://www.youtube.com/watch?v=a" }, { title: "Watched a video that has been removed" }]));
    expect(y).toEqual([{ medium: "lecture", title: "Justice Ep 1", by: "Harvard University", url: "https://www.youtube.com/watch?v=a", status: "done", on: "2024-05-01" }]);
  });
  it("tallies Spotify by track and show, most listened first", () => {
    const s = fromSpotify(JSON.stringify([{ endTime: "2024-01-01 10:00", artistName: "Miles Davis", trackName: "So What", msPlayed: 120000 }, { endTime: "2024-01-02 10:00", artistName: "Miles Davis", trackName: "So What", msPlayed: 300000 }, { ts: "2024-01-03T00:00:00Z", episode_name: "Ep", episode_show_name: "Philosophize This!", ms_played: 60000 }]));
    expect(s).toEqual([{ medium: "music", title: "So What", by: "Miles Davis", status: "done", on: "2024-01-02", note: "7 min" }, { medium: "podcast", title: "Philosophize This!", status: "doing", on: "2024-01-03", note: "1 min" }]);
  });
  it("reads lists, Steam, and leaves out what is kept", () => {
    expect(fromLines("# games\nChess | anyone\nchess\nGo", "game")).toEqual([{ medium: "game", title: "Chess", by: "anyone", status: "done" }, { medium: "game", title: "Go", status: "done" }]);
    expect(readImport("steam", "Civilization VI\n")[0]!.medium).toBe("game");
    expect(notKept([{ medium: "game", title: "Go!", status: "done" }, { medium: "film", title: "Go", status: "done" }], [{ medium: "game", title: "go" }]).map((c) => c.medium)).toEqual(["film"]);
  });
});
