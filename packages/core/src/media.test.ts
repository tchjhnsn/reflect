import { describe, expect, it } from "vitest";
import { clock, embedOf, readTranscript, searchTranscript, seconds } from "./media";

describe("media", () => {
  it("reads times both ways", () => {
    expect(seconds("1:02:03")).toBe(3723);
    expect(seconds("12:34")).toBe(754);
    expect(seconds("1m15s")).toBe(75);
    expect(seconds("90")).toBe(90);
    expect(seconds("x")).toBeNull();
    expect(clock(75)).toBe("1:15");
    expect(clock(3723)).toBe("1:02:03");
  });
  it("finds the player for each site", () => {
    expect(embedOf("https://www.youtube.com/watch?v=kBdfcR-8hEY&t=90")).toEqual({ kind: "youtube", id: "kBdfcR-8hEY", start: 90 });
    expect(embedOf("https://youtu.be/kBdfcR-8hEY")).toMatchObject({ id: "kBdfcR-8hEY" });
    expect(embedOf("https://vimeo.com/76979871")).toMatchObject({ src: "https://player.vimeo.com/video/76979871" });
    expect(embedOf("https://archive.org/details/nosferatu_1922")).toMatchObject({ src: "https://archive.org/embed/nosferatu_1922" });
    expect(embedOf("https://open.spotify.com/episode/abc123")).toMatchObject({ src: "https://open.spotify.com/embed/episode/abc123", tall: false });
    expect(embedOf("https://example.com/v")).toBeNull();
  });
  it("reads YouTube, inline and WebVTT transcripts, and searches them", () => {
    const yt = readTranscript("0:00\nGood morning.\n1:05\nThis is a course\nabout justice.\n");
    expect(yt).toEqual([{ at: 0, text: "Good morning." }, { at: 65, text: "This is a course about justice." }]);
    expect(readTranscript("0:05 hello there\n0:09 general")).toEqual([{ at: 5, text: "hello there" }, { at: 9, text: "general" }]);
    expect(readTranscript("WEBVTT\n\n1\n00:01:02.500 --> 00:01:04.000\nThe moral side\nof murder\n\n2\n00:01:05.000 --> 00:01:06.000\nPart two")).toEqual([{ at: 62, text: "The moral side of murder" }, { at: 65, text: "Part two" }]);
    expect(readTranscript("no times here")).toEqual([{ at: 0, text: "no times here" }]);
    expect(searchTranscript(yt, "COURSE justice").map((p) => p.at)).toEqual([65]);
  });
});
